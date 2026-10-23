import tempfile
from pathlib import Path

import torch
import torchvision.transforms as tfm
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Normalize

from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path, disable_xformers, patch_sample_keypoints_device

add_to_path(THIRD_PARTY_DIR.joinpath("dad"))
add_to_path(THIRD_PARTY_DIR.joinpath("RoMa"))

from dad.detectors.dedode_detector import dedode_detector_S, DaD
from dad.detectors import dedode_detector as dad_detector_module
from dad import utils as dad_utils
from romatch import roma_outdoor

patch_sample_keypoints_device(dad_utils, dad_detector_module)


class DaDMatcher(BaseMatcher):
    def __init__(
        self,
        device="cpu",
        max_num_keypoints=2048,
        *args,
        **kwargs,
    ):
        super().__init__(device, **kwargs)
        self.max_keypoints = max_num_keypoints
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self._get_detector()
        self._get_matcher()

        if "cuda" not in str(self.device):
            disable_xformers()

    def _get_detector(self):
        encoder, decoder = dedode_detector_S()
        self.detector = DaD(
            encoder,
            decoder,
            resize=1024,
            nms_size=3,
            remove_borders=True,
            increase_coverage=False,
            subpixel=True,
            subpixel_temp=0.5,
            keep_aspect_ratio=True,
        )

        weights = torch.hub.load_state_dict_from_url(
            "https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth",
            map_location="cpu",
            weights_only=True,
        )
        self.detector.load_state_dict(weights)
        self.detector.eval().to(self.device)

        # Disable amp on non-CUDA devices to prevent float16 CPU autocast failure
        if "cuda" not in str(self.device):
            for m in self.detector.modules():
                if hasattr(m, "amp"):
                    m.amp = False

        return self.detector

    def _get_matcher(self):
        amp_dtype = torch.float16 if "cuda" in str(self.device) else torch.float32
        self.matcher = roma_outdoor(device=self.device, amp_dtype=amp_dtype)
        self.matcher.train(False)
        self.matcher.symmetric = False
        return self.matcher

    def _prepare_detector_input(self, img_tensor: torch.Tensor):
        _, H, W = img_tensor.shape
        scale = 1024 / max(W, H)
        new_w = int((scale * W) // 8 * 8)
        new_h = int((scale * H) // 8 * 8)

        resized = tfm.functional.resize(img_tensor, [new_h, new_w], antialias=True)
        normalized = self.normalizer(resized).unsqueeze(0).to(self.device)
        return {"image": normalized}, (H, W)

    def _save_temp_image(self, img: torch.Tensor):
        temp = tempfile.NamedTemporaryFile("w+b", suffix=".png", delete=False)
        to_pil_image(img.cpu()).save(temp.name, format="png")
        return temp

    def _forward(self, img0, img1):
        batch0, (h0, w0) = self._prepare_detector_input(img0)
        batch1, (h1, w1) = self._prepare_detector_input(img1)

        det0 = self.detector.detect(batch0, num_keypoints=self.max_keypoints)
        det1 = self.detector.detect(batch1, num_keypoints=self.max_keypoints)

        kpts0_norm = det0["keypoints"][0]  # (N, 2) in [-1, 1]
        kpts1_norm = det1["keypoints"][0]  # (N, 2) in [-1, 1]

        all_kpts0 = self.detector.to_pixel_coords(kpts0_norm, h0, w0)
        all_kpts1 = self.detector.to_pixel_coords(kpts1_norm, h1, w1)

        img0_temp = self._save_temp_image(img0)
        img1_temp = self._save_temp_image(img1)

        warp, certainty = self.matcher.match(img0_temp.name, img1_temp.name, batched=False, device=self.device)

        img0_temp.close(), img1_temp.close()
        Path(img0_temp.name).unlink()
        Path(img1_temp.name).unlink()

        matches = self.matcher.match_keypoints(
            kpts0_norm,
            kpts1_norm,
            warp,
            certainty,
            return_tuple=False,
        )

        if matches is None or len(matches) == 0:
            mkpts0 = torch.empty((0, 2), device=self.device)
            mkpts1 = torch.empty((0, 2), device=self.device)
        else:
            mkpts0, mkpts1 = self.matcher.to_pixel_coordinates(matches, h0, w0, h1, w1)

        return mkpts0, mkpts1, all_kpts0, all_kpts1, None, None, None
