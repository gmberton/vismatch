import torch
import torchvision.transforms as tfm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path, patch_sample_keypoints_device

add_to_path(THIRD_PARTY_DIR.joinpath("SANDesc"))
add_to_path(THIRD_PARTY_DIR.joinpath("LightGlue"))
add_to_path(THIRD_PARTY_DIR.joinpath("DeDoDe"))

from model.network_descriptor import SANDesc
from utils.utils_2D import grid_sample_nan
from utils.utils_matches import match_descriptors_mnn_scores_ratio_test
from lightglue import ALIKED
from DeDoDe import dedode_detector_L
from DeDoDe import utils as dedode_utils
from DeDoDe.detectors import dedode_detector as dedode_detector_module

patch_sample_keypoints_device(dedode_utils, dedode_detector_module)


class SANDescMatcher(BaseMatcher):
    """SANDesc descriptors at ALIKED or DeDoDe keypoints, matched with mutual nearest neighbours, as in SANDesc's
    evaluation code (PoseBench)."""

    border = 16  # PoseBench drops the keypoints this close to the image border

    def __init__(
        self, device="cpu", max_num_keypoints=2048, detector="aliked", min_score=0.5, ratio_test=0.98, *args, **kwargs
    ):
        super().__init__(device, **kwargs)
        assert detector in ("aliked", "dedode"), f"Unknown SANDesc detector {detector}"
        self.detector_name, self.max_num_keypoints = detector, max_num_keypoints
        self.min_score, self.ratio_test = min_score, ratio_test
        self.net = SANDesc(skip_connection=True, spatial_attention=True, third_block=True)
        self.net.load_state_dict(load_file(hf_hub_download("vismatch/sandesc", f"sandesc_{detector}.safetensors")))
        self.net = self.net.eval().to(self.device)
        if detector == "aliked":
            # the top max_num_keypoints, as the official ALIKED with top_k that PoseBench runs
            self.detector = ALIKED(
                model_name="aliked-n16rot", max_num_keypoints=max_num_keypoints, detection_threshold=-1
            )
        else:
            # PoseBench's dedode runs DeDoDe's v1 detector on ImageNet-normalized images
            weights = load_file(hf_hub_download("vismatch/dedode", "dedode_detector_L.safetensors"))
            self.detector = dedode_detector_L(weights=weights, device=self.device)
            self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.detector = self.detector.eval().to(self.device)

    def _detect(self, img):
        """Keypoints in vismatch pixels and in the pixels where PoseBench filters the border and samples SANDesc."""
        wh = torch.tensor(img.shape[-2:][::-1], device=img.device)
        if self.detector_name == "aliked":
            kpts = self.detector.extract(img, resize=None)["keypoints"][0]
            return kpts, kpts * wh / (wh - 1)  # PoseBench maps ALIKED's [-1, 1] keypoints with W * (x + 1) / 2
        kpts = self.detector.detect({"image": self.normalize(img)[None]}, num_keypoints=self.max_num_keypoints)
        kpts = (kpts["keypoints"][0] + 1) / 2 * wh
        return kpts - 0.5, kpts

    def _extract(self, img):
        img = img[:, : img.shape[1] // 16 * 16, : img.shape[2] // 16 * 16]  # the UNet needs sides divisible by 16
        kpts, kpts_pb = self._detect(img)
        wh = torch.tensor(img.shape[-2:][::-1], device=img.device)
        valid = ((kpts_pb > self.border) & (kpts_pb < wh - self.border)).all(-1)
        kpts, kpts_pb = kpts[valid], kpts_pb[valid]
        desc = grid_sample_nan(kpts_pb[None], self.net(img[None]), mode="nearest")[0][0].T
        return kpts, desc

    def _forward(self, img0, img1):
        kpts0, desc0 = self._extract(img0)
        kpts1, desc1 = self._extract(img1)
        matches, scores = match_descriptors_mnn_scores_ratio_test([desc0], [desc1], self.min_score, self.ratio_test)
        m0, m1 = matches[0][:, 0], matches[0][:, 1]
        return kpts0[m0], kpts1[m1], kpts0, kpts1, desc0, desc1, scores[0][m0, m1]
