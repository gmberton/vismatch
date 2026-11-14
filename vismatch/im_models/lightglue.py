import torch
import kornia.feature as KF
from kornia.geometry.transform import resize

from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("LightGlue"))

# superpoint-, sift- and doghardnet-lightglue keep the official LightGlue implementation: kornia has no SuperPoint,
# and the official SIFT and DoGHardNet detect with OpenCV's SIFT, which the LightGlue weights were trained on.
from lightglue import match_pair
from lightglue import LightGlue, SuperPoint, SIFT, DoGHardNet


class LightGlueBase(BaseMatcher):
    """
    This class is the parent for all methods that use LightGlue as a matcher,
    with different local features. It implements the forward which is the same
    regardless of the feature extractor of choice.
    Therefore this class should *NOT* be instatiated, as it needs its children to define
    the extractor and the matcher.
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

    def _forward(self, img0, img1):
        """Run matching using LightGlue. "extractor" and "matcher" are instantiated by subclasses."""
        feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, img0, img1, device=self.device)
        kpts0, kpts1, matches = (
            feats0["keypoints"],
            feats1["keypoints"],
            matches01["matches"],
        )

        desc0 = feats0["descriptors"]
        desc1 = feats1["descriptors"]

        mkpts0, mkpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        mconf = matches01.get("scores")

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1, mconf


class KorniaLightGlueBase(BaseMatcher):
    """LightGlue from kornia, matching the features of a kornia extractor.

    Children set ``features`` (the LightGlue weights to load) and ``self.extractor``, and implement ``_detect``.
    As in the official LightGlue ``extract``, each image is resized so that its long side is ``resize`` pixels
    before detection and matching, and the keypoints are mapped back to the input image afterwards.
    """

    features: str
    resize = 1024

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)
        self.matcher = KF.LightGlueMatcher(self.features, {"depth_confidence": -1, "width_confidence": -1})
        self.matcher = self.matcher.eval().to(self.device)

    def _detect(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the LAFs (1, N, 2, 3) and descriptors (N, D) of a (1, 3, H, W) image."""
        raise NotImplementedError

    def _resize(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = img.shape[-2:]
        img = resize(img[None], self.resize, side="long", antialias=True)
        scale = torch.tensor([img.shape[-1] / w, img.shape[-2] / h], device=img.device)
        return img, scale

    def _forward(self, img0, img1):
        img0, scale0 = self._resize(img0)
        img1, scale1 = self._resize(img1)
        lafs0, desc0 = self._detect(img0)
        lafs1, desc1 = self._detect(img1)

        scores, idxs = self.matcher(desc0, desc1, lafs0, lafs1, hw1=tuple(img0.shape[-2:]), hw2=tuple(img1.shape[-2:]))

        kpts0 = (KF.get_laf_center(lafs0)[0] + 0.5) / scale0 - 0.5
        kpts1 = (KF.get_laf_center(lafs1)[0] + 0.5) / scale1 - 0.5
        mkpts0, mkpts1 = kpts0[idxs[:, 0]], kpts1[idxs[:, 1]]

        return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1, scores[:, 0]


class SiftLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="sift", depth_confidence=-1, width_confidence=-1).to(self.device)


class SuperpointLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).to(self.device)


class DiskLightGlue(KorniaLightGlueBase):
    features = "disk"

    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.max_num_keypoints = max_num_keypoints
        self.extractor = KF.DISK.from_pretrained("depth").eval().to(self.device)

    def _detect(self, img):
        feats = self.extractor(img, n=self.max_num_keypoints, window_size=5, pad_if_not_divisible=True)[0]
        return KF.laf_from_center_scale_ori(feats.keypoints[None].float()), feats.descriptors


class AlikedLightGlue(KorniaLightGlueBase):
    features = "aliked"

    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        # also used by aliked-lightglue-subpx, which reads the score map through extractor.extract_dense_map
        self.extractor = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=max_num_keypoints)
        self.extractor = self.extractor.eval().to(self.device)

    def _detect(self, img):
        feats = self.extractor(img)[0]
        return KF.laf_from_center_scale_ori(feats.keypoints[None]), feats.descriptors


class DognetLightGlue(LightGlueBase):
    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="doghardnet", depth_confidence=-1, width_confidence=-1).to(self.device)
