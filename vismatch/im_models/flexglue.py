from vismatch import BaseMatcher, THIRD_PARTY_DIR
from vismatch.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("FlexGlue"))
add_to_path(THIRD_PARTY_DIR.joinpath("RaCo"))

from flexglue import FlexGlue
from flexglue.utils import pack_matches, rbd
from raco import RaCo


class FlexGlueMatcher(BaseMatcher):
    """FlexGlue sampling matches without keypoints, or matching RaCo keypoints as recommended in FlexGlue's demo."""

    def __init__(self, device="cpu", max_num_keypoints=2048, detector=None, *args, **kwargs):
        super().__init__(device, **kwargs)
        assert detector in (None, "raco"), f"Unknown FlexGlue detector {detector}"
        # FlexGlue's default float16 transformer layers only pay off on cuda
        mp = "float16" if "cuda" in self.device else None
        self.matcher = FlexGlue(mp=mp, max_num_samples=max_num_keypoints).eval().to(self.device)
        self.detector = None
        if detector == "raco":
            # the ranker and covariance heads are not used for matching: disabling them leaves keypoints unchanged
            self.detector = RaCo(max_num_keypoints=max_num_keypoints, ranker=False, covariance_estimator=False)
            self.detector = self.detector.eval().to(self.device)

    def _forward(self, img0, img1):
        kpts0 = kpts1 = None
        if self.detector is not None:
            kpts0, kpts1 = self.detector.extract(img0)["keypoints"], self.detector.extract(img1)["keypoints"]
        data0, data1 = self.matcher.descriptor.describe(img0, kpts0), self.matcher.descriptor.describe(img1, kpts1)
        pred = rbd(pack_matches(self.matcher.match(data0, data1)))
        # FlexGlue puts the top-left pixel center at (0.5, 0.5) (COLMAP), vismatch at (0, 0)
        mkpts0 = pred["keypoints0"][pred["matches"][:, 0]] - 0.5
        mkpts1 = pred["keypoints1"][pred["matches"][:, 1]] - 0.5
        if self.detector is None:
            return mkpts0, mkpts1, None, None, None, None, pred["scores"]
        desc0, desc1 = data0["descriptors"][0], data1["descriptors"][0]
        return mkpts0, mkpts1, kpts0[0] - 0.5, kpts1[0] - 0.5, desc0, desc1, pred["scores"]
