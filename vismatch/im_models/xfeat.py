import torch
import kornia.feature as KF
from huggingface_hub import snapshot_download

from vismatch import BaseMatcher


class xFeatMatcher(BaseMatcher):
    """XFeat from kornia: sparse (mutual nearest neighbours), semi-dense (XFeat*) or LighterGlue matching."""

    def __init__(self, device="cpu", max_num_keypoints=4096, mode="sparse", *args, **kwargs):
        super().__init__(device, **kwargs)
        assert mode in ["sparse", "semi-dense", "lighterglue"], (
            f'unsupported mode for xfeat: {mode}. Must choose from ["sparse", "semi-dense", "lighterglue"]'
        )
        self.max_num_keypoints = max_num_keypoints
        self.mode = mode

        self.model = KF.XFeat(top_k=max_num_keypoints)
        weights = torch.load(f"{snapshot_download('vismatch/xfeat')}/xfeat.pt", map_location="cpu", weights_only=True)
        self.model.net.load_state_dict(weights)
        self.model = self.model.eval().to(self.device)

        if self.mode == "lighterglue":
            self.lighterglue = KF.LightGlueMatcher("xfeat").eval().to(self.device)

    def _forward(self, img0, img1):
        img0, img1 = img0[None], img1[None]

        if self.mode == "semi-dense":
            # the steps of XFeat.match_xfeat_star, keeping the dense features it does not return
            feats0 = self.model.detectAndComputeDense(img0)
            feats1 = self.model.detectAndComputeDense(img1)
            idxs = self.model._batch_match(feats0["descriptors"], feats1["descriptors"])
            matches = self.model._refine_matches(feats0, feats1, matches=idxs, batch_idx=0)
            mkpts0, mkpts1 = matches[:, :2], matches[:, 2:]
            kpts0, kpts1 = feats0["keypoints"][0], feats1["keypoints"][0]
            desc0, desc1 = feats0["descriptors"][0], feats1["descriptors"][0]
            return mkpts0, mkpts1, kpts0, kpts1, desc0, desc1, None

        feats0 = self.model.detectAndCompute(img0)[0]
        feats1 = self.model.detectAndCompute(img1)[0]
        kpts0, kpts1 = feats0["keypoints"], feats1["keypoints"]
        desc0, desc1 = feats0["descriptors"], feats1["descriptors"]

        if self.mode == "lighterglue":
            lafs0 = KF.laf_from_center_scale_ori(kpts0[None])
            lafs1 = KF.laf_from_center_scale_ori(kpts1[None])
            scores, idxs = self.lighterglue(
                desc0, desc1, lafs0, lafs1, hw1=tuple(img0.shape[-2:]), hw2=tuple(img1.shape[-2:])
            )
            mconf = scores[:, 0]
        else:  # sparse
            _, idxs = KF.match_mnn(desc0, desc1)
            mconf = None

        return kpts0[idxs[:, 0]], kpts1[idxs[:, 1]], kpts0, kpts1, desc0, desc1, mconf
