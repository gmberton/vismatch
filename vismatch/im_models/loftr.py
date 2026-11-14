import torch
from huggingface_hub import hf_hub_download
from kornia.feature import LoFTR
import torchvision.transforms as tfm
from vismatch import BaseMatcher


class LoftrMatcher(BaseMatcher):
    def __init__(self, device="cpu", *args, **kwargs):
        super().__init__(device, **kwargs)

        self.model = LoFTR(pretrained=None).to(self.device)
        weights = torch.load(
            hf_hub_download("kornia/loftr", "loftr_outdoor.ckpt"), map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(weights["state_dict"])
        # kornia >= 0.9 builds LoFTR(pretrained=None) in training mode, where coarse matching needs ground truth
        self.model = self.model.eval()

    def preprocess(self, img):
        return tfm.Grayscale()(img).unsqueeze(0).to(self.device)

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)
        batch = {"image0": img0, "image1": img1}

        output = self.model(batch)
        mkpts0, mkpts1, mconf = output["keypoints0"], output["keypoints1"], output["confidence"]

        return mkpts0, mkpts1, None, None, None, None, mconf
