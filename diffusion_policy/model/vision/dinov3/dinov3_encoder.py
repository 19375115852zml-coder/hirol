import torch
import torch.nn as nn
from .backbones import dinov3_vits16 
from PIL import Image


class DinoV3ImageEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.model = config.get("dinov3_model", "dinov3_vits16")
        self.encoder_dict={
            "dinov3_vits16" : dinov3_vits16 
        }
        if self.model not in self.encoder_dict:
            raise ValueError
        self.dino_param = config.get("dino_param",{})  
        backbone_fn = self.encoder_dict[f"{self.model}"]
        self.backbone = backbone_fn(**self.dino_param) # **self.param 将字典展开为关键字参数
        self.feature_dim = self.backbone.num_features
        if self.feature_dim is None:
            self.feature_dim = self.backbone.embed_dim

    def forward(self, img):
        feature = self.backbone(img)
        if not isinstance(feature, torch.Tensor):
            raise TypeError("backbone output must be a torch.Tensor")
        print(f"feature shape: {feature.shape}")
        if feature.nidm != 2:
            raise ValueError("expect feature dim 2 but get {feature_dim}")
        if feature.shape[-1] != self.feature_dim:
            raise ValueError("expect {self.feature_dim} but get {feature.shape}")
        return feature
    
    

if __name__ == "__main__" :
    from img_loader import Image_loader
    img_path = "data/train_episode/pick_and_place/pick_and_place_hirol/episode_0001/colors/000000_ee_cam_color.jpg"
    image_loader = Image_loader
    image 
    
    
    
    
    
    encoder = DinoV3ImageEncoder
    img = encoder.forward()