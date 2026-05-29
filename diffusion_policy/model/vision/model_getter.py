import torch
import torchvision



def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

def get_dino_v3(
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        pooling="pooler",
        freeze=False,
        output_dim=None,
        trust_remote_code=False,
        interpolate_pos_encoding=True,
        **kwargs):
    from diffusion_policy.model.vision.dinov3_encoder import Dinov3Encoder
    return Dinov3Encoder(
        model_name=model_name,
        pooling=pooling,
        freeze=freeze,
        output_dim=output_dim,
        trust_remote_code=trust_remote_code,
        interpolate_pos_encoding=interpolate_pos_encoding,
    )
