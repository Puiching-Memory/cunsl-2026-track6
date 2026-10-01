"""Strict local loading and explicit token pooling for reviewed DINOv3 releases."""
import importlib.metadata
import json
from pathlib import Path

from .data import sha256

POOLING = ('cls', 'avg', 'foreground', 'cls_foreground')

def reviewed_resource(path, weights):
    resource = json.loads(Path(path).read_text(encoding='utf-8'))
    if not (resource.get('public_before_cutoff_reviewed')
            and resource['code'].get('license_reviewed')
            and resource['weights'].get('license_reviewed')):
        raise ValueError('Individual date, code and weight license review required')
    if resource['weights']['public_url_present_by'] >= resource['cutoff']:
        raise ValueError('Weights must predate competition cutoff')
    if not resource['weights'].get('sha256') or sha256(weights) != resource['weights']['sha256']:
        raise ValueError('Exact reviewed weight SHA-256 required')
    if Path(weights).stat().st_size != resource['weights']['bytes']:
        raise ValueError('Weight size differs from reviewed release')
    if resource.get('backend') != 'timm':
        raise ValueError('Expected reviewed timm resource')
    return resource

def load_encoder(resource_path, weights, image_size=256):
    resource = reviewed_resource(resource_path, weights)
    if importlib.metadata.version('timm') != resource['code']['version']:
        raise ValueError('Installed timm differs from reviewed exact version')
    if image_size < 128 or image_size % 16:
        raise ValueError('DINOv3 input size must be >=128 and divisible by 16')
    import timm
    from safetensors.torch import load_file
    model = timm.create_model(resource['variant'], pretrained=False, num_classes=0,
                              img_size=image_size)
    # No hub download, pickle execution, key dropping or automatic weight conversion.
    model.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
    return model, resource

def preprocess(pixels, encoder, image_size, *, augment=False):
    import torch
    import torch.nn.functional as F
    x = pixels.permute(0, 3, 1, 2).to(device='cuda', dtype=torch.float32) / 255.
    if tuple(x.shape[-2:]) != (image_size, image_size):
        x = F.interpolate(x, size=(image_size,image_size), mode='bicubic',
                          align_corners=False, antialias=True).clamp(0,1)
    if augment:
        # Preserve the complete hull: independent flips/180deg rotation, no random crop.
        horizontal = torch.rand(len(x), device=x.device) < .5
        vertical = torch.rand(len(x), device=x.device) < .5
        x = torch.where(horizontal[:,None,None,None], x.flip(-1), x)
        x = torch.where(vertical[:,None,None,None], x.flip(-2), x)
    foreground = (x.amax(dim=1, keepdim=True) > 8/255).float()
    cfg = encoder.pretrained_cfg
    mean = x.new_tensor(cfg['mean']).view(1,3,1,1)
    std = x.new_tensor(cfg['std']).view(1,3,1,1)
    return (x-mean)/std, foreground

def pool_tokens(tokens, encoder, foreground, method):
    import torch
    import torch.nn.functional as F
    if method not in POOLING:
        raise ValueError('Unknown token pooling')
    patches = tokens[:,encoder.num_prefix_tokens:]
    cls = tokens[:,0].float()
    if method == 'cls':
        return cls
    if method == 'avg':
        return patches.float().mean(dim=1)
    # Soft occupancy weights avoid letting the zero-padded background dominate pooling.
    side = int(patches.shape[1] ** .5)
    if side*side != patches.shape[1]:
        raise ValueError('Foreground pooling requires square patch grid')
    weights = F.adaptive_avg_pool2d(foreground, (side,side)).flatten(2).transpose(1,2)
    count = weights.sum(dim=1)
    pooled = (patches.float()*weights).sum(dim=1)/count.clamp_min(1e-6)
    # Keep official constant test images and produce a finite embedding.
    pooled = torch.where(count>0, pooled, patches.float().mean(dim=1))
    if method == 'foreground':
        return pooled
    return torch.cat([F.normalize(cls,dim=1),F.normalize(pooled,dim=1)],dim=1)

def encoder_metadata(encoder, resource, image_size, pooling):
    cfg = encoder.pretrained_cfg
    return {'variant':resource['variant'], 'backend':'timm',
            'timm':importlib.metadata.version('timm'),
            'safetensors':importlib.metadata.version('safetensors'),
            'weights_sha256':resource['weights']['sha256'],
            'resource_code_commit':resource['code']['commit'],
            'input_size':image_size, 'pooling':pooling,
            'mean':list(cfg['mean']), 'std':list(cfg['std']),
            'input':'whole RGB uint8 image; no center/random crop; per-release normalization',
            'foreground_rule':'max RGB >8/255; soft patch occupancy; empty-mask fallback to mean',
            'conversion_behavior':'default timm float32 RoPE periods; zero QKV biases absent',
            'parameters':sum(p.numel() for p in encoder.parameters())}
