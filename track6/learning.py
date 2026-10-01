"""Image-level cross-modal ReID with a reviewed encoder and identity supervision."""
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import pool_tokens, preprocess

STEM_MODES = ('frozen', 'shared', 'separate')


def structure_energy(patch_tokens, height, width, eps=1e-5):
    """SDF-inspired adjacent gradient energies, standardized per image.

    Patch tokens only: CLS/register tokens have no place on the spatial grid.
    This is a statistical constraint, not a physical segmentation or alignment.
    """
    if patch_tokens.ndim!=3 or patch_tokens.shape[1]!=height*width or min(height,width)<2:
        raise ValueError('A complete two-dimensional patch grid is required')
    grid = patch_tokens.float().reshape(len(patch_tokens),height,width,-1)
    energy = (grid[:,1:]-grid[:,:-1]).abs().mean(dim=(1,2))
    energy = energy+(grid[:,:,1:]-grid[:,:,:-1]).abs().mean(dim=(1,2))
    return (energy-energy.mean(dim=1,keepdim=True))/torch.sqrt(
        energy.var(dim=1,unbiased=False,keepdim=True)+eps)


def structural_pair_loss(descriptors, labels, modalities):
    """The current sampler has one optical and one SAR image per identity."""
    opt, sar = descriptors[modalities==0], descriptors[modalities==1]
    if len(opt)<2 or len(opt)!=len(sar) or not torch.equal(labels[modalities==0],labels[modalities==1]):
        raise ValueError('Balanced, ordered optical/SAR identity pairs required')
    if len(torch.unique(labels[modalities==0])) != len(opt):
        raise ValueError('Distinct identities required in each pair batch')
    return F.mse_loss(opt,sar)


class SensorReID(nn.Module):
    def __init__(self, encoder, identities, image_size, pooling, train_last_blocks,
                 stem_mode='frozen', structural_layer=6):
        super().__init__()
        if not 0 <= train_last_blocks <= len(encoder.blocks) or identities < 2:
            raise ValueError('Valid block budget and at least two training identities required')
        self.encoder = encoder
        self.image_size = image_size
        self.pooling = pooling
        if stem_mode not in STEM_MODES:
            raise ValueError('Unknown patch stem mode')
        if stem_mode!='frozen' and not all(hasattr(encoder,k) for k in ['patch_embed','_pos_embed','norm_pre']):
            raise ValueError('Sensor stems require the reviewed timm EVA encoder')
        if stem_mode!='frozen' and getattr(encoder,'rope_mixed',False):
            raise ValueError('Mixed rotary embeddings are outside the reviewed encoder scope')
        self.stem_mode = stem_mode
        self.structural_layer = structural_layer
        dim = encoder.num_features * (2 if pooling=='cls_foreground' else 1)
        self.dimension = dim
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        if train_last_blocks:
            for block in encoder.blocks[-train_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in encoder.norm.parameters():
                parameter.requires_grad_(True)
        if stem_mode!='frozen':
            for parameter in encoder.patch_embed.parameters():
                parameter.requires_grad_(True)
        # Copy the exact reviewed initialization; no external optical/SAR weights.
        if stem_mode=='separate':
            self.sar_patch_embed = deepcopy(encoder.patch_embed)
        self.heads = nn.ModuleList([nn.Linear(dim,dim,bias=False) for _ in range(2)])
        with torch.no_grad():
            for head in self.heads:
                head.weight.copy_(torch.eye(dim))
        self.classifier = nn.Linear(dim,identities,bias=False)
        nn.init.normal_(self.classifier.weight, std=.01)

    def encoder_tokens(self, images, modalities, *, return_structure=False):
        if self.stem_mode=='frozen' and not return_structure:
            return self.encoder.forward_features(images),None
        if return_structure and not 1<=self.structural_layer<=len(self.encoder.blocks):
            raise ValueError('Structural extraction layer must be within the encoder')
        patches = self.encoder.patch_embed(images)
        if self.stem_mode=='separate':
            sar = self.sar_patch_embed(images)
            patches = torch.where((modalities==0)[:,None,None],patches,sar)
        tokens,rope = self.encoder._pos_embed(patches)
        tokens = self.encoder.norm_pre(tokens)
        descriptor = None
        for index,block in enumerate(self.encoder.blocks,1):
            tokens = block(tokens,rope=rope)
            if return_structure and index==self.structural_layer:
                grid = self.encoder.patch_embed.grid_size
                descriptor = structure_energy(tokens[:,self.encoder.num_prefix_tokens:],*grid)
        return self.encoder.norm(tokens),descriptor

    def encode(self, pixels, modalities, *, augment=False, return_structure=False):
        x,mask = preprocess(pixels,self.encoder,self.image_size,augment=augment)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            tokens,structure = self.encoder_tokens(x,modalities,return_structure=return_structure)
        base = F.normalize(pool_tokens(tokens,self.encoder,mask,self.pooling),dim=1)
        optical = self.heads[0](base)
        sar = self.heads[1](base)
        embeddings = F.normalize(torch.where((modalities==0)[:,None],optical,sar),dim=1)
        return (embeddings,structure) if return_structure else embeddings

    def identity_logits(self, embeddings):
        return F.linear(embeddings,F.normalize(self.classifier.weight,dim=1))*16

def retrieval_losses(model, embeddings, labels, modalities, temperature=.07,
                     classification_weight=.5, triplet_weight=.2):
    optical = embeddings[modalities==0]
    sar = embeddings[modalities==1]
    opt_labels = labels[modalities==0]
    sar_labels = labels[modalities==1]
    if len(optical)<2 or len(optical)!=len(sar) or not torch.equal(opt_labels,sar_labels):
        raise ValueError('Balanced, ordered optical/SAR identity pairs required')
    if len(torch.unique(opt_labels)) != len(opt_labels):
        raise ValueError('Distinct identities required in each pair batch')
    logits = optical@sar.T/temperature
    targets = torch.arange(len(optical),device=embeddings.device)
    contrastive = (F.cross_entropy(logits,targets)+F.cross_entropy(logits.T,targets))/2
    classification = F.cross_entropy(model.identity_logits(embeddings),labels,label_smoothing=.1)
    distance = (1-embeddings@embeddings.T).clamp_min(0)
    same = labels[:,None]==labels[None,:]
    positive = same & ~torch.eye(len(labels),device=labels.device,dtype=torch.bool)
    hardest_positive = distance.masked_fill(~positive,-torch.inf).max(dim=1).values
    hardest_negative = distance.masked_fill(same,torch.inf).min(dim=1).values
    triplet = F.softplus(hardest_positive-hardest_negative+.2).mean()
    loss = contrastive + classification_weight*classification + triplet_weight*triplet
    return loss, {'contrastive':contrastive,'classification':classification,'triplet':triplet}

def trainable_state(model):
    names = {name for name,p in model.named_parameters() if p.requires_grad}
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items() if name in names}

def restore_trainable_state(model, state):
    expected = {name for name,p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError('Checkpoint trainable parameter coverage differs from architecture')
    full = model.state_dict()
    for name,value in state.items():
        if value.shape != full[name].shape or not torch.isfinite(value).all():
            raise ValueError('Checkpoint tensor shape or finiteness error')
        full[name] = value
    model.load_state_dict(full,strict=True)
