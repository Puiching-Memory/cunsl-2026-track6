"""Image-level cross-modal ReID with a reviewed encoder and identity supervision."""
import torch
from torch import nn
import torch.nn.functional as F

from .backbones import pool_tokens, preprocess

class SensorReID(nn.Module):
    def __init__(self, encoder, identities, image_size, pooling, train_last_blocks):
        super().__init__()
        if not 0 <= train_last_blocks <= len(encoder.blocks) or identities < 2:
            raise ValueError('Valid block budget and at least two training identities required')
        self.encoder = encoder
        self.image_size = image_size
        self.pooling = pooling
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
        self.heads = nn.ModuleList([nn.Linear(dim,dim,bias=False) for _ in range(2)])
        with torch.no_grad():
            for head in self.heads:
                head.weight.copy_(torch.eye(dim))
        self.classifier = nn.Linear(dim,identities,bias=False)
        nn.init.normal_(self.classifier.weight, std=.01)

    def encode(self, pixels, modalities, *, augment=False):
        x,mask = preprocess(pixels,self.encoder,self.image_size,augment=augment)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            tokens = self.encoder.forward_features(x)
        base = F.normalize(pool_tokens(tokens,self.encoder,mask,self.pooling),dim=1)
        optical = self.heads[0](base)
        sar = self.heads[1](base)
        return F.normalize(torch.where((modalities==0)[:,None],optical,sar),dim=1)

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
