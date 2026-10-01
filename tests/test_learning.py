"""Checks for resource provenance, foreground pooling, supervision and replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from track6.backbones import reviewed_resource

class ResourceReviewTests(unittest.TestCase):
    def test_exact_reviewed_checkpoint_and_cutoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            weight=directory/'weight'
            weight.write_bytes(b'audited checkpoint')
            resource={'public_before_cutoff_reviewed':True,'backend':'timm','cutoff':'2026-09-01',
                      'code':{'license_reviewed':True},'weights':{'license_reviewed':True,
                      'public_url_present_by':'2025-09-17','bytes':weight.stat().st_size,
                      'sha256':hashlib.sha256(weight.read_bytes()).hexdigest()}}
            manifest=directory/'resource.json'
            manifest.write_text(json.dumps(resource),encoding='utf-8')
            self.assertEqual(reviewed_resource(manifest,weight)['backend'],'timm')
            weight.write_bytes(b'tampered checkpoint')
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                reviewed_resource(manifest,weight)
            resource['weights']['public_url_present_by']='2026-09-01'
            manifest.write_text(json.dumps(resource),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'predate'):
                reviewed_resource(manifest,weight)

@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch available on compute host')
class LearningTests(unittest.TestCase):
    def test_foreground_excludes_prefix_and_background_with_empty_fallback(self):
        import torch
        from track6.backbones import pool_tokens
        patches=torch.tensor([[[1.,0.],[0.,1.],[2.,1.],[1.,2.]]]).repeat(2,1,1)
        prefix=torch.full((2,5,2),1000.)
        tokens=torch.cat([prefix,patches],dim=1)
        mask=torch.zeros((2,1,4,4)); mask[0,0,:2,:2]=1
        encoder=SimpleNamespace(num_prefix_tokens=5)
        actual=pool_tokens(tokens,encoder,mask,'foreground')
        torch.testing.assert_close(actual[0],patches[0,0])
        torch.testing.assert_close(actual[1],patches[1].mean(dim=0))
        mixed=pool_tokens(tokens,encoder,mask,'cls_foreground')
        self.assertEqual(tuple(mixed.shape),(2,4))
        self.assertTrue(torch.isfinite(mixed).all())

    def test_pair_supervision_rewards_correct_matches_and_rejects_bad_order(self):
        import torch
        from track6.learning import retrieval_losses
        class Prototype:
            def identity_logits(self,embedding): return embedding*16
        labels=torch.tensor([0,1,0,1]); modalities=torch.tensor([0,0,1,1])
        correct=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]],requires_grad=True)
        wrong=torch.tensor([[1.,0.],[0.,1.],[0.,1.],[1.,0.]])
        good,_=retrieval_losses(Prototype(),correct,labels,modalities)
        bad,_=retrieval_losses(Prototype(),wrong,labels,modalities)
        self.assertLess(float(good.detach()),float(bad))
        good.backward(); self.assertTrue(torch.isfinite(correct.grad).all())
        with self.assertRaisesRegex(ValueError,'ordered'):
            retrieval_losses(Prototype(),correct,torch.tensor([0,1,1,0]),modalities)

    def test_partial_checkpoint_restores_only_trainable_layers(self):
        import torch
        from torch import nn
        from track6.learning import SensorReID,trainable_state,restore_trainable_state
        def make_encoder():
            encoder=nn.Module()
            encoder.num_features=3
            encoder.blocks=nn.ModuleList([nn.Linear(3,3),nn.Linear(3,3)])
            encoder.norm=nn.LayerNorm(3)
            return encoder
        model=SensorReID(make_encoder(),2,256,'cls',1)
        frozen=model.encoder.blocks[0].weight.detach().clone()
        original=trainable_state(model)
        with torch.no_grad(): model.encoder.blocks[1].weight.add_(10)
        restore_trainable_state(model,original)
        torch.testing.assert_close(model.encoder.blocks[1].weight,original['encoder.blocks.1.weight'])
        torch.testing.assert_close(model.encoder.blocks[0].weight,frozen)
        self.assertNotIn('encoder.blocks.0.weight',original)
        incomplete=dict(original); incomplete.pop(next(iter(incomplete)))
        with self.assertRaisesRegex(ValueError,'coverage'):
            restore_trainable_state(model,incomplete)

if __name__=='__main__':
    unittest.main()
