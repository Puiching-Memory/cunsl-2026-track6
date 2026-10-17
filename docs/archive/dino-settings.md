# DINO 历史配置与资源审查

2026-10-07项目收敛到TransOSS；以下内容仅为研究记录，原实现和可执行配置已删除。历史命令不再是当前入口。

## dinov3-reid-example-toml

```toml
# Recorded recipe; tools use the CLI documented in docs/09-modern-models.md.
[experiment]
track = "track6"
code_commit = "4c2394bdc35b340ab6f53de25acbb8a5a2e82990"
seed = 20261001
confirmation_seed = 20261002
status = "two_local_holdouts_measured"

[data]
archive_sha256 = "68aaa274e11e81ab65a3e199449595ff9efa7d73f7a5927cfab469e7513e1513"
split_sha256 = "b33cbea7e9d9771f12ee4f507ef13a868fe0f3be0d1a4c77bbbc1290402e5bd5"
confirmation_split_sha256 = "aa7b5835cbbf9699b4b0d1ece79717d36cd9a169ea5d273f6e7c3bb0a39304e0"
validation_protocol = "identity-holdout-v1"

[model]
backend = "timm"
timm_version = "1.0.20"
variant = "vit_base_patch16_dinov3.lvd1689m"
weights_sha256 = "1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332"
image_size = 256
pooling = "cls_foreground"
foreground_threshold_uint8 = 8
train_last_blocks = 4
precision = "BF16 encoder; FP32 heads, embeddings and losses"

[training]
epochs = 12
early_stopping_patience = 0 # Historical fixed-final-epoch reproduction.
identities_per_batch = 32
inference_batch_size = 32
backbone_lr = 0.00002
head_lr = 0.001
weight_decay = 0.01
temperature = 0.07
classification_weight = 0.5
triplet_weight = 0.2
augment = false
selection = "fixed final epoch"

[results]
first_holdout_weighted_score = 0.3365939470485164
confirmation_holdout_weighted_score = 0.3114983463797728
rankings_independently_checked = true
checkpoint_replays_byte_identical = true
official_scorer_confirmed = false
leaderboard_submitted = false
```

## domain-ablation-example-toml

```toml
# Controlled ablations motivated by primary ship optical/SAR ReID sources.
# These are our DINOv3 adaptations, not reproductions of TransOSS or SDF-Net.
[protocol]
splits = ["local/splits/identity-v1.json", "local/splits/identity-v2.json"]
seeds = [20261001, 20261002]
selection = "fixed final epoch; all variants reported"
test_used_for_training = false

[model]
resource = "resources/dinov3-vitb16-timm.json"
weights = "data/weights/dinov3-vitb16-timm.safetensors"
image_size = 256
pooling = "cls_foreground"
train_last_blocks = 4

[training]
epochs = 12
early_stopping_patience = 0 # Historical fixed-final-epoch reproduction.
identities_per_batch = 32
backbone_lr = 0.00002
head_lr = 0.001
weight_decay = 0.01
temperature = 0.07
classification_weight = 0.5
triplet_weight = 0.2
augmentation = "none"

[[variants]]
name = "frozen-stem-control"
stem_mode = "frozen"
structure_weight = 0.0

[[variants]]
name = "learned-shared-stem"
stem_mode = "shared"
structure_weight = 0.0

[[variants]]
name = "learned-sensor-stems"
stem_mode = "separate"
structure_weight = 0.0

[[variants]]
name = "sensor-stems-struct6"
stem_mode = "separate"
structure_layer = 6
structure_weight = 1.0
```

## early-stopping-example-toml

```toml
# Proposed settings only; no GPU training result yet. This is documentation,
# not a config-file CLI input. See tools/README.md for the explicit command.
[protocol]
split = "local/submission-v1/recovered/splits/identity-v1.json"
selection = "best local weighted R@1/mAP@10, including epoch zero"
ap_denominator = "min_relevant_k"
official_scorer_confirmed = false
validation_used_for_checkpoint_selection = true
validation_used_for_gradient_updates = false
test_used_for_training = false

[model]
resource = "resources/dinov3-vitb16-timm.json"
weights = "data/weights/dinov3-vitb16-timm.safetensors"
image_size = 256
pooling = "cls_foreground"
stem_mode = "separate"
train_last_blocks = 4
structure_weight = 0.0

[training]
seed = 20261001
epochs = 60
early_stopping_patience = 8
early_stopping_min_delta = 0.001
early_stopping_min_epochs = 10
identities_per_batch = 32
backbone_lr = 0.00002
head_lr = 0.001
temperature = 0.07
classification_weight = 0.5
triplet_weight = 0.2
augmentation = "none; keep fixed for first controlled comparison"

[comparison]
# Extending the cosine horizon from 12 to 60 also changes learning rates.
# First use maximum 12 epochs with the same optimizer/seed and no augmentation;
# compare the best score with the final score along the same training trajectory.
diagnostic_maximum_epochs = 12
confirm = "Independently retrain on the second identity split; not a replay of split one"
```

## experiment-example-toml

```toml
# Example settings for the audited fixed-feature baseline; CLI commands are in docs/08-baselines.md.
[experiment]
track = "track6"
seed = 20261001
status = "local_baselines_completed"

[data]
root = "/root/autodl-tmp/cunsl-2026/track6/data"
archive_sha256 = "68aaa274e11e81ab65a3e199449595ff9efa7d73f7a5927cfab469e7513e1513"
split_sha256 = "b33cbea7e9d9771f12ee4f507ef13a868fe0f3be0d1a4c77bbbc1290402e5bd5"
validation_protocol = "identity-holdout-v1"

[model]
framework = "pytorch_existing_autodl_environment"
initialization = "reviewed_original_dinov2_vits14_frozen"
pretrained_url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
pretrained_sha256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
resource_rules_reviewed = true

[adapter]
epochs = 20
identities_per_batch = 128
lr = 0.001
temperature = 0.07
identity_regularization = 0.01

[results]
local_metrics_independently_checked = true
official_scorer_confirmed = false
leaderboard_submitted = false
```

## dinov2-vits14-json

```json
{
  "name": "DINOv2 original ViT-S/14 LVD-142M backbone without registers",
  "variant": "dinov2_vits14",
  "use": "frozen image embedding baseline; optional supervised adaptation using official training identities",
  "cutoff": "2026-09-01",
  "reviewed_at": "2026-10-01",
  "public_before_cutoff_reviewed": true,
  "code": {
    "repository": "https://github.com/facebookresearch/dinov2",
    "commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
    "commit_date": "2026-06-03T17:28:49+02:00",
    "license": "Apache-2.0",
    "license_url": "https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE",
    "license_reviewed": true
  },
  "weights": {
    "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "public_url_present_by": "2023-04-17",
    "date_evidence": "https://github.com/facebookresearch/dinov2/blob/fc49f49d734c767272a4ea0e18ff2ab8e60fc92d/README.md",
    "license": "Apache-2.0",
    "license_evidence": "https://github.com/facebookresearch/dinov2/blob/81b2b6419385a321287de91e00282ef7cbd26f94/MODEL_CARD.md",
    "license_evidence_date": "2023-08-31",
    "license_reviewed": true,
    "sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    "bytes": 88283115
  },
  "scope": "Review applies only to the exact original ViT-S/14 checkpoint. XRay, Cell and other variants are not covered. Download fingerprint verified on 2026-10-01."
}
```

## dinov3-vitb16-timm-json

```json
{
  "name": "DINOv3 ViT-B/16 LVD-1689M, timm maintainer conversion",
  "variant": "vit_base_patch16_dinov3.lvd1689m",
  "backend": "timm",
  "cutoff": "2026-09-01",
  "reviewed_at": "2026-10-01",
  "use": "Modern generic image backbone comparison; frozen features and supervised cross-modal fine-tuning on official training identities",
  "public_before_cutoff_reviewed": true,
  "code": {
    "repository": "https://github.com/huggingface/pytorch-image-models",
    "version": "1.0.20",
    "commit": "234907eca5d801a369b536e001ec0b45d8ec9d60",
    "public_by": "2025-09-21",
    "license": "Apache-2.0",
    "license_url": "https://github.com/huggingface/pytorch-image-models/blob/234907eca5d801a369b536e001ec0b45d8ec9d60/LICENSE",
    "license_reviewed": true,
    "wheel_sha256": "f6e62f780358476691996c47aa49de87b95cc507edf923c3042f74a07e45b7fe"
  },
  "weights": {
    "url": "https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m/resolve/e2d80655c8189be2e61f55ed9cc4d4d8e7e31b88/model.safetensors",
    "repository": "timm/vit_base_patch16_dinov3.lvd1689m",
    "revision": "e2d80655c8189be2e61f55ed9cc4d4d8e7e31b88",
    "public_url_present_by": "2025-09-17",
    "date_evidence": "https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m/commit/e2d80655c8189be2e61f55ed9cc4d4d8e7e31b88",
    "license": "DINOv3 License (2025-08-19)",
    "license_reviewed": true,
    "license_evidence": "https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m/blob/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da/LICENSE.md",
    "license_sha256": "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e",
    "sha256": "1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332",
    "bytes": 342579728,
    "format": "safetensors",
    "public_ungated": true
  },
  "original_source": "https://github.com/facebookresearch/dinov3",
  "model_card": "https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m/blob/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da/README.md",
  "license_obligations": [
    "Retain DINOv3 License when redistributing weights or derivatives.",
    "Acknowledge DINOv3 in published research.",
    "Comply with the stated permitted use restrictions, including section 1.b.v on ITAR and end uses prohibited by Trade Controls.",
    "Recheck applicable terms before redistribution; the Meta website license also requires a prominent Built with DINOv3 attribution."
  ],
  "conversion_notes": "Maintainer removes zero QKV bias and computes RoPE periods in float32; use default timm numerical behavior and record it. Not byte-identical to Meta's native checkpoint.",
  "verification_evidence": "../docs/evidence/2026-10-01-dinov3-resources.json",
  "eligibility_scope": "Public-date and source review for public algorithm competition research; not a specific organizer approval or authorization for controlled military or espionage end uses. See docs/09-modern-models.md for the archived and website license differences.",
  "scope": "Exact safetensors SHA matches initial 2025-09-17 release; Windows download and compute-host transfer both verified on 2026-10-01. Loader continues to verify each local file. Generic web pretraining does not establish SAR applicability."
}
```

## dinov3-vitl16-sat-timm-json

```json
{
  "name": "DINOv3 ViT-L/16 SAT-493M, timm maintainer conversion",
  "variant": "vit_large_patch16_dinov3.sat493m",
  "backend": "timm",
  "cutoff": "2026-09-01",
  "reviewed_at": "2026-10-01",
  "use": "Satellite optical RGB pretrained backbone comparison; frozen features and supervised cross-modal fine-tuning on official training identities",
  "public_before_cutoff_reviewed": true,
  "code": {
    "repository": "https://github.com/huggingface/pytorch-image-models",
    "version": "1.0.20",
    "commit": "234907eca5d801a369b536e001ec0b45d8ec9d60",
    "public_by": "2025-09-21",
    "license": "Apache-2.0",
    "license_url": "https://github.com/huggingface/pytorch-image-models/blob/234907eca5d801a369b536e001ec0b45d8ec9d60/LICENSE",
    "license_reviewed": true,
    "wheel_sha256": "f6e62f780358476691996c47aa49de87b95cc507edf923c3042f74a07e45b7fe"
  },
  "weights": {
    "url": "https://huggingface.co/timm/vit_large_patch16_dinov3.sat493m/resolve/edb5320108653705f6a402434a1ce00e2cc214f3/model.safetensors",
    "repository": "timm/vit_large_patch16_dinov3.sat493m",
    "revision": "edb5320108653705f6a402434a1ce00e2cc214f3",
    "public_url_present_by": "2025-09-17",
    "date_evidence": "https://huggingface.co/timm/vit_large_patch16_dinov3.sat493m/commit/edb5320108653705f6a402434a1ce00e2cc214f3",
    "license": "DINOv3 License (2025-08-19)",
    "license_reviewed": true,
    "license_evidence": "https://huggingface.co/timm/vit_large_patch16_dinov3.sat493m/blob/ce148bd5ed3de1eb75ebdccafcb2c280bb3e8f7c/LICENSE.md",
    "license_sha256": "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e",
    "sha256": "8d18337f766102f951ac0ca543a22536f58caeb6b6c33cfef73326f866791286",
    "bytes": 1212347640,
    "format": "safetensors",
    "public_ungated": true
  },
  "original_source": "https://github.com/facebookresearch/dinov3",
  "model_card": "https://huggingface.co/timm/vit_large_patch16_dinov3.sat493m/blob/ce148bd5ed3de1eb75ebdccafcb2c280bb3e8f7c/README.md",
  "license_obligations": [
    "Retain DINOv3 License when redistributing weights or derivatives.",
    "Acknowledge DINOv3 in published research.",
    "Comply with the stated permitted use restrictions, including section 1.b.v on ITAR and end uses prohibited by Trade Controls.",
    "Recheck applicable terms before redistribution; the Meta website license also requires a prominent Built with DINOv3 attribution."
  ],
  "conversion_notes": "Maintainer removes zero QKV bias and computes RoPE periods in float32; use default timm numerical behavior and record it. Not byte-identical to Meta's native checkpoint.",
  "verification_evidence": "../docs/evidence/2026-10-01-dinov3-resources.json",
  "eligibility_scope": "Public-date and source review for public algorithm competition research; not a specific organizer approval or authorization for controlled military or espionage end uses. See docs/09-modern-models.md for the archived and website license differences.",
  "scope": "Exact safetensors SHA matches initial 2025-09-17 release; Windows download and compute-host transfer both verified on 2026-10-01. Loader continues to verify each local file. SAT-493M is optical RGB imagery, not optical/SAR paired pretraining."
}
```
