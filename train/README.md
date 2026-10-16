# Training

`train_ext_jtc_distill.py` trains the MorphoStyle style branch on top of a frozen ShapeMyMove FSQ-VAE backbone, with either
configuration in `configs/` (`base.yaml`, `finetune.yaml`). Run it from the repository root with
`PYTHONPATH=$PWD/train`, after `data_prep/prepare_data.py` has built `data/` (see the top-level README).

## What the configs read

| what | config key | where it comes from |
|---|---|---|
| HumanML3D `Xn`, `Xr`, `betas`, splits | `hml3d.*` | `data/HumanML3D/`, built by `data_prep/prepare_data.py` |
| 100STYLE motions, training mapping, shared beta | `data.data_dir`, `data.mapping_path`, `data.betas_shared_path` | `data/100STYLE/`, built by `data_prep/prepare_data.py` |
| HumanML3D feature mean / std | `data.motion_mean_path`, `paths.hml3d_*_path` | `assets/` |
| ShapeMyMove SA-VAE backbone | `paths.shapemove_ckpt` | `checkpoints/shapemymove_savae.pt` |
| Guo et al. text-motion evaluator and GloVe | `paths.t2m_dir`, `jtc.t2m_ckpt`, `paths.glove_dir` | `assets/t2m/finest.tar`, `assets/glove/` |
| style and content classifier for the classifier losses | `paths.classifier_path` | `assets/eval_classifiers_full.pt` |
| SMooDi style classifier and statistics | `paths.smoodi_*` | a SMooDi checkout at `../SMooDi` |

Captions: HumanML3D texts are read from `data/HumanML3D/texts` (`HUMANML3D_TEXT_DIR` overrides it); 100STYLE captions from
`data/100STYLE/texts` if present (`STYLE100_TEXT_DIR`), otherwise a template caption per content type is used.

## Code layout

| module | what it holds |
|---|---|
| `train_ext_jtc_distill.py` | the trainer: data, losses, optimisation, validation |
| `model/style_transfer.py` | the manifold-preserving style modulator (low-rank offset + temporal gate) |
| `model/joint_text_clip.py` | text-guided joint routing on top of the Guo et al. evaluator |
| `model/style_encoder.py` | style encoder and linear probe |
| `model/losses.py`, `model/losses_adv.py`, `model/shape_foot_losses.py` | reconstruction and contrastive terms; WGAN-GP and moment matching; shape disentanglement and foot physics |
| `model/critic.py`, `model/joint_aware_style.py` | the critic of the adversarial term; optional joint-aware gates |
| `model/engine/`, `model/nnutils/` | the frozen ShapeMyMove FSQ-VAE and its motion utilities |
| `auxiliary.py` | the frozen text-motion evaluator and classifiers, and the validation metrics |
| `dataset/` | `Style100Dataset`, `HumanML3DIdentityDataset`, `CrossDomainDataset`, `HumanML3DValDataset` |
| `data_prep/prepare_data.py` | links the data into `data/` and writes the leak-checked splits |

## Commands

```bash
export PYTHONPATH=$PWD/train:$PYTHONPATH
python train/train_ext_jtc_distill.py --config train/configs/base.yaml --gpu 0
python train/train_ext_jtc_distill.py --config train/configs/finetune.yaml --gpu 0 --resume outputs/train/morphostyle_base/latest.pt
```

`--resume` loads the style branch and critic of a training checkpoint and continues from its epoch; the fine-tuning run starts
from the base run's last epoch. `--max_steps N` stops after N training steps of the first epoch and saves a checkpoint.

## What the fine-tuning config adds (see the `loss:` block of `configs/finetune.yaml`)

- `traj_passthrough`: the content's root angular / XZ velocity (channels 0:3) replace the model's before every loss, so the
  content path is kept by construction (MCM-LDM conditioning).
- `lambda_content_contact` / `lambda_content_contact_h`: at frames where the content's foot-contact labels (channels 259:263)
  say a foot is planted, the output foot must be stationary in the world and within 3 cm of the floor.
- `lambda_foot_skate_geo`, `lambda_stance_duty`: geometric foot-skate penalty (contact weight from the foot height itself)
  and a stance-duty hinge that forbids floating.
- `clf_on_derived`, `lambda_native_consistency`: the style classifier sees features re-extracted from the output positions.
- Data guards: samples with |normalised value| > 30 standard deviations are dropped (`data_max_abs`), non-finite
  losses restore the last good snapshot, and validation failures never stop training.
- `teacher` is off: both configs draw every cross-domain pair at random.

`finetune.yaml` uses contact weight 1000, classifier weight 5 and lr 1e-5.

