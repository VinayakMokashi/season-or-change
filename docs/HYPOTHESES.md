# Hypotheses

Written on 2026-10-23, before running the linear-probe sweep in `src/probe/`
(`results/probes/`). Every hypothesis below is decided by the decision rule
stated with it.

## Protocol fixed in advance

- **Backbones.** ResNet-18 encoders, frozen and in `eval()` mode. Features are
  the 512-d global-average-pooled output.
- **Inputs.** `ToTensor()` only, which is the authors' EuroSAT evaluation
  transform. UC Merced images are first resized to 256×256.
- **Head.** A single linear layer trained with Adam (lr 1e-3) for 100 epochs.
  The learning rate drops ×0.1 at epochs 60 and 80, and the batch size is 256.
- **Primary metric.** Top-1 accuracy at the final epoch, averaged over five
  probe seeds (0–4). We never select an epoch on the test split.
- **Pairwise comparisons.** Paired bootstrap over test images (10,000
  resamples) of the per-image correctness averaged over seeds. A difference is
  "resolved" when its 95% interval excludes 0.
- **Splits.** EuroSAT uses the SeCo/CACo train/val split (16,200 / 5,400).
  UC Merced uses a stratified 80/20 split per class with seed 0
  (1,680 / 420), committed in `splits/`.

## H1 — The paper's component chain at 100k (claim verification)

The CACo paper (Table 6) reports that, on EuroSAT, adding long-term temporal
contrast and then change-awareness each improve accuracy. The prediction is
that the authors' released 100k checkpoints order
**SeCo < TeCo < CACo** on EuroSAT.

- **Supported** if both TeCo − SeCo and CACo − TeCo are resolved and positive.
- **Refuted** if either is resolved and negative.
- **Otherwise unresolved.**

## H2 — Satellite-specific objectives beat MoCo v2 (claim verification)

The paper (Table 1) reports that MoCo v2 trained on satellite imagery
underperforms SeCo and CACo. The prediction is that the authors' MoCo-100k
checkpoint scores below both their SeCo-100k and their CACo-100k on EuroSAT.

- **Decision rule:** the same as H1.

## H3 — Change-awareness at small scale (ablation)

The prediction is that, for our own 10k-pretrained encoders, change-awareness
adds no meaningful accuracy over long-term contrast alone.

- **Supported** if CACo − TeCo on EuroSAT is below +1.0 point and its 95%
  interval includes 0.
- **Refuted** if the difference is resolved and at least +1.0.

## H4 — Distribution shift to 1-ft aerial imagery (stress test)

UC Merced is aerial imagery at about 0.3 m/pixel. EuroSAT and the pretraining
data are Sentinel-2 at 10 m/pixel. The prediction is that ImageNet's advantage
over the best satellite-pretrained encoder of the same scale is larger on
UC Merced than on EuroSAT, at both 10k and 100k.

- **Supported** if ImageNet − (best satellite encoder) is larger on UC Merced
  for both scales.

## Notes added after the sweep (2026-10-23)

- **Outcomes** (`results/probes/hypotheses.json`): H1 supported, H2 unresolved
  (CACo beats MoCo-100k, while SeCo's lead over it is not resolved), H3
  supported, H4 supported.
- **H3** was motivated by our earlier single-seed evaluation of the same encoders
  (`results/original_eval/`), where CACo and TeCo were within half a point on
  EuroSAT.
- **H4.** "Best satellite-pretrained encoder" includes the authors' MoCo v2
  checkpoint at 100k. Our own MoCo v2 encoder was not part of the sweep, so the
  10k comparison covers SeCo, TeCo and CACo. In the original single-seed
  evaluation, ImageNet's lead over our MoCo v2 fell from 8.0 points (EuroSAT) to
  5.7 points (UC Merced, on a different split).
