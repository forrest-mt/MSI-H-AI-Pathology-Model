# Discerning MSI status from H&E slides in colorectal cancer

A reproduction of [Kather et al. 2019](https://doi.org/10.1038/s41591-019-0462-y) using a
frozen pathology foundation model (UNI2-h) and a small attention-MIL head (CLAM), evaluated
with 5-fold patient-level cross-validation on TCGA-COAD.

## Method

**Encoder — frozen.** UNI2-h ([Chen et al. 2024](https://doi.org/10.1038/s41591-024-02857-3)),
  a pathology foundation model pretrained on 100k+ slides, converts each tile into a 1536-dim
  embedding. Embeddings were **pre-extracted and released** by Mahmood Lab
  ([`MahmoodLab/UNI2-h-features`](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features)),
  so the encoder is never run here. This is what makes the project feasible on a free Colab T4:
  the expensive one-time upstream work was already done by someone else.

**Aggregator — trained.** CLAM_SB ([Lu et al. 2021](https://doi.org/10.1038/s41551-020-00682-w)),
  gated attention-MIL with an instance-level clustering loss. A `Linear(1536→1024)` layer bridges
  the dimension mismatch. This is the only part that learns.

**Labels.** MSIsensor scores were taken from the cBioPortal COADREAD PanCancer Atlas clinical file and
thresholded at ≥10.

**Splits.** `StratifiedGroupKFold(n_splits=5, groups=patient_id)` — grouping keeps all of a
patient's slides on one side (asserted every fold), stratification keeps prevalence comparable
across folds.

**Training.** 3 epochs, `Adam(lr=1e-4)`, `batch_size=1` (bags have variable tile counts and
can't be stacked), class-weighted `CrossEntropyLoss` with `reduction='sum'`.

## Reproducing

```bash
# Runtime: Colab T4. CPU works but takes hours.
```

1. Open `Phase2_MSI_CLAM_CV.ipynb` in Colab, set runtime to **T4 GPU**.
2. Sections 0–1 download the pre-extracted UNI2-h features. Requires a Hugging Face account with
   access granted to the gated dataset. The tar is cached to Drive (persists) and extracted to
   local disk (fast, rebuilt each session).
3. Section 2 needs `coadread_tcga_pan_can_atlas_2018_clinical_data.tsv`, downloaded from
   [cBioPortal](https://www.cbioportal.org/study/summary?id=coadread_tcga_pan_can_atlas_2018).
4. Run top to bottom. Five folds × 3 epochs is roughly 20 minutes on a T4.

To rerun the permuted-label control, set `PERMUTE_LABELS = True` in the mode cell, **restart the
kernel**, and run again. The guard will confirm the permutation reached the training loop.


## Repository contents

```
Phase2_MSI_CLAM_CV.ipynb      end-to-end notebook, committed with outputs
figures/                      all figures (png + editable svg)
oof_predictions.csv           out-of-fold predictions, slide level
oof_patient_level.csv         aggregated to patient level
```

Every metric in this README is recomputable from `oof_predictions.csv` in seconds, without
retraining anything.

## References

- Kather JN, Pearson AT, Halama N, et al. Deep learning can predict microsatellite instability
  directly from histology in gastrointestinal cancer. *Nat Med* 25:1054–1056 (2019).
  [doi:10.1038/s41591-019-0462-y](https://doi.org/10.1038/s41591-019-0462-y)
- Chen RJ, et al. Towards a general-purpose foundation model for computational pathology.
  *Nat Med* 30:850–862 (2024).
- Lu MY, et al. Data-efficient and weakly supervised computational pathology on whole-slide
  images. *Nat Biomed Eng* 5:555–570 (2021).
- Schoenpflug LA, et al. A protocol for evaluating robustness to H&E staining variation in
  computational pathology models. *J Pathol Inform* (2026).
- Bass C, et al. H&E-based MSI/MMR testing with AI in colorectal cancer: a multi-centred blinded
  evaluation. *npj Digit Med* 9:44 (2026).
- Hanley JA, McNeil BJ. The meaning and use of the area under a ROC curve. *Radiology*
  143:29–36 (1982).
- Bengio Y, Grandvalet Y. No unbiased estimator of the variance of k-fold cross-validation.
  *JMLR* 5:1089–1105 (2004).

## License and attribution

Code in this repository: MIT.

UNI2-h features are **CC-BY-NC-ND** (non-commercial, no derivatives) — see the
[dataset card](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features). TCGA data are
available from the [GDC](https://portal.gdc.cancer.gov/).
