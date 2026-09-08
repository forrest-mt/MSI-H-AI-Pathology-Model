# Discerning MSI status from H&E slides in colorectal cancer

# MSI-H vs MSS from H&E

**Frozen UNI2-h tile embeddings + CLAM attention-MIL for MSI-H vs MSS classification in TCGA-COAD**

This repository contains the code and analysis for a learning/reproduction project asking:

> **Can a pathology foundation model extract enough information from a routine H&E slide to distinguish MSI-H from MSS colon cancer?**

This is a training project, not a novel clinical or methodological contribution. The task has already been demonstrated in published research and commercialized. The purpose here is to reproduce the basic result with modern pathology foundation-model embeddings, then examine where the result holds up and where it breaks.

---

## 1. Project scope

The model predicts **MSI-H vs MSS** from H&E pathology slides using:

1. **UNI2-h** — a pretrained pathology foundation model whose slide-level tile embeddings are used here rather than running the encoder ourselves.
2. **CLAM** — an attention-based multiple-instance learning (MIL) model that aggregates thousands of tile embeddings into a slide-level prediction.

The current experiment uses **TCGA-COAD**, so the cohort is colon adenocarcinoma rather than the full colorectal cancer (CRC) population.

The primary endpoint is **patient-level discrimination**, measured by AUROC.

### What this project does establish

The current experiment shows that:

> Frozen UNI2-h representations contain substantial information associated with MSIsensor-defined MSI-H status in the TCGA-COAD cohort.

### What it does not establish

It does **not** establish:

* clinical-grade diagnostic performance;
* performance on new hospitals or scanners;
* generalization to independent cohorts;
* equivalence to MMR-IHC, MSI-PCR, or clinical NGS testing;
* that CLAM attention is necessary to obtain the observed performance;
* that the model predicts treatment response directly.

---

# 2. Data

## 2.1 Pathology features

The pathology input is the precomputed **UNI2-h feature dataset** released by Mahmood Lab.

The notebook downloads the TCGA-COAD archive and extracts one `.h5` file per slide.

Each slide contains a variable number of tile embeddings with dimension:

```text
[N_tiles, 1536]
```

The UNI2-h encoder itself is **not run in this project**. The encoder computation has already been performed by the feature provider.

The current archive contains:

* **442 available COAD slide feature files**
* **419 slides** after the clinical-label join and manifest filtering
* **411 unique patients** represented in the final analysis cohort.

---

## 2.2 MSI labels

MSI status is derived from the continuous **MSIsensor Score** supplied in the TCGA clinical data.

The binary label is:

```text
MSI-H = MSIsensor >= 10
MSS   = MSIsensor < 10
```

This threshold is treated as a modeling choice rather than an immutable clinical truth.

The full clinical table contains:

* 584 patients with non-missing MSIsensor scores
* 78 patients labeled MSI-H
* 506 patients labeled MSS.

The final image cohort is smaller because only patients with matching COAD pathology features are retained.

### Important label limitation

The reference label is **MSIsensor-derived**, not MMR-IHC or MSI-PCR.

Therefore performance numbers in this project should not be treated as directly comparable with studies evaluated against clinical reference-standard labels. The notebook explicitly treats this as a limitation.

---

# 3. Slide selection and manifest

Each slide is represented by:

* `patient_id`
* `slide_id`
* `label`
* `site_id`
* `sample_type`
* local path to its `.h5` feature file

The notebook explicitly checks that the clinical table contains one unique patient per row before joining pathology data.

Site ID is derived from the TCGA slide identifier and is retained for later site-grouped experiments.

One metastatic slide is currently retained in the 419-slide cohort. This is an explicit choice rather than an accidental inclusion.

---

# 4. Model architecture

## 4.1 Input

UNI2-h produces **1536-dimensional tile embeddings**.

CLAM's selected configuration expects **1024-dimensional input**, so the notebook inserts a learned linear projection:

```text
1536 → 1024
```

The projection contains approximately **1.57 million trainable parameters** and represents roughly two-thirds of the total trainable parameter count in the main configuration.

---

## 4.2 CLAM configuration

The model uses the single-branch CLAM architecture (`CLAM_SB`) with:

```text
model_size = "small"
gate       = True
dropout    = True
k_sample   = 8
n_classes  = 2
bag_loss   = "ce"
inst_loss  = "svm"
feat_dropout = False
```

The implementation is loaded from the Mahmood Lab CLAM repository.

The notebook records the CLAM commit used:

```text
53e2409d4a8189c682c173382964a85f114f923c
```

The code is intended to pin this version for reproducibility.

---

# 5. Multiple-instance learning formulation

A whole-slide image contains thousands of tiles.

Rather than treating each tile as an independent labeled example, the slide is treated as a **bag of tile embeddings**:

```text
Slide
 ├── tile 1 → 1536-d embedding
 ├── tile 2 → 1536-d embedding
 ├── ...
 └── tile N → 1536-d embedding
```

CLAM learns an attention mechanism over the tiles and produces a **single slide-level prediction**.

The project ultimately evaluates predictions at the **patient level**, not the tile level.

---

# 6. Training

## 6.1 Loss

Training uses a weighted cross-entropy bag loss plus the CLAM instance-level loss:

```text
total loss =
    0.7 × bag classification loss
  + 0.3 × instance loss
```

Class weights are calculated from the training subset of each fold.

The cross-entropy uses:

```python
reduction="sum"
```

because the DataLoader uses `batch_size=1`. With the default `reduction="mean"`, the per-example class weight would effectively cancel in this setup.

---

## 6.2 Optimization

The current training configuration is:

```text
optimizer : Adam
learning rate : 1e-4
maximum epochs : 8
seed : 42
```

Epoch count is **not fixed in advance**.

Instead, each outer fold selects its training epoch using an inner validation split.

---

# 7. Cross-validation design

The primary evaluation uses **5-fold patient-level cross-validation**.

The key rule is:

> **No patient may appear in both the training and test portion of a fold.**

This is enforced with `StratifiedGroupKFold` using `patient_id` as the grouping variable. The notebook additionally asserts that no patient appears on both sides of a fold.

This matters because some patients have multiple slides. Splitting by slide instead of patient could allow the model to see one slide from a patient during training and another during testing.

---

## 7.1 Nested inner split

Each outer training fold is divided again into:

```text
Outer training fold
├── Inner-fit set
└── Inner-validation set
```

The inner split uses approximately:

```text
75% inner-fit
25% inner-validation
```

at the patient level.

The inner-validation set is used for two purposes:

1. **Selecting the training epoch**
2. **Fitting the Platt calibration model**

The untouched outer test fold is only used after those decisions have been made.

This was introduced to remove test-set tuning from the earlier version of the experiment.

---

# 8. Calibration

Calibration is a secondary part of the analysis.

The CLAM model produces probabilities, but probabilities from separately trained folds do not necessarily use exactly the same numerical scale.

A **Platt scaling** model is therefore fitted within each outer training fold using the inner-validation predictions.

The calibrator is then applied to the untouched outer test fold.

Conceptually:

```text
inner-fit patients
       ↓
   train CLAM
       ↓
inner-val predictions
       ↓
  fit Platt model
       ↓
outer-test predictions
       ↓
 calibrated probabilities
```

No outer-test labels are used to fit the calibrator.

Calibration has little effect on discrimination in the current run:

```text
raw pooled patient AUROC       0.876
calibrated pooled AUROC        0.875
```

This is expected: calibration changes the score scale, whereas AUROC primarily measures ranking.

---

# 9. Slide-level vs patient-level evaluation

The model generates one prediction per slide.

However, the clinical unit of interest is the **patient**.

A patient with multiple slides therefore has its slide probabilities aggregated:

```text
patient probability = mean(slide probabilities)
```

The analysis then calculates AUROC at the patient level.

A maximum-slide probability is also calculated as a sensitivity analysis:

```text
patient probability = max(slide probabilities)
```

In the current cohort only **7 patients have more than one slide**, and mean and maximum aggregation produce nearly identical AUROC values.

---

# 10. What does "macro AUROC" mean?

The headline metric is the **mean patient-level AUROC across the five outer folds**.

For each fold:

```text
AUROC_fold = AUROC on that fold's held-out patients
```

Then:

```text
macro AUROC = mean(AUROC_fold_1 ... AUROC_fold_5)
```

This is different from pooling all five sets of predictions together and calculating one AUROC.

The macro estimate is preferred here because each fold is produced by a different trained model. AUROC within a fold depends only on the ordering produced by that model and is therefore unaffected by cross-fold score-scale differences.

---

# 11. Confidence intervals

Two different estimands are reported separately.

### Primary

**Macro AUROC**

A cluster bootstrap resamples patients **within each fold**, recomputes the five fold AUROCs, and averages them.

Current result:

```text
Macro patient-level AUROC
0.895
95% CI [0.850, 0.934]

Per-fold:
[0.932, 0.943, 0.865, 0.957, 0.776]

Range:
0.776–0.957

SD:
0.075
```

### Secondary

**Pooled AUROC**

A separate patient-level bootstrap is performed after pooling all OOF predictions.

Current result:

```text
Pooled patient-level AUROC
0.876
95% CI [0.824, 0.923]
```

The pooled confidence interval should **not** be attached to the macro AUROC.

Both intervals should still be interpreted cautiously because the five CV folds share training data and therefore are not independent replicates.

---

# 12. Main result

The current main run uses:

```text
411 patients
419 slides
66 MSI-H patients
345 MSS patients
16.1% patient-level MSI-H prevalence
```

The headline result is:

```text
Mean patient-level AUROC: 0.895

95% CI: 0.850–0.934
Fold range: 0.776–0.957
Fold SD: 0.075
```

The result indicates substantial MSI-H/MSS signal within this TCGA-COAD cohort, but the relatively wide fold range indicates meaningful sampling variability.

---

# 13. Why the pooled AUROC is lower

The current results are:

```text
Patient-level macro AUROC      0.895
Patient-level pooled AUROC     0.876
```

The difference is modest compared with earlier versions of the experiment.

A rank-normalisation diagnostic produces:

```text
Raw pooled AUROC               0.876
Rank-normalised pooled AUROC   0.890
Macro AUROC                    0.895
```

This supports the idea that score-scale differences between independently trained folds explain part of the pooling gap.

However, the notebook intentionally avoids claiming that the entire gap is caused by calibration or score scale. Fold prevalence and other distributional differences can also contribute.

# 14. Mean-pooling baseline

A deliberately simple baseline tests whether CLAM attention is actually necessary.

For each slide:

1. Read all UNI2-h tile embeddings.
2. Average them into one 1536-dimensional slide vector.
3. Fit a standardized logistic regression classifier.

The same cross-validation folds are reused.

Results:

```text
                              Macro AUROC
Mean-pool + logistic regression   0.879
CLAM attention-MIL                0.895

Difference                        +0.015
```

The fold-to-fold standard deviation is approximately 0.075.

Therefore:

This experiment does not establish that CLAM attention provides meaningful improvement over simple mean pooling at this sample size.

The more conservative interpretation is that **the pretrained UNI2-h representation already contains substantial predictive information**, while the additional 2.36M-parameter aggregation head has not been shown to earn its complexity.

### Important precision

This experiment does **not** show that individual tile embeddings are linearly separable.

The linear classifier is trained after averaging the tiles into a single slide representation.

---

# 15. Continuous MSIsensor analysis

The binary label is created by thresholding a continuous MSIsensor score at 10.

The notebook therefore asks a stronger question:

> Does the model's score track the underlying continuous MSI phenotype, rather than simply learning the binary boundary?

Current result:

```text
Spearman correlation, all patients
rho = +0.279
p = 8.5e-09

Spearman correlation, MSS patients only
rho = -0.042
p = 0.43
```

The MSS-only result is particularly informative because all patients in that analysis share the same binary label.

The current evidence therefore supports:

> **The model captures the broad MSI-H/MSS distinction, but does not demonstrate a continuous relationship with MSI burden within MSS tumors.**

The grey-zone sensitivity analysis also shows only a small change:

```text
All patients       AUROC 0.895
Grey zone excluded AUROC 0.899
```

with six patients in the chosen 3.5–10 MSIsensor grey zone.

---

# 16. Operating points

The notebook reports sensitivity/specificity operating points primarily as an illustration of the clinical tradeoff.

At a post-hoc threshold corresponding to approximately 95% sensitivity:

```text
Sensitivity    95.5%
Specificity    22.6%
```

This should **not** be interpreted as a prospectively validated clinical operating point.

The sensitivity target is chosen after inspecting the ROC results, so the specificity at that target is an exploratory estimate.

The calibrated probabilities themselves are out-of-fold; the operating-point selection is still post hoc.

---

# 17. Site effects

TCGA slides contain information about their tissue source institutions, including potential differences in:

* staining;
* fixation;
* scanners;
* section preparation;
* other site-specific characteristics.

A site-prevalence analysis checks whether MSI-H status is strongly associated with a particular site.

This is useful but insufficient to establish that the model is free of site-related confounding.

The stronger test is **site-grouped cross-validation**, in which patients from a site are withheld together so the model cannot rely on having previously seen that institution.

The notebook supports this experiment through:

```python
GROUP_COL = "site_id"
```

but the current saved results include only the main patient-grouped run.

# 18. Negative control

The notebook also supports a patient-level label permutation control:

```python
PERMUTE_LABELS = True
```

Labels are shuffled at the patient level so patients with multiple slides retain internally consistent labels.

Expected behavior:

```text
AUROC ≈ 0.5
```

A substantially higher result would indicate either a wiring problem or a potential source of leakage.

The current v3 notebook contains this control, but the saved `permuted` output is not yet present.

# 19. Reproducibility and run configuration

The notebook uses explicit run tags:

| Run           | Purpose                                           |
| ------------- | ------------------------------------------------- |
| `main`        | Primary patient-grouped experiment                |
| `sitegrouped` | Test sensitivity to site-level distribution shift |
| `permuted`    | Negative control                                  |
| `noproj`      | Freeze the 1536→1024 projection                   |

Each run saves:

```text
oof_<RUN_TAG>.csv
epochs_<RUN_TAG>.csv
```

to the configured Drive directory.

The notebook is designed to be run from a fresh Colab kernel, using a T4 GPU.

---

# 20. Current run status

At the time of this README:

```text
Completed:
✓ main patient-level CV
✓ nested epoch selection
✓ nested calibration
✓ patient-level aggregation
✓ macro confidence interval
✓ pooled confidence interval
✓ rank-normalisation diagnostic
✓ mean-pooling baseline
✓ continuous MSIsensor analysis

Implemented but not yet run:
□ site-grouped CV
□ permuted-label control
□ frozen-projection ablation
```

The notebook currently reports missing saved outputs for the final three runs.

---

# 21. Known limitations

### Single cohort

All current performance estimates come from TCGA-COAD.

There is no external validation cohort, so no genuine generalization claim can be made.

### Derived labels

Labels are based on `MSIsensor >= 10`, rather than IHC/PCR clinical reference standards.

### Colon only

This is a COAD experiment. It should not be described as a full colorectal cancer validation.

### Small MSI-H sample

There are only 66 MSI-H patients in the final cohort.

This contributes to the variability between folds and makes high-sensitivity operating points especially unstable.

### Small calibration set

Each outer fold uses approximately 82 patients for inner validation/calibration, including roughly 13 MSI-H patients.

The calibrator is therefore honest with respect to the test set but still statistically noisy.

### Inner validation has two roles

The same inner-validation patients are used for both:

* epoch selection;
* calibration.

This avoids throwing away another large fraction of the data, but means the calibration estimates may remain somewhat optimistic.

### Operating points are post hoc

Sensitivity targets are chosen after inspecting model performance.

Therefore reported specificity values at those targets should be viewed as exploratory rather than prospectively validated thresholds.

### Attention was not shown to outperform averaging

The CLAM model achieves 0.895 macro AUROC versus 0.879 for mean pooling.

The difference is small relative to fold-to-fold variability, so the experiment does not demonstrate that attention meaningfully improves performance.

### No external site validation

Site-grouped CV has not yet been run.

### No direct treatment-response prediction

MSI status is a biomarker associated with treatment response; this project predicts the biomarker itself.

It does not predict whether an individual patient will respond to immunotherapy.

---

# 22. Interpretation

The most defensible conclusion from this experiment is:

> **Frozen UNI2-h representations contain substantial information associated with MSI-H status in TCGA-COAD. A CLAM attention-MIL head achieves a mean patient-level AUROC of 0.895 in five-fold patient-level cross-validation, but performance varies across folds and is not demonstrably better than a simple mean-pooling baseline. The result is therefore evidence of within-cohort predictive signal, not evidence of clinical readiness or external generalization.**

The most interesting technical takeaway is arguably not the 0.895 itself.

It is that:

> **Most of the predictive signal may already be present in the pretrained pathology representation.**

This changes the next question from:

> “Can I build a better classifier?”

to:

> “What survives when the cohort, hospital, scanner, and population change?”

---

# 23. Planned next experiments

Highest priority:

### 1. Site-grouped cross-validation

Run:

```python
RUN_TAG = "sitegrouped"
GROUP_COL = "site_id"
```

This is the most valuable remaining internal test of generalization.

### 2. Patient-level permutation control

Run:

```python
RUN_TAG = "permuted"
PERMUTE_LABELS = True
```

The expected result is approximately chance performance.

### 3. External cohort

The strongest future validation would be an independent pathology cohort such as PAIP, MCO-CRC, or another accessible external dataset.

External validation is the experiment that would move the project from:

> “works within TCGA”

toward:

> “may generalize beyond TCGA.”

---

# 24. References

Core methods:

* Lu et al. — **CLAM: Weakly supervised learning for whole-slide image classification**
* Chen et al. — **UNI / pathology foundation-model work**
* Kather et al. — **Deep learning detection of MSI from histopathology**
* Relevant MSI-H / CRC clinical literature for the biomarker and treatment context

Data:

* TCGA-COAD
* cBioPortal / TCGA PanCancer Atlas clinical data
* Mahmood Lab `UNI2-h-features`

Implementation:

* Mahmood Lab CLAM repository
* `CLAM_SB` implementation at the pinned commit documented above

---

# 25. Reproducibility note

This project intentionally prioritizes **honest evaluation over maximizing the headline metric**.

The notebook therefore:

* uses patient-level grouping;
* keeps outer test patients isolated from model selection;
* calibrates within the training fold;
* reports fold-level variability;
* compares against a simple baseline;
* explicitly records limitations;
* distinguishes internal validation from external validation.

A higher AUROC obtained through a less rigorous evaluation design would not be considered an improvement.


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
