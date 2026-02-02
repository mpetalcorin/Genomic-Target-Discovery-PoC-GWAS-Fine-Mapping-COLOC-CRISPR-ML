# Genomic-Target-Discovery-PoC-GWAS-Fine-Mapping-COLOC-CRISPR-ML
Reproducible, simulation-based end-to-end pipeline integrating GWAS summary stats, ABF fine-mapping, COLOC colocalization, CRISPR screen evidence, and calibrated ML to prioritize causal genes.

## Genomic Target Discovery, End-to-End Proof of Concept (GWAS, Fine-Mapping, COLOC, CRISPR, ML)
This repository is a reproducible, single-file proof of concept for **genetic and functional genomic evidence integration**, aligned to a modern **Genomic Technologies** workflow in translational target discovery. It simulates a locus-scale dataset with realistic structure, generates **GWAS summary statistics**, performs **ABF fine-mapping** and **COLOC-style colocalization**, adds **CRISPR knockout screen signals**, then trains **machine learning models** to prioritize candidate causal genes. The pipeline produces **decision-ready tables** and **figures** (PNG + PDF).

> This is a simulation-based workflow, designed for method development, testing, and demonstration. It does not use any patient-level or proprietary datasets.
Human genetics has become a central engine for target discovery, yet robust translation from association signals to actionable causal genes requires coordinated analysis across multiple evidence layers, including fine-mapping, regulatory genomics, perturbation screens, and predictive modelling. Here, a reproducible, simulation-based proof of concept that implements an end-to-end genomic target discovery workflow at a single locus is presented, designed to mirror practical tasks performed in translational genomics teams. The pipeline simulates a locus with linkage disequilibrium structure and common-variant allele frequencies, generates a polygenic quantitative trait with controllable heritability, and constructs cis-expression quantitative trait locus (cis-eQTL) summary statistics with a mixture distribution for allelic fold-change magnitudes consistent with population-scale regulatory genomics studies. From the simulated cohort, the GWAS summary statistics is computed, and the approximate Bayes factor (ABF) fine-mapping posterior inclusion probabilities (PIPs) is derived, and then the COLOC-style Bayesian colocalization to estimate posterior probabilities supporting shared causal variants between GWAS and eQTL signals is performed. In parallel, the pooled CRISPR knockout screen readouts that include essential-gene depletion and proliferation-suppressor enrichment phenotypes is simulated. These heterogeneous evidence types are integrated into a gene-level feature table, and supervised machine learning models are trained and evaluated using stratified cross-validation to predict high-priority candidate causal genes. The workflow produces decision-ready tables and visualizations, including Manhattan and QQ plots, PIP tracks, colocalization posterior summaries, CRISPR volcano plots, and model performance and calibration curves. This proof of concept provides a transparent template for method development, benchmarking, and onboarding, and it is readily extensible to real-world GWAS, xQTL, and perturbation datasets, enabling scalable evidence integration for target nomination and portfolio decisions.

## What this project demonstrates

- **Large-scale analysis outputs** from locus-level genetic data, including GWAS scan results.
- **State-of-the-art evidence types** used for target discovery:
  - Locus association (GWAS)
  - Fine-mapping (Wakefield ABF → SNP **PIPs**)
  - Colocalization (**COLOC ABF**, posterior **PP.H4**)
  - Functional genomics (simulated **CRISPR screen** readouts)
- **Integrated gene-level prioritization** across genetics, ‘omics, and perturbation evidence.
- **Predictive modelling** of causal gene likelihood using ML classifiers:
  - Logistic regression
  - Gradient boosting
  - Random forest
- **Robust evaluation** using stratified cross-validation and reliability metrics.
- **FAIR-style outputs**, saved as clean CSV tables, model metrics JSON, and figures.

### Tables (CSV)
- `snp_info.csv`  
  SNP metadata for the simulated locus (position, MAF, LD block).
- `gwas_sumstats.csv`  
  SNP-level GWAS summary statistics (beta, SE, z, p).
- `finemap_pips.csv`  
  Fine-mapping results with Wakefield ABF and SNP posterior inclusion probability (PIP).
- `eqtl_summary.csv`  
  Per-gene eQTL summary (lead SNP, log2 aFC, beta, SE, p).
- `coloc_posteriors.csv`  
  Per-gene COLOC posterior probabilities (H0–H4), with PP.H4 indicating shared causal variant evidence.
- `crispr_screen.csv`  
  Gene-level CRISPR knockout screen outputs (log2FC, p, class).
- `gene_level_features.csv`  
  Integrated gene-level feature table for prioritization and ML.
- `ml_predictions_cv.csv`  
  Cross-validated predictions across models and folds.
- `simulated_expression_matrix.csv`  
  Simulated expression matrix for the locus genes.

### Model metrics (JSON)
- `ml_metrics.json`  
  Mean ROC-AUC, average precision, Brier score, and class balance.

### Figures (PNG + PDF)
- `Fig1_locus_manhattan.(png|pdf)`  
  Locus Manhattan plot (GWAS).
- `Fig2_qq_plot.(png|pdf)`  
  GWAS QQ plot.
- `Fig3_finemap_pips.(png|pdf)`  
  SNP PIPs across the locus (fine-mapping).
- `Fig4_coloc_pp4.(png|pdf)`  
  Top PP.H4 genes (colocalization evidence).
- `Fig5_crispr_volcano.(png|pdf)`  
  CRISPR volcano plot (effect vs significance).
- `Fig6A_ml_roc.(png|pdf)`  
  ROC curves for ML models.
- `Fig6B_ml_pr.(png|pdf)`  
  Precision–Recall curves for ML models.
- `Fig6C_ml_calibration.(png|pdf)`  
  Calibration curves for ML models.
- `Fig7_feature_importance.(png|pdf)`  
  Permutation importance (which evidence drives predictions).

## Pipeline overview

1. **Simulate a genomic locus** with LD blocks and realistic minor allele frequencies.  
2. **Simulate cis-eQTL effects** for genes in the locus (including a fraction of larger allelic fold-changes).  
3. **Simulate a complex trait** with polygenic architecture and controllable heritability.  
4. **Run GWAS** to generate summary statistics.  
5. **Fine-map** using Wakefield approximate Bayes factors to compute SNP PIPs.  
6. **Colocalize** GWAS with each gene’s eQTL signal using COLOC ABF posteriors (PP.H4).  
7. **Simulate CRISPR perturbation effects** (essential genes and proliferation suppressors).  
8. **Integrate evidence** into gene-level features.  
9. **Train ML models** and evaluate with cross-validation.  
10. **Export tables and publication-ready figures**.

## Installation

This repository uses standard scientific Python libraries.

### Recommended environment
- Python 3.10–3.12
- NumPy
- pandas
- SciPy
- scikit-learn
- matplotlib

### Install with:
pip install numpy pandas scipy scikit-learn matplotlib

# Model Card, Genomic Target Discovery Classifier (Simulation-Based Proof of Concept)

## Model details

**Model name:** Genomic Target Discovery Classifier (PoC)  
**Version:** 1.0  
**Type:** Supervised binary classification, gene-level prioritization  
**Primary task:** Predict whether a gene in a locus is **likely causal** for a simulated trait based on integrated evidence from genetics, expression, and perturbation data.  
**Outputs:** A probability score `P(causal_gene=1)` for each gene, plus cross-validated predictions.

**Implemented models (baseline ensemble of approaches):**
- **Logistic regression** with feature scaling and class weighting.
- **Histogram-based gradient boosting** classifier.
- **Random forest** classifier with class weighting.

**Intended use setting:** Translational genomics, target discovery, and portfolio decision support workflows, demonstration and method development.

## Intended use

### Intended uses
- Demonstrate an end-to-end **evidence integration** pipeline, from locus simulation to gene prioritization.
- Provide a reproducible template that can be adapted to real datasets:
  - GWAS summary statistics
  - fine-mapping PIPs (e.g., SuSiE/FINEMAP outputs)
  - eQTL/xQTL summary statistics
  - CRISPR screen gene effect estimates
  - additional functional evidence (chromatin, protein interaction, pathway priors)

### Out of scope uses
- Clinical decision-making.
- Individual-level patient risk prediction.
- Any use where real-world harm could occur without extensive validation, governance, and domain review.

## Factors

The model score is driven by integrated locus-level and gene-level evidence. The main evidence factors represented in features are:

- **Genetic association strength** (e.g., `gwas_p_lead`, `gwas_z_lead`).
- **Fine-mapping confidence** (e.g., lead SNP `pip_lead` from ABF-based PIPs).
- **Colocalization evidence** (e.g., `coloc_pp4` representing COLOC posterior probability of shared causal variant).
- **Expression regulation magnitude** (e.g., `log2_aFC` for cis-eQTL allelic fold-change).
- **Perturbation effects** (e.g., `crispr_log2FC`, `crispr_p`).

These evidence types reflect a common translational logic: **genes are prioritized when association, causality, regulatory evidence, and functional impact are concordant.**

## Training data

### Data source
**Simulated data only.** No real patient-level or cohort data are used.

The pipeline simulates:
- A genomic locus with LD blocks and realistic minor allele frequencies.
- A complex quantitative trait with polygenic architecture and tunable heritability.
- cis-eQTL summary statistics for genes in the locus, including a mixture that yields a fraction of larger effect sizes.
- CRISPR knockout screen gene-level effects (including essential genes and proliferation suppressors).

### Label definition
The positive class (`label_causal_gene = 1`) is defined **within the simulation** by selecting a fixed number of top-ranked genes according to a multi-evidence heuristic score combining:
- colocalization PP.H4
- fine-mapping PIP of the gene’s lead SNP
- GWAS lead SNP significance
- eQTL magnitude
- CRISPR significance

This labeling is designed to:
- avoid cross-validation collapse from too few positives,
- produce a stable proof of concept for predictive modelling.

> In real-world target discovery, causal status is rarely known with certainty; labels should be derived from curated gold standards or prospective validation outcomes.

## Evaluation data

- Evaluation is performed using **stratified K-fold cross-validation**.
- `n_splits` is automatically constrained to be ≤ the minority class count.
- Metrics are aggregated across folds and reported in `ml_metrics.json`.

## Metrics

The following metrics are computed in cross-validation:

- **ROC-AUC**: measures ranking quality for binary classification.  
- **Average precision (AP)**: more informative under class imbalance, reflects precision–recall performance.  
- **Brier score**: measures probability calibration (lower is better).

The repository also produces diagnostic figures:
- ROC curves
- Precision–Recall curves
- Calibration curves
- Permutation feature importance

## Model performance (example)

Performance depends on the random seed and simulation settings. Typical output includes:

- Mean ROC-AUC across folds
- Mean AP across folds
- Mean Brier score across folds
- Class balance and fold count

## Limitations

- **Simulation-only**: results are illustrative, not evidence of real biological validity.
- **Simplified causal structure**:
  - Fine-mapping uses a single-causal ABF approximation, real loci often have multiple causal variants.
  - Colocalization uses idealized eQTL patterns, real xQTL signals can be complex and context-dependent.
- **CRISPR effects are stylized**:
  - Real screens are influenced by copy number artifacts, guide efficiency, cell line context, and experimental batch effects.
- **Label bias**:
  - Labels are generated from a heuristic score that partially overlaps with features, creating a best-case environment.
  - Real use should rely on independent labels where possible (validated causal genes, curated benchmarks, or prospective outcomes).

## Biases and fairness considerations

- This model is not trained on human demographic attributes, and the synthetic data does not represent population structure, ancestry differences, or sampling bias.
- When adapted to real biobank-scale data, key considerations include:
  - ancestry and population stratification controls,
  - differential power across allele frequency spectra,
  - phenotype heterogeneity and ascertainment bias,
  - tissue/cell-type specificity and batch effects.

## Explainability

Explainability is supported by:
- **Permutation importance** on the final fitted model (`Fig7_feature_importance.*`), showing which evidence types most influence the output score.

For real-world deployment, recommended additions include:
- SHAP value explanations per gene and per evidence channel,
- uncertainty quantification and applicability domain checks,
- sensitivity analyses across priors and modeling assumptions.

## Ethical and safety notes

- This repository is intended for research, engineering, and demonstration.
- It must not be used for clinical decisions or patient stratification without appropriate validation, governance, and regulatory review.
- When adapted to real data, consider privacy protection, secure compute environments, and data access controls.

## How to use the model output

The model produces gene-level probabilities and ranks. A practical interpretation is:

- **High probability**: gene is a strong candidate for follow-up based on concordant multi-evidence support.
- **Intermediate probability**: gene may be worth triage depending on biological plausibility and pathway context.
- **Low probability**: gene is less supported by integrated evidence within the modeled evidence types.

The output is intended to support:
- shortlist selection for experimental validation,
- portfolio discussion,
- target nomination prioritization,
- hypothesis generation.

## Reproducibility

- A single seed controls simulation and model randomness (`Config.seed`).
- Outputs are saved as deterministic filenames for versioning.

# Datasheet for Dataset, Genomic Target Discovery PoC (Simulated Multi-Modal Locus Data)

This datasheet documents the **simulated datasets** generated by the repository pipeline for an end-to-end proof of concept integrating **GWAS**, **fine-mapping**, **colocalization**, **cis-eQTL**, **CRISPR screening**, and **machine learning** at a locus level. It follows a dataset “datasheet” style format to support transparency, reproducibility, and safe reuse.

## 1. Motivation

### 1.1 Why was this dataset created?
To provide a **fully reproducible** and **self-contained** dataset for demonstrating and testing a modern genomic target discovery workflow, without using any patient-level data or restricted resources.

### 1.2 Who created the dataset?
Mark I.R. Petalcorin.

### 1.3 Who is it for?
- Scientists and engineers building workflows for:
  - genetic association interpretation,
  - gene prioritization,
  - evidence integration across genetic and functional genomics.
- Recruiters and collaborators assessing end-to-end translational genomics capability.
- Researchers who want a clean starting template for swapping in real data.

### 1.4 What tasks does it support?
- Generating GWAS-like summary statistics.
- Fine-mapping using approximate Bayes factors (ABF) and computing PIPs.
- COLOC-style colocalization (ABF) and PP.H4 prioritization.
- Simulating CRISPR gene knockout effects.
- Training and evaluating gene-level ML models for prioritization.
- Producing publication-quality figures and decision-ready tables.

## 2. Composition

### 2.1 What does the dataset contain?
The dataset is generated as a set of CSV tables
It contains:

- SNP-level metadata (`snp_info.csv`)
- GWAS summary statistics (`gwas_sumstats.csv`)
- Fine-mapping outputs (`finemap_pips.csv`)
- Per-gene cis-eQTL summaries (`eqtl_summary.csv`)
- Per-gene colocalization posteriors (`coloc_posteriors.csv`)
- Per-gene CRISPR screen outputs (`crispr_screen.csv`)
- Integrated gene-level feature table (`gene_level_features.csv`)
- ML cross-validated predictions (`ml_predictions_cv.csv`)
- Simulated expression matrix (`simulated_expression_matrix.csv`)

> Note: The genotype matrix is generated in-memory during execution and is not saved by default, to keep the repo lightweight.

### 2.2 What are the units of observation?
- SNP-level tables: each row represents a SNP at the simulated locus.
- Gene-level tables: each row represents a gene in the locus.
- Expression matrix: each row represents an individual, each column a gene expression feature.

### 2.3 Does it contain labels?
Yes, but **only in the simulation context**:
- `gene_level_features.csv` contains `label_causal_gene`, a synthetic label used for ML training and evaluation.

This label is **not biological ground truth**. It is derived from a combined heuristic score to create a stable proof-of-concept training target.

### 2.4 What is the approximate size?
Defaults (configurable in code):
- Individuals: 20,000
- SNPs: 4,000
- Genes: 25
- Expression matrix: 20,000 × 25

## 3. Data generation process

### 3.1 How was the dataset generated?
The dataset is produced by a single script that:

1. Simulates a locus with **LD blocks** and SNP **minor allele frequencies**.
2. Simulates **cis-eQTL** signals for locus genes (lead SNP per gene).
3. Simulates a **complex trait** with polygenic architecture and controllable heritability.
4. Runs a GWAS scan to compute SNP-level summary statistics.
5. Computes ABF-based fine-mapping PIPs.
6. Computes COLOC ABF posteriors (PP.H0–PP.H4) per gene.
7. Simulates pooled CRISPR knockout screen gene effects.
8. Integrates evidence into a gene-level feature table.
9. Trains ML models with stratified cross-validation.

### 3.2 Are there any external data sources?
No. The dataset is entirely synthetic and generated locally.

### 3.3 Is randomness involved?
Yes. Randomness is controlled by `Config.seed`. With the same seed and dependencies, the output is reproducible.

## 4. Intended uses

### 4.1 Primary intended uses
- Proof-of-concept demonstrations for:
  - genetics and genomics analytics,
  - evidence integration,
  - target discovery prioritization.
- Development and testing of:
  - plotting and reporting,
  - data engineering pipelines,
  - ML training and evaluation code,
  - software packaging patterns for genomics.

### 4.2 Secondary intended uses
- Teaching and training material for translational genomics methods.
- Template for replacing simulated inputs with real-world GWAS/xQTL/CRISPR outputs.

### 4.3 Out-of-scope uses
- Clinical prediction, patient stratification, medical diagnosis.
- Claims of biological truth or real gene causality.

## 5. Distribution

### 5.1 How is the dataset distributed?
The dataset is generated on demand by running the script. Outputs are saved as CSV, plus figure files (PNG/PDF).

## 6. Licensing
 MIT License for code; generated synthetic outputs can generally be shared freely. 

## 7. File-level documentation

All files are written to `genomic_technologies_poc_outputs/`.

### 7.1 `snp_info.csv`
**Purpose:** SNP metadata for the simulated locus.  
**Rows:** SNPs.

**Columns**
- `snp`: SNP identifier (e.g., `rsSIM00001`)
- `pos`: simulated genomic position
- `maf`: minor allele frequency (uniformly sampled within a configurable range)
- `ld_block`: integer LD block assignment

### 7.2 `gwas_sumstats.csv`
**Purpose:** SNP-level association results.  
**Rows:** SNPs.

**Columns**
- `snp`: SNP identifier
- `beta`: estimated effect size on the simulated trait
- `se`: standard error of the effect size estimate
- `z`: z-statistic
- `p`: two-sided p-value

### 7.3 `finemap_pips.csv`
**Purpose:** ABF fine-mapping outputs.  
**Rows:** SNPs.

**Columns**
- `snp`: SNP identifier
- `beta`, `se`, `z`, `p`: inherited from GWAS sumstats
- `abf`: approximate Bayes factor (Wakefield ABF)
- `pip`: posterior inclusion probability (normalized ABFs)

### 7.4 `eqtl_summary.csv`
**Purpose:** Per-gene cis-eQTL summaries.  
**Rows:** genes.

**Columns**
- `gene`: gene identifier (e.g., `GENE01`)
- `lead_snp`: lead cis-eQTL SNP for that gene
- `log2_aFC`: simulated log2 allelic fold-change
- `beta`: standardized expression effect size proxy
- `se`: standard error proxy
- `z`: z-statistic
- `p`: p-value
- `N`: simulated eQTL sample size
- `large_eqtl_flag`: indicator for larger-effect mixture component

### 7.5 `coloc_posteriors.csv`
**Purpose:** COLOC ABF posterior probabilities per gene.  
**Rows:** genes.

**Columns**
- `gene`: gene identifier
- `PP.H0`: no association with either trait
- `PP.H1`: GWAS-only association
- `PP.H2`: eQTL-only association
- `PP.H3`: both traits associated, distinct causal variants
- `PP.H4`: both traits associated, **shared causal variant** (key signal)

### 7.6 `crispr_screen.csv`
**Purpose:** Simulated pooled CRISPR knockout screen gene effects.  
**Rows:** genes.

**Columns**
- `gene`: gene identifier
- `log2FC`: simulated gene knockout effect on fitness/proliferation
- `z`: standardized effect
- `p`: p-value proxy
- `class`: simulated category (`essential`, `prolif_suppressor`, `nonessential`)

### 7.7 `gene_level_features.csv`
**Purpose:** Integrated gene-level table for prioritization and ML.  
**Rows:** genes.

**Columns**
- `gene`: gene identifier
- `lead_snp`: gene’s lead cis-eQTL SNP
- `gwas_p_lead`: GWAS p-value at the lead SNP
- `gwas_z_lead`: GWAS z-statistic at the lead SNP
- `pip_lead`: fine-mapping PIP at the lead SNP
- `coloc_pp4`: PP.H4 colocalization posterior
- `log2_aFC`: expression effect magnitude proxy
- `crispr_log2FC`: CRISPR effect size
- `crispr_p`: CRISPR p-value proxy
- `gwas_neglog10p_lead`: transformed GWAS p-value
- `crispr_neglog10p`: transformed CRISPR p-value
- `label_causal_gene`: synthetic positive/negative label for ML

### 7.8 `ml_predictions_cv.csv`
**Purpose:** Cross-validated model predictions.  
**Rows:** gene-fold pairs.

**Columns**
- `model`: model name (`logreg`, `hgb`, `rf`)
- `fold`: CV fold index
- `gene`: gene identifier
- `y_true`: true synthetic label
- `y_pred`: predicted probability

### 7.9 `simulated_expression_matrix.csv`
**Purpose:** Simulated expression matrix used to generate an expression-mediated trait component.  
**Rows:** individuals.  
**Columns:** genes (`GENE01`…).

## 8. Quality control

### 8.1 Validation checks performed
- Ensures output directories exist and writes are successful.
- Ensures cross-validation folds are feasible given class balance.
- Ensures GWAS summary statistics are well-defined (finite values).
- Generates QQ plot for sanity checking distribution.

### 8.2 Known failure modes
- Too few positives for CV if `n_positive_genes` is set too low.
- Extremely low p-values require numerical guards (handled by clipping in plotting).

## 9. Ethical considerations

- No human subjects data.
- No identifiable information.
- No clinical inferences.

If adapting to real datasets, ensure:
- appropriate governance, ethics approvals, and privacy protections,
- correct handling of population stratification and cohort biases.

## 10. References 
- Giambartolomei, C., et al. (2014). Bayesian test for colocalisation between pairs of genetic association studies using summary statistics. PLoS Genetics. https://pubmed.ncbi.nlm.nih.gov/24830394/  ￼
- Visscher, P. M., et al. (2017). 10 years of GWAS discovery, biology, function, and translation. American Journal of Human Genetics. https://pubmed.ncbi.nlm.nih.gov/28686856/  ￼
- GTEx Consortium. (2020). The GTEx Consortium atlas of genetic regulatory effects across human tissues. Science. https://pubmed.ncbi.nlm.nih.gov/32913098/  ￼
- Meyers, R. M., et al. (2017). Computational correction of copy number effect improves specificity of CRISPR-Cas9 essentiality screens in cancer cells. Nature Genetics. https://pubmed.ncbi.nlm.nih.gov/29083409/  ￼
- Doench, J. G., et al. (2016). Optimized sgRNA design to maximize activity and minimize off-target effects of CRISPR-Cas9. Nature Biotechnology. https://pubmed.ncbi.nlm.nih.gov/26780180/  ￼ 
