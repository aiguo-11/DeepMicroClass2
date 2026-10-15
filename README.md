# DeepMicroClass2

DeepMicroClass2 is a deep learning based genome sequence classification tool for metagenomic contig labeling and abundance (TPM) estimation.

## Citation and Manuscript Status

DeepMicroClass2 extends the original DeepMicroClass framework for multi-class taxonomic classification of complex marine metagenomic sequences.  
If you use DeepMicroClass2, please cite the original DeepMicroClass publication:

> Hou, S., Tang, T., Cheng, S., Liu, Y., Xia, T., Chen, T., Fuhrman, J. A., & Sun, F.  
> DeepMicroClass sorts metagenomic contigs into prokaryotes, eukaryotes and viruses.  
> *NAR Genomics and Bioinformatics*, 6(2), lqae044, 2024.  
> https://doi.org/10.1093/nargab/lqae044

The manuscript describing DeepMicroClass2, entitled  
**“DeepMicroClass2: An Enhanced Deep Learning Framework for Multi-class Taxonomic Classification in Complex Marine Metagenomes”**,  
is currently under submission.

## Features

- **Multiple model modes**:
  - **8-class model (recommended)**: predicts 8 microbial categories (`arc`, `bac`, `chlor`, `euk`, `eukvir`, `mit`, `pls`, `prokvir`).
  - **Ultra-short sequence classifier**: designed for sequences around 300 bp.
- **Flexible inputs**:
  - Contig FASTA only: outputs contig classification results.
  - Contig FASTA + FASTQ: runs alignment, read counting, and 8-class TPM estimation automatically.
- **Configurable fallback policy**: sequences that do not pass their thresholds are assigned to `bac` by default, or `NA` with `--low-confidence-label NA`.
- **GPU acceleration**: CUDA is detected and used automatically when available.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/DeepMicroClass2.git
   cd DeepMicroClass2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Install a PyTorch build that matches your CUDA version if GPU support is needed.*

3. Run the installation self-check (recommended):
   ```bash
   python selftest.py
   ```
   If you also need FASTQ/TPM workflows, run the stricter check:
   ```bash
   python selftest.py --require-minimap2
   ```

## Usage

### 1. Quick Start

Run the default 8-class model:

```bash
python predict.py --contig examples/contigs_8class.fasta --out_dir result
```

### 2. TPM Estimation

If you also have raw FASTQ reads:

```bash
python predict.py --contig examples/contigs_8class.fasta --fastq examples/reads_1.fastq examples/reads_2.fastq --out_dir result
```

TPM alignment and filtering parameters can be customized:

```bash
python predict.py --contig input.fa --fastq reads_1.fastq reads_2.fastq --out_dir result --tpm-threads 16 --tpm-preset sr --tpm-min-mapq 20 --tpm-exclude-secondary --tpm-exclude-supplementary --tpm-min-contig-length 1000 --tpm-min-reads 5
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--tpm-threads` | `8` | Positive number of alignment threads |
| `--tpm-preset` | `sr` | minimap2 preset: `sr`, `map-ont`, `map-pb`, or `map-hifi` |
| `--tpm-min-mapq` | `0` | Minimum mapping quality (inclusive, 0–255) |
| `--tpm-exclude-secondary` | off | Exclude secondary alignments from counts |
| `--tpm-exclude-supplementary` | off | Exclude supplementary alignments from counts |
| `--tpm-min-contig-length` | `0` | Minimum contig length in bp (inclusive) |
| `--tpm-min-reads` | `0` | Minimum count per contig after alignment filtering (inclusive) |

These options apply when `--fastq` is supplied. Defaults retain the previous
counting and TPM behavior. Counts represent mapped alignment records, so paired
mates are counted individually; secondary and supplementary records are included
unless explicitly excluded. MAPQ 255 is retained when it meets the threshold.
TPM is calculated over classified contigs passing the length and count filters:
`RPK = count / (length_bp / 1000)` and `TPM = RPK / sum(RPK) * 1,000,000`.
Filtering changes both the numerator and normalization denominator. Class TPM
values sum to 1,000,000 when retained counts are positive, or zero otherwise.
The existing `--thresholds` option controls class assignment; TPM filters do not
change `classification.tsv`.

### 3. Other Model Modes

- **Ultra-short sequences (300 bp)**:
  ```bash
  python predict.py --contig examples/contigs_300bp.fasta --model 300bp
  ```

- **High precision mode**:
  ```bash
  python predict.py --contig examples/contigs_high_precision.fasta --model high_precision
  ```

### 4. Custom Thresholds

You can override the 8 class thresholds with `--thresholds` in the following order:
`arc, bac, chlor, euk, eukvir, mit, pls, prokvir`.

```bash
# Example: change thresholds explicitly
python predict.py --contig input.fa --thresholds 0.625000,0.400000,0.390625,0.435547,0.951172,0.261719,0.832031,0.997925
```

### 5. Low-confidence assignments

Use `--low-confidence-label NA` to label a contig `NA` when none of the eight
class probabilities meets its corresponding threshold:

```bash
python predict.py --contig input.fa --out_dir result --low-confidence-label NA
```

This mode applies the bacterial threshold as well as the other seven thresholds.
Classes passing thresholds still follow the existing priority order. Threshold
equality counts as passing, and `--thresholds` can customize these cutoffs.
The default `--low-confidence-label bac` preserves the original bacterial fallback.
For `NA` rows in `classification.tsv`, `confidence` is the maximum of the eight
model probabilities, not a probability for an additional trained class. The eight
probability columns are unchanged. Contigs skipped by length or without model
predictions remain omitted; this option only rejects existing predictions.

When FASTQ is supplied, `tpm_distribution.tsv` includes an `NA` row in NA mode.
NA contigs remain in the TPM normalization denominator and their TPM is assigned
to this row, so the total across the eight classes plus NA remains 1,000,000 when
retained counts are positive. NA is an unassigned result, not a ninth trained class.

## Example Data

The `examples/` directory contains small synthetic files that can be used to validate every supported workflow:

- `contigs_8class.fasta`: synthetic contigs for the default 8-class model and TPM workflow.
- `contigs_300bp.fasta`: short synthetic contigs for the `300bp` model.
- `contigs_high_precision.fasta`: mixed-length synthetic contigs for the `high_precision` mode.
- `reads_1.fastq` and `reads_2.fastq`: paired-end reads derived from `contigs_8class.fasta`.

## Output Files

The output directory may contain:

- `classification.tsv`: predicted label, confidence, and per-class probabilities for each contig.
- `tpm_distribution.tsv`: 8-class TPM summary when FASTQ reads are provided, with an additional `NA` row when NA fallback is enabled.

## Labels

- **arc**: Archaea
- **bac**: Bacteria
- **chlor**: Chloroplast
- **euk**: Eukaryote
- **eukvir**: Eukaryotic Virus
- **mit**: Mitochondrion
- **pls**: Plasmid
- **prokvir**: Prokaryotic Virus
