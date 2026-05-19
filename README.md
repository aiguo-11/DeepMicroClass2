# DeepMicroClass2

DeepMicroClass2 is a deep learning based genome sequence classification tool for metagenomic contig labeling and abundance (TPM) estimation.

## Features

- **Multiple model modes**:
  - **8-class model (recommended)**: predicts 8 microbial categories (`arc`, `bac`, `chlor`, `euk`, `eukvir`, `mit`, `pls`, `prokvir`).
  - **Ultra-short sequence classifier**: designed for sequences around 300 bp.
- **Flexible inputs**:
  - Contig FASTA only: outputs contig classification results.
  - Contig FASTA + FASTQ: runs alignment, read counting, and 8-class TPM estimation automatically.
- **Strict fallback policy**: sequences that do not pass their thresholds are assigned to `bac`.
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

## Example Data

The `examples/` directory contains small synthetic files that can be used to validate every supported workflow:

- `contigs_8class.fasta`: synthetic contigs for the default 8-class model and TPM workflow.
- `contigs_300bp.fasta`: short synthetic contigs for the `300bp` model.
- `contigs_high_precision.fasta`: mixed-length synthetic contigs for the `high_precision` mode.
- `reads_1.fastq` and `reads_2.fastq`: paired-end reads derived from `contigs_8class.fasta`.

## Output Files

The output directory may contain:

- `classification.tsv`: predicted label, confidence, and per-class probabilities for each contig.
- `tpm_distribution.tsv`: 8-class TPM summary when FASTQ reads are provided.

## Labels

- **arc**: Archaea
- **bac**: Bacteria
- **chlor**: Chloroplast
- **euk**: Eukaryote
- **eukvir**: Eukaryotic Virus
- **mit**: Mitochondrion
- **pls**: Plasmid
- **prokvir**: Prokaryotic Virus

## License

[Your License Here]
