import os
import subprocess
import shutil
import sys
try:
    import pysam
except ImportError:
    pysam = None
import random
import numpy as np
import torch
from Bio import SeqIO

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_lengths(fasta_file):
    lengths = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        lengths[record.id] = len(record.seq)
    return lengths

def load_counts(count_file):
    counts = {}
    with open(count_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            contig_id, count = parts[0], parts[1]
            if contig_id == '*':
                continue
            try:
                counts[contig_id] = int(count)
            except ValueError:
                counts[contig_id] = 0
    return counts

NUCLEOTIDES = "ACGT"
BASE2IDX = {b: i for i, b in enumerate(NUCLEOTIDES)}

def one_hot_encode(seq, length):
    seq = seq.upper()
    tensor = np.zeros((length, 4), dtype=np.float32)
    for i in range(min(len(seq), length)):
        base = seq[i]
        if base in BASE2IDX:
            tensor[i, BASE2IDX[base]] = 1.0
        else:
            tensor[i] = [0.25] * 4
    return torch.tensor(tensor).unsqueeze(0)  # (1, L, 4)

def assign_label(probs, thresholds, seq_order, labels, low_confidence_label="bac"):
    """Use class thresholds in priority order; optionally reject as NA.

    In NA mode bacteria must also pass their threshold. Legacy mode retains
    the unconditional bacterial fallback.
    """
    if low_confidence_label not in ("bac", "NA"):
        raise ValueError("low_confidence_label must be 'bac' or 'NA'")
    for label in seq_order:
        idx = labels.index(label)
        if label == "bac" and low_confidence_label == "bac":
            return "bac"
        if probs[idx] >= thresholds[idx]:
            return label
    return low_confidence_label

def calculate_tpm_logic(results, lengths, counts, thresholds, seq_order, labels,
                        min_contig_length=0, min_reads=0, low_confidence_label="bac"):
    # results: dict id -> prob (8,)
    # lengths: dict id -> len
    # counts: dict id -> count
    
    # Calculate RPK
    rpk = {}
    for cid, L in lengths.items():
        if cid in counts and cid in results and L >= min_contig_length and counts[cid] >= min_reads:
            length_kb = L / 1000.0
            cnt = counts.get(cid, 0)
            rpk[cid] = cnt / length_kb if length_kb > 0 else 0.0
            
    total_rpk = sum(rpk.values())
    scale = total_rpk / 1_000_000 if total_rpk > 0 else 1.0
    tpm = {cid: (val/scale) for cid, val in rpk.items()}
    
    class_tpm = {label: 0.0 for label in labels}
    if low_confidence_label == "NA":
        class_tpm["NA"] = 0.0
    
    # Use the same assignment policy as classification.tsv.
    for cid, probs in results.items():
        if cid not in tpm: continue
        val_tpm = tpm[cid]
        
        assigned = assign_label(probs, thresholds, seq_order, labels, low_confidence_label)
            
        class_tpm[assigned] += val_tpm
        
    return class_tpm

def check_alignment_tools():
    """Check if minimap2 and pysam are available."""
    minimap2_path = shutil.which("minimap2")
    if not minimap2_path:
        raise RuntimeError("minimap2 not found in PATH. Please install minimap2 (e.g., conda install minimap2).")
    if pysam is None:
        raise RuntimeError("pysam not installed. Please install pysam (pip install pysam).")
    return minimap2_path

def run_alignment_pipeline(contigs_file, fastq_files, output_dir, threads=8,
                           preset="sr", min_mapq=0, exclude_secondary=False,
                           exclude_supplementary=False):
    """
    Runs alignment pipeline:
    1. minimap2 index
    2. minimap2 align
    3. samtools sort (via pysam)
    4. samtools index (via pysam)
    5. samtools idxstats (via pysam)
    
    Returns path to counts file.
    """
    check_alignment_tools()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Index
    mmi_file = os.path.join(output_dir, "ref.mmi")
    # Only build index if not exists or force? Let's build every time to be safe or check timestamp.
    # For simplicity, build every time.
    # Keep the original index settings for the default short-read workflow.
    index_options = [] if preset == "sr" else ["-x", preset]
    cmd_index = ["minimap2"] + index_options + ["-d", mmi_file, contigs_file]
    print(f"Running index: {' '.join(cmd_index)}")
    subprocess.check_call(cmd_index)
    
    # 2. Align
    sam_file = os.path.join(output_dir, "alignment.sam")
    # fastq_files is a list
    cmd_align = ["minimap2", "-ax", preset, "-t", str(threads), mmi_file] + fastq_files
    print(f"Running align: {' '.join(cmd_align)}")
    with open(sam_file, "w") as f:
        subprocess.check_call(cmd_align, stdout=f)
        
    # 3. Sort
    bam_file = os.path.join(output_dir, "alignment.bam")
    print(f"Sorting to {bam_file}...")
    pysam.sort("-o", bam_file, sam_file)
    
    # 4. Index BAM
    print("Indexing BAM...")
    pysam.index(bam_file)
    
    # 5. Idxstats
    print("Counting reads...")
    if min_mapq or exclude_secondary or exclude_supplementary:
        counts = {}
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped or read.mapping_quality < min_mapq:
                    continue
                if exclude_secondary and read.is_secondary:
                    continue
                if exclude_supplementary and read.is_supplementary:
                    continue
                cid = read.reference_name
                counts[cid] = counts.get(cid, 0) + 1
        with open(os.path.join(output_dir, "reads_count.tsv"), "w") as f:
            for cid, count in counts.items():
                f.write(f"{cid}\t{count}\n")
        return os.path.join(output_dir, "reads_count.tsv")

    idxstats_str = pysam.idxstats(bam_file)
    
    counts_file = os.path.join(output_dir, "reads_count.tsv")
    with open(counts_file, "w") as f:
        for line in idxstats_str.splitlines():
            parts = line.split('\t')
            # parts: refname, seqlen, mapped, unmapped
            if len(parts) >= 3:
                # Output format: contig_id <tab> count
                # idxstats: refname, seqlen, mapped, unmapped
                # We want mapped reads? Yes.
                # parts[2] is mapped reads count.
                f.write(f"{parts[0]}\t{parts[2]}\n")
                
    return counts_file
