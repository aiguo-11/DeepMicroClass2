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

def assign_label(probs, thresholds, seq_order, labels):
    assigned = "bac"
    found = False
    for label in seq_order:
        idx = labels.index(label)
        if label == "bac":
            assigned = "bac"
            found = True
            break
        
        if probs[idx] >= thresholds[idx]:
            assigned = label
            found = True
            break
    
    if not found:
        assigned = "bac"
    return assigned

def calculate_tpm_logic(results, lengths, counts, thresholds, seq_order, labels):
    # results: dict id -> prob (8,)
    # lengths: dict id -> len
    # counts: dict id -> count
    
    # Calculate RPK
    rpk = {}
    for cid, L in lengths.items():
        if cid in counts and cid in results:
            length_kb = L / 1000.0
            cnt = counts.get(cid, 0)
            rpk[cid] = cnt / length_kb if length_kb > 0 else 0.0
            
    total_rpk = sum(rpk.values())
    scale = total_rpk / 1_000_000 if total_rpk > 0 else 1.0
    tpm = {cid: (val/scale) for cid, val in rpk.items()}
    
    class_tpm = {label: 0.0 for label in labels}
    
    # Strict Bac Fallback Logic (Seq mode)
    for cid, probs in results.items():
        if cid not in tpm: continue
        val_tpm = tpm[cid]
        
        assigned = assign_label(probs, thresholds, seq_order, labels)
            
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

def run_alignment_pipeline(contigs_file, fastq_files, output_dir, threads=8):
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
    cmd_index = ["minimap2", "-d", mmi_file, contigs_file]
    print(f"Running index: {' '.join(cmd_index)}")
    subprocess.check_call(cmd_index)
    
    # 2. Align
    sam_file = os.path.join(output_dir, "alignment.sam")
    # fastq_files is a list
    cmd_align = ["minimap2", "-ax", "sr", "-t", str(threads), mmi_file] + fastq_files
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
