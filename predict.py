import os
import sys
import argparse
import numpy as np
import torch
from Bio import SeqIO
from models import DeepMicroClass, LightningDMC
from utils import set_seed, one_hot_encode, load_lengths, load_counts, calculate_tpm_logic, assign_label, run_alignment_pipeline

# Constants
LABELS = ["arc", "bac", "chlor", "euk", "eukvir", "mit", "pls", "prokvir"]
# Default thresholds for 8-class (Updated by user request)
DEFAULT_THRESHOLDS = [0.625000, 0.400000, 0.390625, 0.435547, 0.951172, 0.261719, 0.832031, 0.997925]
SEQ_ORDER = ["chlor", "eukvir", "prokvir", "mit", "pls", "euk", "arc", "bac"]

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_instance(ckpt_path, num_classes, device):
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found: {ckpt_path}")
        return None
    core = DeepMicroClass(num_classes=num_classes)
    try:
        model = LightningDMC.load_from_checkpoint(ckpt_path, model=core, num_classes=num_classes)
    except Exception as e:
        print(f"Error loading checkpoint {ckpt_path}: {e}")
        return None
    return model.eval().to(device)

def predict_batch_data(inputs, model, device):
    if len(inputs) == 0:
        return None
    # inputs is list of tensors (1, L, 4)
    # Stack: (B, 1, L, 4)
    x = torch.stack(inputs, dim=0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="DeepMicroClass2 Inference and TPM Calculation")
    parser.add_argument("--contig", required=True, help="Input Contig FASTA file")
    parser.add_argument("--fastq", nargs='+', help="Input FASTQ file(s) for alignment and TPM calculation")
    parser.add_argument("--model", choices=["8class", "300bp", "high_precision"], default="8class", help="Model type. '8class' is recommended. 'high_precision' uses overlapping windows and dynamic model selection.")
    parser.add_argument("--out_dir", default="result", help="Output directory")
    parser.add_argument("--thresholds", help="Comma separated thresholds (8 values). Default: 0.625,0.4,0.39,0.435,0.95,0.26,0.83,0.998")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
    
    # Parse thresholds
    thresholds = DEFAULT_THRESHOLDS
    if args.thresholds:
        try:
            thresholds = [float(x) for x in args.thresholds.split(',')]
            if len(thresholds) != 8: raise ValueError
        except:
            print("Warning: Invalid thresholds format. Using defaults.")
            thresholds = DEFAULT_THRESHOLDS

    # Read sequences and bucket by length
    print("Reading sequences...")
    # Suppress Biopython warning
    import warnings
    from Bio import BiopythonWarning
    warnings.simplefilter('ignore', BiopythonWarning)
    
    buckets = {500: [], 1000: [], 3000: [], '300': []} # '300' is for <500bp if model=300bp
    seq_map = {} # sid -> seq
    
    # We store SIDs in buckets to save memory, seqs in seq_map (or just re-read? re-reading is slower but saves RAM)
    # Let's store seqs in memory for now, assuming not huge genome.
    
    for record in SeqIO.parse(args.contig, "fasta"):
        seq = str(record.seq)
        L = len(seq)
        sid = record.id
        seq_map[sid] = seq
        
        if args.model == "300bp":
            # For 300bp model, we might accept everything >= 300?
            # Or just short ones? The prompt says "ultra-short sequence classifier (300bp)".
            # Usually for < 500bp.
            if L >= 300:
                buckets['300'].append(sid)
        elif args.model == "high_precision":
            if L >= 3000:
                buckets[3000].append(sid)
            elif L >= 1000:
                buckets[1000].append(sid)
            elif L >= 500:
                buckets[500].append(sid)
            elif L >= 300:
                buckets['300'].append(sid)
            else:
                pass # Skip < 300
        else:
            # 8class
            if L >= 3000:
                buckets[3000].append(sid)
            elif L >= 1000:
                buckets[1000].append(sid)
            elif L >= 500:
                buckets[500].append(sid)
            else:
                pass # Skip < 500
    
    results = {} # sid -> prob vector (8,)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    
    # Processing
    if args.model in ["8class", "high_precision"]:
        target_lengths = [3000, 1000, 500]
        if args.model == "high_precision":
            target_lengths.append(300) # '300' key in buckets corresponds to 300bp model

        for length in target_lengths:
            # Handle special key for 300bp
            bucket_key = '300' if length == 300 else length
            sids = buckets[bucket_key]
            if not sids: continue
            
            print(f"Processing {len(sids)} sequences with {length}bp model (Mode: {args.model})...")
            
            if length == 300:
                ckpt = os.path.join(model_dir, "300", "DeepMicroClass-best.ckpt")
            else:
                ckpt = os.path.join(model_dir, "8-class", f"DeepMicroClass-best-{length}-8class.ckpt")
            
            model = load_model_instance(ckpt, 8, device)
            if model is None: continue
            
            # Batch processing could be added here for large sets
            # For simplicity, we process all at once (might OOM if too many).
            # Let's do mini-batches of 128
            batch_size = 128
            
            # Determine stride
            if args.model == "high_precision":
                stride = int(length * 0.5)
            else:
                stride = length
            
            for i in range(0, len(sids), batch_size):
                batch_sids = sids[i:i+batch_size]
                batch_inputs = []
                valid_sids = []
                
                for sid in batch_sids:
                    seq = seq_map[sid]
                    # Slice windows. For inference we usually take windows and average.
                    windows = [seq[j:j+length] for j in range(0, len(seq)-length+1, stride)]
                    if not windows: continue
                    
                    # Encode
                    encoded = [one_hot_encode(w, length) for w in windows]
                    # We have multiple windows per sequence.
                    # We need to predict them all and average.
                    # This complicates batching across sequences.
                    # Simplification: Process one sequence at a time (contains multiple windows).
                    pass
                
                # Revert to sequence-by-sequence for variable number of windows
                pass
            
            # Sequence by sequence (safer for memory if windows are many)
            for sid in sids:
                seq = seq_map[sid]
                windows = [seq[j:j+length] for j in range(0, len(seq)-length+1, stride)]
                if not windows: continue
                
                encoded = [one_hot_encode(w, length) for w in windows]
                probs = predict_batch_data(encoded, model, device)
                if probs is not None:
                    avg_prob = np.mean(probs, axis=0)
                    results[sid] = avg_prob
            
            del model
            torch.cuda.empty_cache()

    elif args.model == "300bp":
        sids = buckets['300']
        if sids:
            print(f"Processing {len(sids)} sequences with 300bp model...")
            ckpt = os.path.join(model_dir, "300", "DeepMicroClass-best.ckpt")
            model = load_model_instance(ckpt, 8, device) # Assuming 8-class
            
            if model:
                for sid in sids:
                    seq = seq_map[sid]
                    # Sliding window of 300
                    windows = [seq[j:j+300] for j in range(0, len(seq)-300+1, 300)]
                    if not windows: continue
                    encoded = [one_hot_encode(w, 300) for w in windows]
                    probs = predict_batch_data(encoded, model, device)
                    if probs is not None:
                        avg_prob = np.mean(probs, axis=0)
                        results[sid] = avg_prob
                del model
                torch.cuda.empty_cache()

    # Output results
    print("Writing results...")

    # Classification TSV
    out_class_file = os.path.join(args.out_dir, "classification.tsv")
    with open(out_class_file, 'w') as f:
        f.write("contig\tlabel\tconfidence\t" + "\t".join(LABELS) + "\n")
        for sid, probs in results.items():
            pred_label = assign_label(probs, thresholds, SEQ_ORDER, LABELS)
            # Find confidence for the assigned label
            pred_idx = LABELS.index(pred_label)
            conf = probs[pred_idx]
            
            probs_str = "\t".join([f"{p:.4f}" for p in probs])
            f.write(f"{sid}\t{pred_label}\t{conf:.4f}\t{probs_str}\n")
    
    # TPM
    reads_file = None
    if args.fastq:
        print("Generating reads count from FASTQ...")
        try:
            reads_file = run_alignment_pipeline(args.contig, args.fastq, args.out_dir)
        except Exception as e:
            print(f"Error in alignment pipeline: {e}")
            reads_file = None

    if reads_file:
        print("Calculating TPM...")
        lengths = load_lengths(args.contig)
        counts = load_counts(reads_file)
        
        class_tpm = calculate_tpm_logic(results, lengths, counts, thresholds, SEQ_ORDER, LABELS)
        
        out_tpm_file = os.path.join(args.out_dir, "tpm_distribution.tsv")
        with open(out_tpm_file, 'w') as f:
            f.write("Class\tTPM\n")
            for label in LABELS:
                f.write(f"{label}\t{class_tpm[label]:.4f}\n")
            f.write(f"Total\t{sum(class_tpm.values()):.4f}\n")
            
    print("Done.")

if __name__ == "__main__":
    main()
