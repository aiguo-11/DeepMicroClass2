import argparse
import os
import shutil
import subprocess
import sys
import tempfile


LABELS = ["arc", "bac", "chlor", "euk", "eukvir", "mit", "pls", "prokvir"]


def _ok(msg):
    print(f"[OK] {msg}")


def _warn(msg):
    print(f"[WARN] {msg}")


def _fail(msg):
    print(f"[FAIL] {msg}")


def check_python_deps():
    deps = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("pytorch_lightning", "pytorch_lightning"),
        ("torchmetrics", "torchmetrics"),
        ("sklearn", "scikit-learn"),
        ("pysam", "pysam"),
        ("Bio", "biopython"),
    ]

    ok = True
    for mod, display in deps:
        try:
            __import__(mod)
            _ok(f"Python dependency import succeeded: {display}")
        except Exception as e:
            ok = False
            _fail(f"Python dependency import failed: {display} ({e})")
    return ok


def check_model_files(base_dir):
    paths = [
        os.path.join(base_dir, "model", "8-class", "DeepMicroClass-best-500-8class.ckpt"),
        os.path.join(base_dir, "model", "8-class", "DeepMicroClass-best-1000-8class.ckpt"),
        os.path.join(base_dir, "model", "8-class", "DeepMicroClass-best-3000-8class.ckpt"),
        os.path.join(base_dir, "model", "300", "DeepMicroClass-best.ckpt"),
    ]
    ok = True
    for p in paths:
        if os.path.exists(p):
            _ok(f"Model file found: {os.path.relpath(p, base_dir)}")
        else:
            ok = False
            _fail(f"Model file missing: {os.path.relpath(p, base_dir)}")
    return ok


def check_minimap2(require=False):
    path = shutil.which("minimap2")
    if path:
        _ok(f"minimap2 is available: {path}")
        return True
    if require:
        _fail("minimap2 was not found (required for FASTQ/TPM workflows)")
        return False
    _warn("minimap2 was not found (contig-only classification can still run, but FASTQ/TPM needs it)")
    return True


def run_predict_smoke(base_dir):
    with tempfile.TemporaryDirectory(prefix="deepmicroclass2_selftest_") as td:
        contig = os.path.join(td, "test.fasta")
        out_dir = os.path.join(td, "out")
        seq = "A" * 500
        with open(contig, "w") as f:
            f.write(">seq1\n")
            f.write(seq + "\n")

        cmd = [
            sys.executable,
            os.path.join(base_dir, "predict.py"),
            "--contig",
            contig,
            "--model",
            "8class",
            "--out_dir",
            out_dir,
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if p.returncode != 0:
            _fail("predict.py failed during the smoke test; combined output follows")
            print(p.stdout.rstrip())
            return False

        class_file = os.path.join(out_dir, "classification.tsv")
        if not os.path.exists(class_file):
            _fail("classification.tsv was not generated")
            print(p.stdout.rstrip())
            return False

        with open(class_file, "r") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
        if len(lines) < 2:
            _fail("classification.tsv is empty or missing prediction rows")
            return False

        parts = lines[1].split("\t")
        if len(parts) < 3 or parts[1] not in LABELS:
            _fail("classification.tsv format is not valid")
            return False

        _ok(f"Inference smoke test passed: label={parts[1]} confidence={parts[2]}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="DeepMicroClass2 installation self-check (dependency import + inference smoke test)"
    )
    parser.add_argument(
        "--require-minimap2",
        action="store_true",
        help="Treat minimap2 as required as well (recommended when you need FASTQ/TPM workflows)",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    ok = True
    ok = check_python_deps() and ok
    ok = check_model_files(base_dir) and ok
    ok = check_minimap2(require=args.require_minimap2) and ok
    ok = run_predict_smoke(base_dir) and ok

    if ok:
        _ok("DeepMicroClass2 self-check passed")
        return 0
    _fail("DeepMicroClass2 self-check failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
