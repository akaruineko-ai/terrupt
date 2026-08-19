#!/usr/bin/env bash
# Colab / cloud GPU setup for terrupt-textcorrupt training.
#
# Run this at the top of your Colab notebook before training:
#   !bash scripts/colab_setup.sh
#
# Or on a fresh cloud instance:
#   curl -sSf https://raw.githubusercontent.com/.../scripts/colab_setup.sh | bash
set -euo pipefail

echo "=== terrupt-textcorrupt: Colab / cloud GPU setup ==="

# --- HF CLI (provides `hf sync`) ---
if ! command -v hf &>/dev/null; then
    echo "Installing HuggingFace CLI..."
    curl -LsSf https://hf.co/cli/install.sh | bash
fi
echo "hf CLI: $(hf --version 2>/dev/null || echo 'installed')"

# --- Python deps (torch handled separately: keep Colab's CUDA build) ---
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q sentencepiece sacrebleu peft accelerate datasets transformers

# --- Torch: keep/restore a CUDA-enabled build ---
TORCH_CUDA_TAG="${TORCH_CUDA_INDEX_TAG:-cu124}"
TORCH_IS_CPU=$(python3 -c "import torch; print('no' if getattr(torch, 'version', None).cuda else 'yes')" 2>/dev/null || echo "unknown")
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "torch: $(python3 -c "import torch; print(torch.__version__)") (CUDA OK, keeping existing build)"
elif [ "$TORCH_IS_CPU" = "yes" ]; then
    echo "torch: CPU-only build detected — installing CUDA build from PyTorch index"
    pip install -q --force-reinstall --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" torch
else
    echo "torch: $(python3 -c "import torch; print(torch.__version__)") has CUDA compiled but"
    echo "       torch.cuda.is_available() is False — driver/runtime problem, not a CPU build."
    if command -v nvidia-smi &>/dev/null; then
        echo "  nvidia-smi:"
        nvidia-smi 2>&1 | head -8 || true
    else
        echo "  nvidia-smi: NOT FOUND — NVIDIA driver is not installed/loaded in this session."
    fi
    echo "  This usually means the Colab runtime has no GPU attached."
    echo "  Fix: Runtime > Change runtime type > T4 GPU (or A100), then re-run this script."
    echo "  If the driver IS present, force a matching torch build with:"
    echo "    pip install --force-reinstall --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} torch"
fi

# --- Auth (optional, needed for private repos / bucket sync) ---
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN detected, logging in..."
    echo "$HF_TOKEN" | huggingface-cli login --token-stdin
else
    echo "Set HF_TOKEN env var if you need private repo access:"
    echo "  export HF_TOKEN=hf_xxxxx"
    echo "  huggingface-cli login"
fi

# --- Check GPU ---
python3 -c "
import torch
if not torch.cuda.is_available():
    print('WARNING: torch.cuda.is_available() is False!')
    print(f'  torch: {torch.__version__}')
    if torch.version.cuda:
        print('  Build has CUDA compiled in, but the driver/runtime is not usable.')
        print('  Check nvidia-smi above. Likely no GPU attached to this runtime,')
        print('  or a driver/torch mismatch.')
    else:
        print('  This is a CPU-only torch build.')
    print('  Fix: Runtime > Change runtime type > GPU, then re-run this script.')
    raise SystemExit(1)
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
cc = torch.cuda.get_device_capability(0)
bf16 = 'yes' if cc[0] >= 8 else 'no (use fp16)'
print(f'GPU: {name}  VRAM: {vram:.0f}GB  bf16: {bf16}')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Full 29M on A100 + 50GB checkpoint cap (local + bucket):"
echo "  python scripts/finetune.py \\"
echo "    --data akaruineko/terrupt-textcorrupt \\"
echo "    --target-rows -1 --epochs 1 --precision bf16 \\"
echo "    --hf-bucket hf://buckets/akaruineko/restratext \\"
echo "    --limit-checkpoints-folder 50gb"
echo ""
echo "Save every 2k steps instead of every epoch:"
echo "  python scripts/finetune.py \\"
echo "    --data akaruineko/terrupt-textcorrupt \\"
echo "    --target-rows -1 --precision bf16 \\"
echo "    --save-steps 2000 \\"
echo "    --hf-bucket hf://buckets/akaruineko/restratext \\"
echo "    --limit-checkpoints-folder 50gb"
echo ""
echo "Resume from checkpoint later:"
echo "  python scripts/finetune.py \\"
echo "    --data akaruineko/terrupt-textcorrupt \\"
echo "    --target-rows -1 --precision bf16 \\"
echo "    --hf-bucket hf://buckets/akaruineko/restratext \\"
echo "    --limit-checkpoints-folder 50gb \\"
echo "    --resume-from-checkpoint hf://buckets/akaruineko/restratext/checkpoint-XXXX"
