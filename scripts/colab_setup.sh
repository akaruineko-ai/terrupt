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

# --- Torch: never force-overwrite an existing CUDA-enabled build ---
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "torch: $(python3 -c "import torch; print(torch.__version__)") (CUDA OK, keeping existing build)"
else
    echo "torch: CUDA not available — installing CUDA build from PyTorch index"
    CUDA_TAG="${TORCH_CUDA_INDEX_TAG:-cu124}"
    pip install -q --upgrade torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
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
    print('  This is a CPU-only / misbuilt torch. Fix:')
    print('  pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124')
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
