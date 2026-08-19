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

# --- Python deps ---
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q sentencepiece sacrebleu peft accelerate datasets transformers torch

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
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    cc = torch.cuda.get_device_capability(0)
    bf16 = 'yes' if cc[0] >= 8 else 'no (use fp16)'
    print(f'GPU: {name}  VRAM: {vram:.0f}GB  bf16: {bf16}')
else:
    print('WARNING: No GPU detected!')
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
