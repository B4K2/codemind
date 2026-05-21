#!/bin/bash
# CodeMind RunPod Setup Script
# Usage: bash setup.sh <YOUR_HF_TOKEN> <YOUR_WANDB_TOKEN>

set -e  # exit on any error

HF_TOKEN=${1:-""}
WANDB_TOKEN=${2:-""}
GITHUB_REPO="https://github.com/B4K2/codemind"

echo "============================================"
echo "   CodeMind Cloud Setup"
echo "============================================"

# ── 1. System check ───────────────────────────────────────────────────────────
echo ""
echo "▶ Checking system..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 --version
nvcc --version 2>/dev/null | grep release || echo "nvcc not in PATH (fine if torch has CUDA)"

# ── 2. Install uv ─────────────────────────────────────────────────────────────
echo ""
echo "▶ Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
else
    echo "uv already installed: $(uv --version)"
fi

# ── 3. Clone repo ─────────────────────────────────────────────────────────────
echo ""
echo "▶ Cloning CodeMind repository..."
if [ -d "codemind" ]; then
    echo "Directory exists — pulling latest..."
    cd codemind
    git pull
else
    git clone $GITHUB_REPO codemind
    cd codemind
fi

# ── 4. Python environment ─────────────────────────────────────────────────────
echo ""
echo "▶ Setting up Python environment..."
uv venv --python 3.12
source .venv/bin/activate

# ── 5. PyTorch — check if pre-installed, install if not ───────────────────────
echo ""
echo "▶ Checking PyTorch..."
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)")
    echo "PyTorch $TORCH_VER with CUDA $CUDA_VER already available ✅"
    # Still install into venv so imports work
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --quiet
else
    echo "Installing PyTorch with CUDA 12.8..."
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
fi

# ── 6. Core dependencies ──────────────────────────────────────────────────────
echo ""
echo "▶ Installing core dependencies..."
uv pip install \
    tiktoken \
    datasets \
    wandb \
    safetensors \
    tqdm \
    einops \
    ninja \
    packaging \
    python-dotenv \
    huggingface_hub \
    transformers

# ── 7. Flash Attention 2 ──────────────────────────────────────────────────────
echo ""
echo "▶ Installing Flash Attention 2 (this takes 5-10 mins, compiling from source)..."
uv pip install flash-attn --no-build-isolation

# ── 8. Credentials ────────────────────────────────────────────────────────────
echo ""
echo "▶ Setting up credentials..."

# Write .env file
cat > .env << EOF
HF_TOKEN=${HF_TOKEN}
WANDB_API_KEY=${WANDB_TOKEN}
EOF

# Login to services
if [ -n "$WANDB_TOKEN" ]; then
    wandb login $WANDB_TOKEN
    echo "W&B logged in ✅"
fi

if [ -n "$HF_TOKEN" ]; then
    python3 -c "from huggingface_hub import login; login('${HF_TOKEN}')"
    echo "HuggingFace logged in ✅"
fi

# ── 9. Verify everything works ────────────────────────────────────────────────
echo ""
echo "▶ Running verification..."
python3 - << 'PYEOF'
import torch
print(f"PyTorch:     {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(f"VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(f"tiktoken:    {len(enc.encode('hello world'))} tokens for test ✅")
try:
    import flash_attn
    print(f"FlashAttn:   {flash_attn.__version__} ✅")
except:
    print("FlashAttn:   not available ⚠ (will use standard attention)")
PYEOF

# ── 10. Quick model sanity check ──────────────────────────────────────────────
echo ""
echo "▶ Testing model forward pass..."
python3 -m model.codemind
python3 -m model.blocks.transformer_block

echo ""
echo "============================================"
echo "   Setup Complete!"
echo "   Start training with:"
echo "   python -m training.trainer"
echo "============================================"