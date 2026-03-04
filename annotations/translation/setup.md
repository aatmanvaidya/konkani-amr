### setup

```bash
git clone https://github.com/AI4Bharat/IndicTrans2
cd IndicTrans2/huggingface_interface
root_dir=$(pwd)
echo "Setting up the environment in $root_dir"
echo "Creating uv virtual environment (Python 3.10 recommended)"
pip install uv
uv venv --python 3.10
source .venv/bin/activate
echo "Upgrading pip"
uv pip install --upgrade pip
echo "Installing PyTorch"
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
echo "Installing utility packages"
uv pip install \
    nltk \
    sacremoses \
    pandas \
    regex \
    mock \
    "transformers>=4.33.2" \
    mosestokenizer
python -c "import nltk; nltk.download('punkt')"
echo "Installing acceleration packages"
uv pip install bitsandbytes scipy accelerate datasets wheel setuptools sentencepiece
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install flash_attn-2.5.5+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
echo "Cloning IndicTransToolkit"
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit
uv pip install -e .
cd "$root_dir"
```

```bash
git clone https://github.com/AI4Bharat/IndicTrans2
cd IndicTrans2/huggingface_interface
root_dir=$(pwd)
echo "Setting up the environment in $root_dir"
echo "Creating uv virtual environment (Python 3.10 recommended)"
pip install uv
echo "Upgrading pip"
uv pip install --upgrade pip --system
echo "Installing PyTorch"
uv pip install torch --index-url https://download.pytorch.org/whl/cu118 --system
echo "Installing utility packages"
uv pip install \
    nltk \
    sacremoses \
    pandas \
    regex \
    mock \
    "transformers>=4.33.2" \
    mosestokenizer --system
python -c "import nltk; nltk.download('punkt')"
echo "Installing acceleration packages"
uv pip install bitsandbytes scipy accelerate datasets wheel setuptools sentencepiece --system
echo "Installing flash-attn (optional)"
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu118torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
uv pip install flash_attn-2.5.5+cu118torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl --system
echo "Cloning IndicTransToolkit"
git clone https://github.com/VarunGumma/IndicTransToolkit
cd IndicTransToolkit
uv pip install -e . --system
cd "$root_dir"
```