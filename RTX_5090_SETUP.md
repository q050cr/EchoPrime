# RTX 5090 Setup for EchoPrime

## Installation

RTX 5090 (sm_120) requires PyTorch nightly with CUDA 12.8:

```bash
uv sync
uv pip install --prerelease=allow --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision
```

## Running Scripts

**Recommended:** Use `--no-sync` to prevent PyTorch downgrade:

```bash
uv run --no-sync scripts/your_script.py <args>
```

**Alternative:** Run directly with venv python:

```bash
.venv/bin/python scripts/your_script.py <args>
```

**Note:** Standard `uv run` (without `--no-sync`) will downgrade PyTorch to stable version

## Verification

```bash
.venv/bin/python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Compute: {torch.cuda.get_device_capability(0)}')"
```

Expected output: `GPU: NVIDIA GeForce RTX 5090`, `Compute: (12, 0)`

## Critical Dependencies

- PyTorch nightly (2.11.0.dev+cu128)
- PyWavelets==1.4.1 (exact version required)
- opencv-python-headless==4.5.5.64
- Python 3.11 (not 3.12)

## References

- [PyTorch sm_120 Support](https://github.com/pytorch/pytorch/issues/159207)
- [RTX 5090 Discussion](https://discuss.pytorch.org/t/nvidia-geforce-rtx-5090/218954)
- [EchoPrime PyWavelets Fix](https://github.com/echonet/EchoPrime/issues/15)
