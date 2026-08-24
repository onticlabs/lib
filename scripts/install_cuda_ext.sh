#!/usr/bin/env bash
# Build and install the optional CUDA extensions from ext/ into the active
# environment (or the one given by $PYTHON). Requires nvcc and torch.
#
#   ./scripts/install_cuda_ext.sh              # all three
#   ./scripts/install_cuda_ext.sh pointops     # just one
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
python="${PYTHON:-python}"

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    targets=(pointops point_rope point_serialization)
fi

"$python" -c "import torch" || { echo "torch must be installed first" >&2; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc not found — CUDA toolkit required" >&2; exit 1; }

for t in "${targets[@]}"; do
    echo "== building ext/$t"
    "$python" -m pip install --no-build-isolation "$here/ext/$t"
done
