#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Usage: ./run_train_test_fsa_dncnn.sh [--check | --smoke]

Train and test a PyTorch DnCNN baseline for FSA tx01/tx11 -> tx75.

Options:
  --check  Validate data pairing and split isolation only.
  --smoke  Run one-batch train/eval under /tmp.
  --help   Show this message.

Environment:
  PYTHON                Python executable (default: ../TCNS/venv/bin/python)
  DATA                  FSA dataset root
  OUTPUT                Output directory (default: results/fsa_dncnn_tx75)
  MPLCONFIGDIR          Matplotlib cache (default: /tmp/tcns-matplotlib)
  FORCE_TRAIN=1         Retrain even if model_best.pt already exists.
EOF
}

MODE="run"
case "${1:-}" in
  "")
    ;;
  --check)
    MODE="check"
    ;;
  --smoke)
    MODE="smoke"
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

export PYTHON="${PYTHON:-../TCNS/venv/bin/python}"
export DATA="${DATA:-../TCNS/data/mydata/fsa_dataset_fullres}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tcns-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found or not executable: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$DATA" ]]; then
  echo "Dataset directory not found: $DATA" >&2
  exit 1
fi

if [[ "$MODE" == "check" ]]; then
  "$PYTHON" train_fsa_dncnn.py \
    --data_dir "$DATA" \
    --input_levels tx01 tx11 \
    --target_level tx75 \
    --verify_data_only
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  OUTPUT="${OUTPUT:-/tmp/fsa_dncnn_tx75_smoke}"
  TRAIN_ARGS=(
    --epochs 1
    --repeat 1
    --batch_size 2
    --num_workers 0
    --max_train_batches 1
    --max_val_batches 1
  )
  TEST_ARGS=(
    --batch_size 2
    --max_cases 1
    --grid_cases_per_level 1
  )
else
  OUTPUT="${OUTPUT:-results/fsa_dncnn_tx75}"
  TRAIN_ARGS=(
    --epochs 80
    --repeat 8
    --batch_size 16
    --num_workers 4
  )
  TEST_ARGS=(
    --batch_size 8
  )
fi

BEST="$OUTPUT/model_best.pt"
if [[ -f "$BEST" && "${FORCE_TRAIN:-0}" != "1" ]]; then
  echo "[dncnn] skip training, checkpoint exists: $BEST"
else
  "$PYTHON" train_fsa_dncnn.py \
    --data_dir "$DATA" \
    --input_levels tx01 tx11 \
    --target_level tx75 \
    --crop_size 256 \
    --lr 1e-3 \
    --depth 17 \
    --features 64 \
    --loss mse \
    --val_fraction 0.15 \
    --output_dir "$OUTPUT" \
    "${TRAIN_ARGS[@]}"
fi

"$PYTHON" test_fsa_dncnn.py \
  --data_dir "$DATA" \
  --split test \
  --input_levels tx01 tx11 \
  --target_level tx75 \
  --crop_size 256 \
  --checkpoint "$BEST" \
  --output_dir "$OUTPUT/eval" \
  "${TEST_ARGS[@]}"

echo "[dncnn] done: $OUTPUT/eval"
