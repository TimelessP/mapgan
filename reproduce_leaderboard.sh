#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TARGET_URL="https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
EXPECTED_TARGET_SHA256="bc2356a26a2976f98e4aaf1b24c5693d5a4dc9b6178aeb952dbafbcd42c73bcd"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is required to create .venv" >&2
        exit 1
    fi
    python3 -m venv "$ROOT_DIR/.venv"
fi

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

echo "==> Installing pinned runtime"
"$PYTHON_BIN" -m pip install --upgrade "numpy==2.4.3" "Pillow==12.1.1"

echo "==> Fetching target map"
"$PYTHON_BIN" mapgan.py fetch-target \
    --width 256 \
    --height 128 \
    --target-url "$TARGET_URL" \
    --refresh

ACTUAL_TARGET_SHA256="$($PYTHON_BIN - <<'PY'
from hashlib import sha256
from pathlib import Path

print(sha256(Path("data/countries.geo.json").read_bytes()).hexdigest())
PY
)"

if [[ "$ACTUAL_TARGET_SHA256" != "$EXPECTED_TARGET_SHA256" ]]; then
    echo "Target checksum mismatch: expected $EXPECTED_TARGET_SHA256, got $ACTUAL_TARGET_SHA256" >&2
    exit 1
fi

echo "==> Running full leaderboard sweep"
"$PYTHON_BIN" mapgan.py solve \
    --min-blobs 0 \
    --max-blobs 48 \
    --population 48 \
    --steps 24 \
    --refine-passes 4 \
    --seed 7 \
    --search-width 128 \
    --search-height 64 \
    --render-width 256 \
    --render-height 128 \
    --stop-on-overfit \
    --overfit-min-delta 0.001 \
    --overfit-train-delta 0.003 \
    --overfit-verify-delta 0.003 \
    --overfit-patience 4 \
    --overfit-min-blobs 6 \
    --compress-best \
    --compress-min-blobs 6 \
    --dense-fraction 0.92 \
    --target-url "$TARGET_URL"

echo "==> Verifying saved best-overall model"
BEST_OVERALL_VERIFY="$($PYTHON_BIN mapgan.py verify \
    --model out/best_overall.json \
    --width 256 \
    --height 128 \
    --target-url "$TARGET_URL")"
printf '%s\n' "$BEST_OVERALL_VERIFY"

echo "==> Verifying saved best-dense model"
BEST_DENSE_VERIFY="$($PYTHON_BIN mapgan.py verify \
    --model out/best_dense.json \
    --width 256 \
    --height 128 \
    --target-url "$TARGET_URL")"
printf '%s\n' "$BEST_DENSE_VERIFY"

echo "==> Validating leaderboard summary"
"$PYTHON_BIN" - <<'PY'
import json
import math
from pathlib import Path

OLD_BEST_ACCURACY = 0.78143310546875
OLD_BEST_IOU = 0.5102905982905983

EXPECTED = {
    "best_verify_blob_count": 24,
    "best_render_accuracy": 0.8876953125,
    "best_render_iou": 0.7031539888682746,
    "best_dense_blobs": 8,
    "best_dense_params": 49,
    "best_dense_accuracy": 0.862274169921875,
    "best_dense_iou": 0.6478620474406991,
    "smallest_beating_old_blobs": 6,
    "smallest_beating_old_params": 37,
    "smallest_beating_old_accuracy": 0.841033935546875,
    "smallest_beating_old_iou": 0.6129727320008916,
}


def assert_close(name: str, actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise SystemExit(f"{name}: expected {expected}, got {actual}")


report = json.loads(Path("out/leaderboard.json").read_text())

if report["best_verify_blob_count"] != EXPECTED["best_verify_blob_count"]:
    raise SystemExit(
        f"best_verify_blob_count: expected {EXPECTED['best_verify_blob_count']}, got {report['best_verify_blob_count']}"
    )

best_render = report["best_render_metrics"]
assert_close("best_render_metrics.accuracy", best_render["accuracy"], EXPECTED["best_render_accuracy"])
assert_close("best_render_metrics.iou", best_render["iou"], EXPECTED["best_render_iou"])

best_dense = report["best_dense"]
if best_dense["blobs"] != EXPECTED["best_dense_blobs"]:
    raise SystemExit(f"best_dense.blobs: expected {EXPECTED['best_dense_blobs']}, got {best_dense['blobs']}")
if best_dense["params"] != EXPECTED["best_dense_params"]:
    raise SystemExit(f"best_dense.params: expected {EXPECTED['best_dense_params']}, got {best_dense['params']}")
assert_close("best_dense.verify.accuracy", best_dense["verify"]["accuracy"], EXPECTED["best_dense_accuracy"])
assert_close("best_dense.verify.iou", best_dense["verify"]["iou"], EXPECTED["best_dense_iou"])

candidates = sorted(report["compression"] + report["results"], key=lambda row: (row["params"], row["verify"]["iou"]))
smallest = next(
    row
    for row in candidates
    if row["verify"]["accuracy"] > OLD_BEST_ACCURACY and row["verify"]["iou"] > OLD_BEST_IOU
)

if smallest["blobs"] != EXPECTED["smallest_beating_old_blobs"]:
    raise SystemExit(
        f"smallest_beating_old.blobs: expected {EXPECTED['smallest_beating_old_blobs']}, got {smallest['blobs']}"
    )
if smallest["params"] != EXPECTED["smallest_beating_old_params"]:
    raise SystemExit(
        f"smallest_beating_old.params: expected {EXPECTED['smallest_beating_old_params']}, got {smallest['params']}"
    )
assert_close("smallest_beating_old.verify.accuracy", smallest["verify"]["accuracy"], EXPECTED["smallest_beating_old_accuracy"])
assert_close("smallest_beating_old.verify.iou", smallest["verify"]["iou"], EXPECTED["smallest_beating_old_iou"])

print(
    json.dumps(
        {
            "best_overall": {
                "blobs": report["best_verify_blob_count"],
                "accuracy": best_render["accuracy"],
                "iou": best_render["iou"],
            },
            "best_dense": {
                "blobs": best_dense["blobs"],
                "params": best_dense["params"],
                "accuracy": best_dense["verify"]["accuracy"],
                "iou": best_dense["verify"]["iou"],
            },
            "smallest_beating_old_best": {
                "blobs": smallest["blobs"],
                "params": smallest["params"],
                "accuracy": smallest["verify"]["accuracy"],
                "iou": smallest["verify"]["iou"],
                "model": smallest["model"],
            },
        },
        indent=2,
    )
)
PY

echo "==> Reproduction completed successfully"