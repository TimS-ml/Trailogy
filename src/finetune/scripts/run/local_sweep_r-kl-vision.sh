#!/bin/bash
# Local sweep — r64 / r256 LoRA grid, 30k steps, mix-50k-v2.
#
# Drives every yaml listed in configs/local_sweep/queue_r-kl-vision.txt
# (one per line, '#' comments + blank lines skipped). Each config is
# trained end-to-end via scripts/run/train.sh, then auto-eval runs (per
# the yaml's eval.enabled) unless --no-eval is passed.
#
# Usage:
#   bash scripts/run/local_sweep_r-kl-vision.sh
#   bash scripts/run/local_sweep_r-kl-vision.sh --no-eval
#   bash scripts/run/local_sweep_r-kl-vision.sh --queue path/to/other.txt
#
# Env vars (all optional):
#   HF_TOKEN              required for gated unsloth/gemma-4-E2B-it
#                         (only needed for cold model fetch — local cache
#                          works without it).
#   WANDB_MODE            default "offline" — write run dirs under
#                         wandb/ for later `wandb sync`. Set "online" to
#                         stream live (needs WANDB_API_KEY).
#   WANDB_PROJECT         default "trailogy-finetune"
#   WANDB_API_KEY         required only when WANDB_MODE=online.
#   CUDA_VISIBLE_DEVICES  default unset (use all visible GPUs)
#   FORCE_RESUME          set to 1 to resume from the latest matching
#                         outputs/<stem>_<TS>/checkpoint-<N> instead of
#                         minting a fresh timestamped folder. Off by
#                         default — fresh runs every time.
#
# Outputs:
#   outputs/<stem>_<TS>/                       per-run artifacts
#   outputs/_local-sweep-r-kl-vision.log       sweep-level tee
#   outputs/_local-sweep-r-kl-vision.summary.txt  one line per run
#
# Skipping individual runs: comment them out in the queue file. The
# script does not glob — the queue is the single source of truth.

set -uo pipefail
cd "$(dirname "$0")/../.."

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
NO_EVAL=""
QUEUE_FILE="configs/local_sweep/queue_r-kl-vision.txt"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-eval) NO_EVAL="--no-eval"; shift;;
        --queue)   QUEUE_FILE="$2"; shift 2;;
        --queue=*) QUEUE_FILE="${1#--queue=}"; shift;;
        --help|-h) sed -n '1,40p' "$0"; exit 0;;
        *)
            echo "ERROR: unknown flag: $1" >&2
            echo "  Edit ${QUEUE_FILE} to change which configs run." >&2
            exit 2
            ;;
    esac
done

if [ ! -f "$QUEUE_FILE" ]; then
    echo "ERROR: queue file not found: $QUEUE_FILE" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-trailogy-finetune}"

# ---------------------------------------------------------------------------
# Parse queue file → parallel arrays QUEUE_PATHS / QUEUE_EXTRAS.
# Syntax per line: <yaml_path> [--flag value ...]
# '#' comments + blank lines skipped. Non-existent yamls warn + skip.
# ---------------------------------------------------------------------------
QUEUE_PATHS=()
QUEUE_EXTRAS=()
while IFS= read -r raw || [ -n "$raw" ]; do
    # Trim leading/trailing whitespace.
    raw="${raw#"${raw%%[![:space:]]*}"}"
    raw="${raw%"${raw##*[![:space:]]}"}"
    [ -z "$raw" ] && continue
    [[ "$raw" == \#* ]] && continue

    yaml_path="${raw%% *}"
    if [ "$yaml_path" = "$raw" ]; then
        extras=""
    else
        extras="${raw#* }"
    fi
    if [ ! -f "$yaml_path" ]; then
        echo "WARN: queue entry not found, skipping: $yaml_path" >&2
        continue
    fi
    QUEUE_PATHS+=("$yaml_path")
    QUEUE_EXTRAS+=("$extras")
done < "$QUEUE_FILE"

if [ "${#QUEUE_PATHS[@]}" -eq 0 ]; then
    echo "ERROR: queue file $QUEUE_FILE has no valid yaml entries." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Optional auto-resume helper. Off by default; opt in via FORCE_RESUME=1.
# Matches outputs/<stem>_YYYYMMDD_HHMMSS/ and picks the latest folder
# that has at least one checkpoint-N/ subdir.
# ---------------------------------------------------------------------------
find_latest_run_dir() {
    local stem="$1"
    python - "$stem" <<'PY'
import re, sys
from pathlib import Path

stem = sys.argv[1]
ts_re = re.compile(r"^" + re.escape(stem) + r"_(\d{8})_(\d{6})$")

candidates = []
outputs = Path("outputs")
if not outputs.is_dir():
    sys.exit(0)
for folder in outputs.iterdir():
    if not folder.is_dir():
        continue
    m = ts_re.match(folder.name)
    if not m:
        continue
    best = (-1, None)
    for d in folder.glob("checkpoint-*"):
        if not d.is_dir():
            continue
        try:
            step = int(d.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if step > best[0]:
            best = (step, str(d))
    if best[1] is None:
        continue
    candidates.append((int(m.group(1)), int(m.group(2)), str(folder), best[1]))

if not candidates:
    sys.exit(0)
candidates.sort()
_, _, folder, ckpt = candidates[-1]
print(folder)
print(ckpt)
PY
}

# ---------------------------------------------------------------------------
# Sweep-level artifacts
# ---------------------------------------------------------------------------
mkdir -p outputs
SWEEP_LOG="outputs/_local-sweep-r-kl-vision.log"
SUMMARY_FILE="outputs/_local-sweep-r-kl-vision.summary.txt"

{
    echo ""
    echo "============================================================"
    echo "local_sweep_r-kl-vision — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "  queue file:   $QUEUE_FILE  (${#QUEUE_PATHS[@]} configs)"
    echo "  eval:         $([ -n "$NO_EVAL" ] && echo 'disabled (--no-eval)' || echo 'per yaml cfg.eval.enabled')"
    echo "  wandb mode:   $WANDB_MODE"
    echo "  wandb proj:   $WANDB_PROJECT"
    echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"
    echo "  FORCE_RESUME: ${FORCE_RESUME:-0}"
    echo "============================================================"
    echo "Queue (KL-first):"
    for i in "${!QUEUE_PATHS[@]}"; do
        extra_tag=""
        [ -n "${QUEUE_EXTRAS[$i]}" ] && extra_tag="  (${QUEUE_EXTRAS[$i]})"
        echo "  [$((i+1))/${#QUEUE_PATHS[@]}] ${QUEUE_PATHS[$i]}${extra_tag}"
    done
    echo "============================================================"
} | tee -a "$SWEEP_LOG"

# ---------------------------------------------------------------------------
# Drive each config sequentially. A failed config doesn't kill the queue.
# ---------------------------------------------------------------------------
total="${#QUEUE_PATHS[@]}"
ok_runs=0
failed_runs=0

for idx in "${!QUEUE_PATHS[@]}"; do
    yaml_path="${QUEUE_PATHS[$idx]}"
    extras="${QUEUE_EXTRAS[$idx]}"
    stem=$(basename "$yaml_path" .yaml)
    pos=$((idx+1))

    # Resume detection — opt-in.
    run_name=""
    resume_args=()
    if [ "${FORCE_RESUME:-0}" = "1" ]; then
        resume_blob=$(find_latest_run_dir "$stem" || true)
        if [ -n "$resume_blob" ]; then
            existing_dir=$(printf '%s\n' "$resume_blob" | sed -n '1p')
            existing_ckpt=$(printf '%s\n' "$resume_blob" | sed -n '2p')
            run_name=$(basename "$existing_dir")
            resume_args=(--resume_from_checkpoint "$existing_ckpt")
        fi
    fi
    if [ -z "$run_name" ]; then
        ts=$(date -u +%Y%m%d_%H%M%S)
        run_name="${stem}_${ts}"
    fi
    output_dir="outputs/$run_name"

    {
        echo ""
        echo "============================================================"
        echo "[${pos}/${total}] $(date -u +%H:%M:%SZ)  $yaml_path"
        echo "  stem:        $stem"
        echo "  run_name:    $run_name"
        echo "  output_dir:  $output_dir"
        if [ "${#resume_args[@]}" -gt 0 ]; then
            echo "  RESUME from: ${resume_args[1]}"
        fi
        if [ -n "$extras" ]; then
            echo "  per-entry:   $extras"
        fi
        echo "============================================================"
    } | tee -a "$SWEEP_LOG"

    # Per-run wandb id so re-runs / resumes reuse the same run.
    export WANDB_RUN_ID="$run_name"
    export WANDB_NAME="$run_name"
    export WANDB_RESUME="allow"

    train_args=("$yaml_path" "--run_name" "$run_name")
    [ -n "$NO_EVAL" ] && train_args+=("$NO_EVAL")
    if [ "${#resume_args[@]}" -gt 0 ]; then
        train_args+=("${resume_args[@]}")
    fi
    # Per-entry overrides go last so they win on argparse last-writer-wins.
    if [ -n "$extras" ]; then
        # shellcheck disable=SC2206
        train_args+=($extras)
    fi

    bash scripts/run/train.sh "${train_args[@]}" 2>&1 | tee -a "$SWEEP_LOG"
    rc=${PIPESTATUS[0]}
    unset WANDB_RUN_ID WANDB_NAME WANDB_RESUME

    if [ "$rc" -eq 0 ]; then
        ok_runs=$((ok_runs+1))
        printf "  %-60s  OK\n" "$yaml_path" >> "$SUMMARY_FILE"
        echo "[${pos}/${total}] $yaml_path  --> OK" | tee -a "$SWEEP_LOG"
    else
        failed_runs=$((failed_runs+1))
        printf "  %-60s  FAILED (rc=%d)\n" "$yaml_path" "$rc" >> "$SUMMARY_FILE"
        echo "[${pos}/${total}] $yaml_path  --> FAILED (rc=$rc)" | tee -a "$SWEEP_LOG"
    fi
done

{
    echo ""
    echo "============================================================"
    echo "local_sweep_r-kl-vision DONE — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "  configs:    $total"
    echo "  OK:         $ok_runs"
    echo "  FAILED:     $failed_runs"
    echo "  summary:    $SUMMARY_FILE"
    echo "  sweep log:  $SWEEP_LOG"
    echo "============================================================"
} | tee -a "$SWEEP_LOG"

[ "$failed_runs" -eq 0 ]
