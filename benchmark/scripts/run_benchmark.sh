#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh — Chay benchmark nhieu YOLO model tuan tu tren SCoralDet
#   Moi model: Train → Evaluate tu dong, lap qua nhieu seed de tinh mean±std
#
# Usage:
#   bash benchmark/scripts/run_benchmark.sh
#   bash benchmark/scripts/run_benchmark.sh --epochs 50 --imgsz 1280
#   bash benchmark/scripts/run_benchmark.sh --seeds 0,1,2
#   bash benchmark/scripts/run_benchmark.sh --eval-split val
#   bash benchmark/scripts/run_benchmark.sh --skip-eval
#   bash benchmark/scripts/run_benchmark.sh --dry-run
# =============================================================================
set -euo pipefail

# ── Yeu cau server: chi dung GPU1 ────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="1"
export PYTHONIOENCODING="utf-8"

# ── Default parameters ───────────────────────────────────────────────────────
EPOCHS=100
IMGSZ=640
BATCH=16
DEVICE="0"          # index 0 vi CUDA_VISIBLE_DEVICES=1 map GPU vat ly 1 -> index 0
WORKERS=0
PROJECT="runs/coral_benchmark"
EVAL_SPLIT="test"
SEEDS=(0 1 2)
LR0=0.001
LRF=0.01
SKIP_EVAL=false
DRY_RUN=false

# ── Parse CLI arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)      EPOCHS="$2";      shift 2 ;;
        --imgsz)       IMGSZ="$2";       shift 2 ;;
        --batch)       BATCH="$2";       shift 2 ;;
        --device)      DEVICE="$2";      shift 2 ;;
        --workers)     WORKERS="$2";     shift 2 ;;
        --project)     PROJECT="$2";     shift 2 ;;
        --eval-split)  EVAL_SPLIT="$2";  shift 2 ;;
        --seeds)       IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --lr0)         LR0="$2";         shift 2 ;;
        --lrf)         LRF="$2";         shift 2 ;;
        --skip-eval)   SKIP_EVAL=true;   shift ;;
        --dry-run)     DRY_RUN=true;     shift ;;
        *)             echo "[ERR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Danh sach model muon benchmark ───────────────────────────────────────────
# Bo comment dong nao muon chay.
# Custom weights: dung prefix __custom__:path/to/weights.pt
MODELS=(
    # ── YOLOv8 ──────────────────────────
    "yolov8n"
    "yolov8s"
    # "yolov8m"
    # "yolov8l"
    # "yolov8x"

    # ── YOLOv10 ─────────────────────────
    "yolov10n"
    "yolov10s"
    # "yolov10m"

    # ── YOLO11 ──────────────────────────
    "yolo11n"
    "yolo11s"
    # "yolo11m"
    # "yolo11l"
    # "yolo11x"

    # ── YOLO12 ──────────────────────────
    "yolo12n"
    "yolo12s"
    # "yolo12m"
    # "yolo12l"
    # "yolo12x"

    # ── YOLO26 (ultralytics hub) ─────────
    "yolo26n"
    "yolo26s"
    # "yolo26m"
    # "yolo26l"
    # "yolo26x"

    # ── RT-DETR (transformer-based) ──────
    "rtdetr-r50"
    # "rtdetr-r101"
)
# ─────────────────────────────────────────────────────────────────────────────

# ── ANSI colors ──────────────────────────────────────────────────────────────
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
MAGENTA='\033[0;35m'
NC='\033[0m'  # No Color

write_header() { echo -e "\n${CYAN}$(printf '=%.0s' {1..62})\n  $1\n$(printf '=%.0s' {1..62})${NC}"; }
write_ok()     { echo -e "${GREEN}[OK]  $1${NC}"; }
write_err()    { echo -e "${RED}[ERR] $1${NC}"; }
write_info()   { echo -e "${YELLOW}[..]  $1${NC}"; }

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ ${#MODELS[@]} -eq 0 ]]; then
    write_err "Khong co model nao duoc chon. Mo run_benchmark.sh va bo comment model muon chay."
    exit 1
fi

LOG_DIR="${PROJECT}/_logs"
if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$LOG_DIR"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_LOG="${LOG_DIR}/summary_${TIMESTAMP}.txt"

write_header "SCoralDet YOLO Benchmark"
write_info "Models    : ${#MODELS[@]} models  x  ${#SEEDS[@]} seeds ($(IFS=','; echo "${SEEDS[*]}"))"
write_info "Epochs    : $EPOCHS  |  ImgSz: $IMGSZ  |  Batch: $BATCH"
write_info "LR        : lr0=$LR0  lrf=$LRF  |  Workers: $WORKERS"
write_info "Device    : $DEVICE  |  Project: $PROJECT"
write_info "Evaluate  : split=$EVAL_SPLIT  SkipEval=$SKIP_EVAL"
if [[ "$DRY_RUN" == true ]]; then
    echo -e "  ${MAGENTA}[DRY RUN]${NC}"
fi

# ── Run loop ─────────────────────────────────────────────────────────────────
# Results stored as lines: Model|Seed|Train|Eval|mAP50|mAP50_95|Duration
RESULTS_FILE=$(mktemp)
TOTAL=$(( ${#MODELS[@]} * ${#SEEDS[@]} ))
SUCCESS=0
FAILED=0
START_TIME=$(date +%s)
RUN_NUM=0

for MODEL_ENTRY in "${MODELS[@]}"; do
for SEED in "${SEEDS[@]}"; do
    RUN_NUM=$((RUN_NUM + 1))

    # Phan biet hub model vs custom weights
    if [[ "$MODEL_ENTRY" == __custom__:* ]]; then
        WEIGHTS_PATH="${MODEL_ENTRY#__custom__:}"
        MODEL_LABEL=$(basename "$WEIGHTS_PATH" .pt)
        TRAIN_CMD="python benchmark/scripts/3_train_baseline.py \
            --weights \"$WEIGHTS_PATH\" \
            --epochs $EPOCHS --imgsz $IMGSZ --batch $BATCH \
            --lr0 $LR0 --lrf $LRF --seed $SEED \
            --device $DEVICE --workers $WORKERS --project \"$PROJECT\""
    else
        MODEL_LABEL="$MODEL_ENTRY"
        TRAIN_CMD="python benchmark/scripts/3_train_baseline.py \
            --model $MODEL_ENTRY \
            --epochs $EPOCHS --imgsz $IMGSZ --batch $BATCH \
            --lr0 $LR0 --lrf $LRF --seed $SEED \
            --device $DEVICE --workers $WORKERS --project \"$PROJECT\""
    fi

    # Path cua best.pt sau khi train
    # ultralytics them "detect/" prefix
    RUN_NAME="${MODEL_LABEL}_imgsz${IMGSZ}_ep${EPOCHS}_lr0${LR0}_lrf${LRF}_s${SEED}"
    BEST_PT="runs/detect/${PROJECT}/${RUN_NAME}/weights/best.pt"
    EVAL_CMD="python benchmark/scripts/4_evaluate.py \
        --weights \"$BEST_PT\" \
        --split $EVAL_SPLIT --imgsz $IMGSZ --device $DEVICE \
        --out_dir \"$LOG_DIR\""

    write_header "[$RUN_NUM/$TOTAL] $MODEL_LABEL  (seed=$SEED)"
    write_info "TRAIN : $TRAIN_CMD"
    if [[ "$SKIP_EVAL" == false ]]; then
        write_info "EVAL  : $EVAL_CMD"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        EVAL_STATUS="DRY_RUN"
        [[ "$SKIP_EVAL" == true ]] && EVAL_STATUS="SKIP"
        echo "${MODEL_LABEL}|${SEED}|DRY_RUN|${EVAL_STATUS}|-|-|-" >> "$RESULTS_FILE"
        continue
    fi

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    TRAIN_LOG="${LOG_DIR}/${MODEL_LABEL}_s${SEED}_train_${TIMESTAMP}.txt"
    RUN_START=$(date +%s)
    TRAIN_EXIT=0

    eval "$TRAIN_CMD" 2>&1 | tee "$TRAIN_LOG" || TRAIN_EXIT=$?

    RUN_END=$(date +%s)
    TRAIN_DUR=$(awk "BEGIN {printf \"%.1f\", ($RUN_END - $RUN_START) / 60}")

    if [[ $TRAIN_EXIT -ne 0 ]]; then
        write_err "$MODEL_LABEL (seed=$SEED) TRAIN FAILED (exit=$TRAIN_EXIT) - xem log trong: $PROJECT/$RUN_NAME/"
        echo "${MODEL_LABEL}|${SEED}|FAIL|SKIP|-|-|${TRAIN_DUR}m" >> "$RESULTS_FILE"
        FAILED=$((FAILED + 1))
        continue   # Khong evaluate neu train that bai
    fi

    write_ok "$MODEL_LABEL (seed=$SEED) train OK in ${TRAIN_DUR}m"
    SUCCESS=$((SUCCESS + 1))

    # ── EVALUATE ──────────────────────────────────────────────────────────────
    EVAL_STATUS="SKIP"
    MAP50="-"
    MAP50_95="-"

    if [[ "$SKIP_EVAL" == false ]]; then
        if [[ ! -f "$BEST_PT" ]]; then
            write_err "best.pt not found: $BEST_PT"
            EVAL_STATUS="NO_WEIGHTS"
        else
            write_info "Evaluating '$MODEL_LABEL' (seed=$SEED) on $EVAL_SPLIT split..."
            EVAL_LOG="${LOG_DIR}/${MODEL_LABEL}_s${SEED}_eval_${TIMESTAMP}.txt"
            EVAL_EXIT=0

            eval "$EVAL_CMD" 2>&1 | tee "$EVAL_LOG" || EVAL_EXIT=$?

            if [[ $EVAL_EXIT -eq 0 ]]; then
                write_ok "$MODEL_LABEL (seed=$SEED) eval OK"
                # Doc mAP50/mAP50-95 tu JSON output cua 4_evaluate.py
                EVAL_JSON="${LOG_DIR}/eval_${RUN_NAME}_${EVAL_SPLIT}.json"
                if [[ -f "$EVAL_JSON" ]]; then
                    MAP50=$(python3 -c "import json; d=json.load(open('$EVAL_JSON')); print(round(d['mAP50'],4))" 2>/dev/null || echo "-")
                    MAP50_95=$(python3 -c "import json; d=json.load(open('$EVAL_JSON')); print(round(d['mAP50_95'],4))" 2>/dev/null || echo "-")
                fi
                EVAL_STATUS="OK"
            else
                write_err "$MODEL_LABEL (seed=$SEED) eval FAILED - log: $EVAL_LOG"
                EVAL_STATUS="FAIL"
            fi
        fi
    fi

    TOTAL_END=$(date +%s)
    TOTAL_DUR=$(awk "BEGIN {printf \"%.1f\", ($TOTAL_END - $RUN_START) / 60}")
    echo "${MODEL_LABEL}|${SEED}|OK (${TRAIN_DUR}m)|${EVAL_STATUS}|${MAP50}|${MAP50_95}|${TOTAL_DUR}m" >> "$RESULTS_FILE"
done
done

# ── Summary ───────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
TOTAL_DURATION=$(awk "BEGIN {printf \"%.1f\", ($END_TIME - $START_TIME) / 60}")

write_header "Benchmark Summary"

# Print results table
printf "\n%-20s %-6s %-18s %-10s %-10s %-12s %-10s\n" \
    "Model" "Seed" "Train" "Eval" "mAP50" "mAP50-95" "Duration"
printf "%-20s %-6s %-18s %-10s %-10s %-12s %-10s\n" \
    "-----" "----" "-----" "----" "-----" "--------" "--------"

while IFS='|' read -r model seed train eval map50 map5095 duration; do
    printf "%-20s %-6s %-18s %-10s %-10s %-12s %-10s\n" \
        "$model" "$seed" "$train" "$eval" "$map50" "$map5095" "$duration"
done < "$RESULTS_FILE"

echo -e "\n${CYAN}Total : $TOTAL   OK: $SUCCESS   FAILED: $FAILED   Time: ${TOTAL_DURATION}m${NC}"

if [[ "$DRY_RUN" == false ]]; then
    # Save summary to file
    {
        printf "%-20s %-6s %-18s %-10s %-10s %-12s %-10s\n" \
            "Model" "Seed" "Train" "Eval" "mAP50" "mAP50-95" "Duration"
        printf "%-20s %-6s %-18s %-10s %-10s %-12s %-10s\n" \
            "-----" "----" "-----" "----" "-----" "--------" "--------"
        while IFS='|' read -r model seed train eval map50 map5095 duration; do
            printf "%-20s %-6s %-18s %-10s %-10s %-12s %-10s\n" \
                "$model" "$seed" "$train" "$eval" "$map50" "$map5095" "$duration"
        done < "$RESULTS_FILE"
        echo ""
        echo "Total : $TOTAL   OK: $SUCCESS   FAILED: $FAILED   Time: ${TOTAL_DURATION}m"
    } > "$SUMMARY_LOG"
    write_info "Summary log  : $SUMMARY_LOG"
    write_info "Eval results : $LOG_DIR/eval_*_${EVAL_SPLIT}.json"
fi

# ── Aggregation: mean±std per model qua cac seed (ddof=1, khop statistics.stdev) ──
# Use Python for accurate float math (consistent with PS1 version)
AGG_CSV="${LOG_DIR}/benchmark_agg_${TIMESTAMP}.csv"

python3 - "$RESULTS_FILE" "$AGG_CSV" "$DRY_RUN" <<'PYEOF'
import csv, statistics, sys
from collections import defaultdict

results_file = sys.argv[1]
agg_csv      = sys.argv[2]
dry_run      = sys.argv[3] == "true"

# Parse results (only rows with Eval == "OK")
by_model = defaultdict(lambda: {"map50": [], "map5095": []})
with open(results_file) as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) < 7:
            continue
        model, seed, train, eval_status, map50, map5095, duration = parts
        if eval_status != "OK":
            continue
        if map50 != "-" and map5095 != "-":
            by_model[model]["map50"].append(float(map50))
            by_model[model]["map5095"].append(float(map5095))

if not by_model:
    sys.exit(0)

# Print aggregation table
sep = "=" * 62
print(f"\n\033[0;36m{sep}")
print(f"  Aggregation (mean +/- std qua seeds)")
print(f"{sep}\033[0m")
print(f"\n{'Model':<20} {'n':>3}  {'mAP50 mean':>10}  {'std':>8}  {'mAP50-95 mean':>14}  {'std':>8}")
print(f"{'-----':<20} {'---':>3}  {'----------':>10}  {'--------':>8}  {'--------------':>14}  {'--------':>8}")

agg_rows = []
for model, vals in by_model.items():
    n = len(vals["map50"])
    m50_mean = statistics.mean(vals["map50"])
    m50_std  = statistics.stdev(vals["map50"]) if n > 1 else 0.0
    m5095_mean = statistics.mean(vals["map5095"])
    m5095_std  = statistics.stdev(vals["map5095"]) if n > 1 else 0.0
    print(f"{model:<20} {n:>3}  {m50_mean:>10.4f}  {m50_std:>8.4f}  {m5095_mean:>14.4f}  {m5095_std:>8.4f}")
    agg_rows.append({
        "Model": model, "n": n,
        "mAP50_mean": round(m50_mean, 4), "mAP50_std": round(m50_std, 4),
        "mAP50_95_mean": round(m5095_mean, 4), "mAP50_95_std": round(m5095_std, 4),
    })

# Save CSV
if not dry_run and agg_rows:
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Model", "n", "mAP50_mean", "mAP50_std", "mAP50_95_mean", "mAP50_95_std"])
        w.writeheader()
        w.writerows(agg_rows)
    print(f"\n\033[0;33m[..]  Aggregation CSV : {agg_csv}\033[0m")
PYEOF

# Cleanup
rm -f "$RESULTS_FILE"
