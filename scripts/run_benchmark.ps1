# =============================================================================
# run_benchmark.ps1 — Chạy benchmark nhiều YOLO model tuần tự trên SCoralDet
#   Mỗi model: Train → Evaluate tự động
#
# Usage:
#   .\scripts\run_benchmark.ps1
#   .\scripts\run_benchmark.ps1 -Epochs 50 -ImgSz 1280
#   .\scripts\run_benchmark.ps1 -EvalSplit val       # Evaluate trên val thay vì test
#   .\scripts\run_benchmark.ps1 -SkipEval            # Chỉ train, bỏ qua evaluate
#   .\scripts\run_benchmark.ps1 -DryRun              # In lệnh mà không chạy
# =============================================================================

param(
    [int]    $Epochs       = 100,
    [int]    $ImgSz        = 640,
    [int]    $Batch        = 16,
    [string] $Device       = "0",
    [int]    $Workers      = 0,
    [string] $Project      = "runs/coral_benchmark",
    [string] $EvalSplit    = "test",          # "val" hoặc "test"
    [switch] $SkipEval,                       # Bỏ qua evaluate sau train
    [switch] $DryRun                          # Chỉ in lệnh, không thực thi
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

# ── Danh sách model muốn benchmark ───────────────────────────────────────────
# Bỏ comment dòng nào muốn chạy.
# YOLO26: dùng prefix __custom__:path/to/weights.pt
$Models = @(
    # ── YOLOv8 ──────────────────────────
    "yolov8n",
    "yolov8s",
    # "yolov8m",
    # "yolov8l",
    # "yolov8x",

    # ── YOLOv10 ─────────────────────────
    "yolov10n",
    "yolov10s",
    # "yolov10m",

    # ── YOLO11 ──────────────────────────
    "yolo11n",
    "yolo11s",
    # "yolo11m",
    # "yolo11l",
    # "yolo11x",

    # ── YOLO12 ──────────────────────────
    "yolo12n",
    "yolo12s",
    # "yolo12m",
    # "yolo12l",
    # "yolo12x",

    # ── YOLO26 (ultralytics hub) ─────────
    "yolo26n",
    "yolo26s"
    # "yolo26m",
    # "yolo26l",
    # "yolo26x",

    # ── RT-DETR (transformer-based) ──────
    "rtdetr-r50"
    # "rtdetr-r101",

)
# ─────────────────────────────────────────────────────────────────────────────

# ── Helper ────────────────────────────────────────────────────────────────────
function Write-Header($msg) {
    $line = "=" * 62
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
}
function Write-Ok($msg)   { Write-Host "[OK]  $msg" -ForegroundColor Green  }
function Write-Err($msg)  { Write-Host "[ERR] $msg" -ForegroundColor Red    }
function Write-Info($msg) { Write-Host "[..]  $msg" -ForegroundColor Yellow }

# ── Validate ──────────────────────────────────────────────────────────────────
if ($Models.Count -eq 0) {
    Write-Err "Khong co model nao duoc chon. Mo run_benchmark.ps1 va bo comment model muon chay."
    exit 1
}

$LogDir = "$Project/_logs"
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

$Timestamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$SummaryLog = "$LogDir/summary_$Timestamp.txt"

Write-Header "SCoralDet YOLO Benchmark"
Write-Info "Models    : $($Models.Count) models"
Write-Info "Epochs    : $Epochs  |  ImgSz: $ImgSz  |  Batch: $Batch"
Write-Info "Device    : $Device  |  Project: $Project"
Write-Info "Evaluate  : split=$EvalSplit  SkipEval=$($SkipEval.IsPresent)"
if ($DryRun) { Write-Host "  [DRY RUN]" -ForegroundColor Magenta }

# ── Run loop ──────────────────────────────────────────────────────────────────
$Results = @()
$Total   = $Models.Count
$Success = 0
$Failed  = 0
$Start   = Get-Date

foreach ($ModelEntry in $Models) {
    $RunNum = $Results.Count + 1

    # Phân biệt hub model vs custom weights
    if ($ModelEntry.StartsWith("__custom__:")) {
        $WeightsPath = $ModelEntry.Substring("__custom__:".Length)
        $ModelLabel  = [System.IO.Path]::GetFileNameWithoutExtension($WeightsPath)
        $TrainCmd    = "python scripts/3_train_baseline.py" +
                       " --weights `"$WeightsPath`"" +
                       " --epochs $Epochs --imgsz $ImgSz --batch $Batch" +
                       " --device $Device --workers $Workers --project `"$Project`""
    } else {
        $ModelLabel = $ModelEntry
        $TrainCmd   = "python scripts/3_train_baseline.py" +
                      " --model $ModelEntry" +
                      " --epochs $Epochs --imgsz $ImgSz --batch $Batch" +
                      " --device $Device --workers $Workers --project `"$Project`""
    }

    # Path của best.pt sau khi train
    # ultralytics thêm "detect/" prefix → thực tế lưu ở runs/detect/<project>/<name>/
    $RunName = "${ModelLabel}_imgsz${ImgSz}_ep${Epochs}"
    $BestPt  = "runs/detect/$Project/$RunName/weights/best.pt"
    $EvalCmd = "python scripts/4_evaluate.py" +
               " --weights `"$BestPt`"" +
               " --split $EvalSplit --imgsz $ImgSz --device $Device" +
               " --out_dir `"$LogDir`""   # JSON lưu vào LogDir để đọc mAP50 vào summary

    Write-Header "[$RunNum/$Total] $ModelLabel"
    Write-Info "TRAIN : $TrainCmd"
    if (-not $SkipEval) {
        Write-Info "EVAL  : $EvalCmd"
    }

    if ($DryRun) {
        $Results += [PSCustomObject]@{
            Model    = $ModelLabel
            Train    = "DRY_RUN"
            Eval     = if ($SkipEval) { "SKIP" } else { "DRY_RUN" }
            mAP50    = "-"
            Duration = "-"
        }
        continue
    }

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    # Start-Transcript: vừa hiển thị đúng progress bar, vừa lưu log file
    $TrainLog = "$LogDir/${ModelLabel}_train_$Timestamp.txt"
    $RunStart = Get-Date

    try {
        Start-Transcript -Path $TrainLog -Append | Out-Null
        Invoke-Expression $TrainCmd
        $TrainExit = $LASTEXITCODE
    } catch {
        $TrainExit = 1
    } finally {
        Stop-Transcript | Out-Null
    }

    $TrainDur = [math]::Round(((Get-Date) - $RunStart).TotalMinutes, 1)

    if ($TrainExit -ne 0) {
        Write-Err "$ModelLabel TRAIN FAILED (exit=$TrainExit) - xem log trong: $Project/$RunName/"
        $Results += [PSCustomObject]@{
            Model    = $ModelLabel
            Train    = "FAIL"
            Eval     = "SKIP"
            mAP50    = "-"
            Duration = "${TrainDur}m"
        }
        $Failed++
        continue   # Không evaluate nếu train thất bại
    }

    Write-Ok "$ModelLabel train OK in ${TrainDur}m"
    $Success++

    # ── EVALUATE ──────────────────────────────────────────────────────────────
    $EvalStatus = "SKIP"
    $mAP50      = "-"

    if (-not $SkipEval) {
        if (-not (Test-Path $BestPt)) {
            Write-Err "best.pt not found: $BestPt"
            $EvalStatus = "NO_WEIGHTS"
        } else {
            Write-Info "Evaluating '$ModelLabel' on $EvalSplit split..."
            $EvalLog = "$LogDir/${ModelLabel}_eval_$Timestamp.txt"

            try {
                Invoke-Expression $EvalCmd | Tee-Object -FilePath $EvalLog
                $EvalExit = $LASTEXITCODE
            } catch {
                $EvalExit = 1
            }

            if ($EvalExit -eq 0) {
                Write-Ok "$ModelLabel eval OK"
                # Đọc mAP50 từ JSON output của 4_evaluate.py
                $EvalJson = "$LogDir/eval_${RunName}_${EvalSplit}.json"
                if (Test-Path $EvalJson) {
                    $mAP50 = [math]::Round((Get-Content $EvalJson | ConvertFrom-Json).mAP50, 4)
                }
                $EvalStatus = "OK"
            } else {
                Write-Err "$ModelLabel eval FAILED - log: $EvalLog"
                $EvalStatus = "FAIL"
            }
        }
    }

    $TotalDur = [math]::Round(((Get-Date) - $RunStart).TotalMinutes, 1)
    $Results += [PSCustomObject]@{
        Model    = $ModelLabel
        Train    = "OK (${TrainDur}m)"
        Eval     = $EvalStatus
        mAP50    = $mAP50
        Duration = "${TotalDur}m"
    }
}

# ── Summary ───────────────────────────────────────────────────────────────────
$TotalDuration = [math]::Round(((Get-Date) - $Start).TotalMinutes, 1)

Write-Header "Benchmark Summary"
$Results | Format-Table -AutoSize | Out-String | ForEach-Object { Write-Host $_ }
Write-Host "Total : $Total   OK: $Success   FAILED: $Failed   Time: ${TotalDuration}m" -ForegroundColor Cyan

if (-not $DryRun) {
    $Results | Format-Table -AutoSize | Out-String | Out-File $SummaryLog -Encoding utf8
    Write-Info "Summary log  : $SummaryLog"
    Write-Info "Eval results : $LogDir/eval_*_${EvalSplit}.json"
}
