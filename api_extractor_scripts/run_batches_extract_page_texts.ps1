# run_batches_extract_page_texts.ps1
# Runs extract_page_texts.py in sequential batches of 1000 pages.
# Adjust $TOTAL_TITLES if you want to change the dataset size.

$PY = "python"
$SCRIPT = "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\extract_page_texts.py"
$OUT_DIR = "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\data\processed"
$TOTAL_TITLES = 12454
$BATCH_SIZE = 1000
$MAX_RETRIES = 5
$BACKOFF_BASE = 0.5
$BACKOFF_FACTOR = 2
$BACKOFF_MAX = 30
$BACKOFF_JITTER = 0.2
$SLEEP = 0.3

$BATCHES = [math]::Ceiling($TOTAL_TITLES / $BATCH_SIZE)
Write-Host "Total titles: $TOTAL_TITLES, batches: $BATCHES (size $BATCH_SIZE)"

# We already ran batch 0. Start from 1.
for ($i = 1; $i -lt $BATCHES; $i++) {
    $start = $i * $BATCH_SIZE
    $out = Join-Path $OUT_DIR "lexicanum_page_texts_batch_$i.json"
    Write-Host "Starting batch $i (start=$start, limit=$BATCH_SIZE) -> $out"
    & $PY $SCRIPT --start $start --limit $BATCH_SIZE --output $out --max-retries $MAX_RETRIES --backoff-base $BACKOFF_BASE --backoff-factor $BACKOFF_FACTOR --backoff-max $BACKOFF_MAX --backoff-jitter $BACKOFF_JITTER --sleep $SLEEP
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Batch $i failed with exit code $LASTEXITCODE. Stopping."
        break
    }
    Write-Host "Batch $i completed. Sleeping 5s before next batch..."
    Start-Sleep -Seconds 5
}

Write-Host "Batch run finished. Check $OUT_DIR for batch files."