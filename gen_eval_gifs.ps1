# gen_eval_gifs.ps1
# Generate evaluation GIFs for all Sokoban conditions.
# Fill in checkpoint paths before running.
# Output: gym_sokoban/figures/final/gifs/

param(
    [int]$NEpisodes = 5,
    [int]$Fps       = 4,
    [int]$Scale     = 4
)

$OutDir = "gym_sokoban/figures/final/gifs"
New-Item -ItemType Directory -Force $OutDir | Out-Null

# ── Fill in checkpoint paths ─────────────────────────────────────────────────
$CKPT_BASELINE     = "gym_sokoban/sokoban_vast_results/runs/<baseline_run>/checkpoints/final.pt"
$CKPT_ORACLE_FREE  = "gym_sokoban/sokoban_vast_results/runs/<oracle_free_run>/checkpoints/final.pt"
$CKPT_ORACLE_COST05= "gym_sokoban/sokoban_vast_results/runs/<oracle_cost05_run>/checkpoints/final.pt"
$CKPT_BUDGET1      = "runs/<budget1_run>/checkpoints/final.pt"
$CKPT_BUDGET3      = "runs/<budget3_run>/checkpoints/final.pt"
$CKPT_BUDGET5      = "runs/<budget5_run>/checkpoints/final.pt"
# ─────────────────────────────────────────────────────────────────────────────

function Gen-Gif {
    param($Label, $Checkpoint, $ExtraArgs)
    $out = "$OutDir/$Label.gif"
    Write-Host "Generating $Label ..."
    $cmd = "python gym_sokoban/eval_gif.py " +
           "--checkpoint `"$Checkpoint`" " +
           "--n-episodes $NEpisodes --fps $Fps --scale $Scale " +
           "--out `"$out`" $ExtraArgs"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  -> $out"
    } else {
        Write-Warning "  FAILED: $Label"
    }
}

Gen-Gif "baseline"      $CKPT_BASELINE      "--no-oracle"
Gen-Gif "oracle_free"   $CKPT_ORACLE_FREE   ""
Gen-Gif "oracle_cost05" $CKPT_ORACLE_COST05 ""
Gen-Gif "budget1"       $CKPT_BUDGET1       "--max-oracle-queries 1"
Gen-Gif "budget3"       $CKPT_BUDGET3       "--max-oracle-queries 3"
Gen-Gif "budget5"       $CKPT_BUDGET5       "--max-oracle-queries 5"

Write-Host "`nAll GIFs saved to $OutDir"
Write-Host "Run export_to_website.ps1 to copy them to the website repo."
