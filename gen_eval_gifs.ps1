# gen_eval_gifs.ps1
# Generate evaluation GIFs for all Sokoban conditions.
# Output: gym_sokoban/figures/gifs/

param(
    [int]$NEpisodes = 5,
    [int]$Fps       = 4,
    [int]$Scale     = 4
)

$OutDir = "gym_sokoban/figures/gifs"
New-Item -ItemType Directory -Force $OutDir | Out-Null

$VastRuns = "gym_sokoban/sokoban_results/runs"

$CKPT_BASELINE      = "$VastRuns/20260528_014748__baseline_minigridcnn_500k__Sokoban-small-v0__seed1/checkpoints/final.pt"
$CKPT_ORACLE_FREE   = "$VastRuns/20260529_023713__oracle_acc100_cost0_500k__Sokoban-small-v0__seed1/checkpoints/final.pt"
$CKPT_ORACLE_COST01 = "$VastRuns/20260528_022559__oracle_acc100_cost01_500k__Sokoban-small-v0__seed1/checkpoints/final.pt"
$CKPT_ORACLE_COST05 = "$VastRuns/20260528_004947__oracle_acc100_cost05_500k__Sokoban-small-v0__seed1/checkpoints/final.pt"
$CKPT_ORACLE_COST08 = "$VastRuns/20260528_033855__oracle_acc100_cost08_500k__Sokoban-small-v0__seed1/checkpoints/final.pt"
$CKPT_LINEAR        = "$VastRuns/20260529_001243__oracle_acc100_cost_linear_0to2_3M__Sokoban-small-v0__seed1/checkpoints/final.pt"

function Gen-Gif {
    param($Label, $Checkpoint, $ExtraArgs)
    $out = "$OutDir/$Label.gif"
    if (Test-Path $out) { Write-Host "Skipping $Label (already exists)"; return }
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

Gen-Gif "baseline"      $CKPT_BASELINE      "--no-oracle --stochastic --success-only"
Gen-Gif "oracle_free"   $CKPT_ORACLE_FREE   "--oracle-cost 0 --stochastic --success-only"
Gen-Gif "oracle_cost01" $CKPT_ORACLE_COST01 "--oracle-cost 0.1 --stochastic --success-only"
Gen-Gif "oracle_cost05" $CKPT_ORACLE_COST05 "--oracle-cost 0.5 --stochastic --success-only"
Gen-Gif "oracle_cost08" $CKPT_ORACLE_COST08 "--oracle-cost 0.8 --stochastic --success-only"
Gen-Gif "oracle_linear" $CKPT_LINEAR        "--oracle-cost 0 --stochastic --success-only"

# Zero-shot transfer to FixedTarget-Sokoban-v2
$CKPT_ACC75_C03 = "$VastRuns/20260528_044448__oracle_acc075_cost03_500k__Sokoban-small-v0__seed1/checkpoints/final.pt"
$CKPT_ACC50_C01 = "$VastRuns/20260528_005304__oracle_acc050_cost01_500k__Sokoban-small-v0__seed1/checkpoints/final.pt"

$TransferArgsSuccess = "--env-id FixedTarget-Sokoban-v2 --stochastic --success-only"
$TransferArgsAny     = "--env-id FixedTarget-Sokoban-v2 --stochastic"

Gen-Gif "transfer_fixedtarget_perfect"   $CKPT_ORACLE_COST01 "--oracle-cost 0.1 $TransferArgsSuccess"
Gen-Gif "transfer_fixedtarget_acc75_c03" $CKPT_ACC75_C03     "--oracle-cost 0.3 --oracle-accuracy 0.75 $TransferArgsAny"
Gen-Gif "transfer_fixedtarget_acc50_c01" $CKPT_ACC50_C01     "--oracle-cost 0.1 --oracle-accuracy 0.5  $TransferArgsAny"
Gen-Gif "transfer_fixedtarget_baseline"  $CKPT_BASELINE      "--no-oracle $TransferArgsAny"

Write-Host "`nAll GIFs saved to $OutDir"
Write-Host "Run export_to_website.ps1 to copy them to the website repo."
