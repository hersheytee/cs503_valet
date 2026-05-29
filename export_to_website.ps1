# export_to_website.ps1
# Copy generated figures and GIFs from cs503_project to the website repo.

$WebsiteRepo = "..\CS503-VALET-Website"

# GIFs
$GifSrc = "gym_sokoban\figures\gifs"
$GifDst = "$WebsiteRepo\static\gifs\sokoban"
if (Test-Path $GifSrc) {
    New-Item -ItemType Directory -Force $GifDst | Out-Null
    Copy-Item "$GifSrc\*.gif" $GifDst -Force
    Write-Host "GIFs -> $GifDst"
} else {
    Write-Warning "No GIFs found at $GifSrc -- run gen_eval_gifs.ps1 first."
}

# Figures (plots)
$FigSrc = "gym_sokoban\figures"
$FigDst = "$WebsiteRepo\static\images"
if (Test-Path $FigSrc) {
    Copy-Item "$FigSrc\*.png" $FigDst -Force
    Write-Host "Figures -> $FigDst"
} else {
    Write-Warning "No figures found at $FigSrc -- run plot_results.py first."
}

Write-Host "Done. Commit and push the website repo to deploy."
