# zip_results.ps1 — pack a PySR run's text results into one small archive
# =============================================================================
# Keeps only .txt and .csv. On the Mase run that is 0.7 MB against about 45 MB
# for the whole folder, because the diagnostic PNGs and the .pkl model caches
# are 95% of the size and are not needed to read a result.
#
# The archive is written NEXT TO the run folder, named after it, so it sits with
# the results rather than on the Desktop.
#
# USAGE
#   .\zip_results.ps1 "C:\...\rerun_05Aug2026\run_20260805_122821"
#   .\zip_results.ps1 "C:\...\rerun_05Aug2026"        # picks the newest run_* inside
#   .\zip_results.ps1                                  # picks the newest run_* under $DefaultRoot
#
# If PowerShell blocks the file, run it once as:
#   powershell -ExecutionPolicy Bypass -File .\zip_results.ps1 "<path>"
#
# Author: Jef Zerrudo / Claude
# =============================================================================

param(
    [string]$RunFolder = "",
    [string]$OutFile   = ""
)

$DefaultRoot = "C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\PYSR"
$Keep        = @('.txt', '.csv')

# ---- locate the run folder --------------------------------------------------
if ($RunFolder -eq "") {
    Write-Host "No folder given. Searching for the newest run_* under:"
    Write-Host "  $DefaultRoot"
    $cand = Get-ChildItem $DefaultRoot -Recurse -Directory -Filter "run_*" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $cand) { Write-Host "  [STOP] no run_* folder found."; return }
    $RunFolder = $cand.FullName
}

if (-not (Test-Path $RunFolder)) { Write-Host "  [STOP] not found: $RunFolder"; return }

# If the path given is a parent (no matching files directly under it), drop into
# its newest run_* child. Saves having to paste the timestamped name.
$here = Get-ChildItem $RunFolder -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $Keep -contains $_.Extension }
if (-not $here) {
    $child = Get-ChildItem $RunFolder -Directory -Filter "run_*" -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($child) { $RunFolder = $child.FullName }
}

$RunFolder = (Resolve-Path $RunFolder).Path.TrimEnd('\')
$name      = Split-Path $RunFolder -Leaf
$parent    = Split-Path $RunFolder -Parent
if ($OutFile -eq "") { $OutFile = Join-Path $parent "$name`_results.zip" }

Write-Host ""
Write-Host "  source : $RunFolder"
Write-Host "  archive: $OutFile"

# ---- stage the files, preserving the folder structure -----------------------
# Compress-Archive flattens a piped file list, and the seed number lives in the
# folder name, so the structure has to be rebuilt in a staging copy first.
$stage = Join-Path $env:TEMP ("ziprun_" + $name)
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue

$files = Get-ChildItem $RunFolder -Recurse -File |
         Where-Object { $Keep -contains $_.Extension }
if (-not $files) { Write-Host "  [STOP] no .txt or .csv files under that folder."; return }

foreach ($f in $files) {
    $rel  = $f.FullName.Substring($RunFolder.Length).TrimStart('\')
    $dest = Join-Path $stage $rel
    New-Item (Split-Path $dest -Parent) -ItemType Directory -Force | Out-Null
    Copy-Item $f.FullName $dest
}

Remove-Item $OutFile -Force -ErrorAction SilentlyContinue
Compress-Archive -Path $stage -DestinationPath $OutFile -CompressionLevel Optimal -Force
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue

# ---- report -----------------------------------------------------------------
$srcMB = [math]::Round((($files | Measure-Object Length -Sum).Sum) / 1MB, 2)
$zipMB = [math]::Round((Get-Item $OutFile).Length / 1MB, 2)
$allMB = [math]::Round(((Get-ChildItem $RunFolder -Recurse -File |
          Measure-Object Length -Sum).Sum) / 1MB, 1)

Write-Host ""
Write-Host "  kept   : $($files.Count) files, $srcMB MB of text and csv"
Write-Host "  skipped: everything else ($allMB MB total in the folder)"
Write-Host "  wrote  : $zipMB MB"
Write-Host ""
Write-Host "  $OutFile"

$seeds = ($files | Where-Object { $_.Name -like "pareto_equations_*" }).Count
$rec   = ($files | Where-Object { $_.Name -like "form_recurrence_*" }).Count
$cv    = ($files | Where-Object { $_.Name -eq "cv_summary.txt" }).Count
Write-Host ""
Write-Host "  contents check: $seeds seed front(s), $rec recurrence report, $cv cv_summary"
if ($seeds -lt 12) { Write-Host "  [WARN] fewer than 12 seed fronts. Did the run finish?" }
if ($cv -eq 0)     { Write-Host "  [WARN] no cv_summary.txt. Stage 8 has not been run yet." }
