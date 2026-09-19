# Builds a clean zip of the project to send to the client: dist\mariam-app.zip
# Leaves out the huge .venv folder, git history and caches. Includes the
# trained models, results and data, so the client's first start does not have
# to train anything. Run from the project folder:
#     powershell -ExecutionPolicy Bypass -File make_client_zip.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$dist = Join-Path $root "dist"
$staging = Join-Path $env:TEMP "mariam-app-staging"
$zip = Join-Path $dist "mariam-app.zip"

foreach ($required in @("models\fusion_model.joblib", "models\text_model", "results\scored_customers.csv", "data\processed\customers.jsonl")) {
    if (-not (Test-Path (Join-Path $root $required))) {
        throw "Missing $required - run the pipeline first (python src/ingest.py; python src/preprocess.py; python src/evaluate.py) so the client does not have to train."
    }
}

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Force $dist | Out-Null
if (Test-Path $zip) { Remove-Item $zip -Force }

$target = Join-Path $staging "mariam-app"
robocopy $root $target /E /NFL /NDL /NJH /NJS /NP `
    /XD ".venv" "venvs" ".git" ".claude" "dist" "__pycache__" ".pytest_cache" ".streamlit" ".vscode" ".idea" | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with code $LASTEXITCODE" }

# Build the zip entry by entry so paths use "/" (works on Windows, Mac and
# Linux); CreateFromDirectory can write "\" separators that break on Mac.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::Open($zip, [System.IO.Compression.ZipArchiveMode]::Create)
Push-Location $staging
try {
    # Resolve-Path -Relative sidesteps short (8.3) vs long path mismatches in
    # %TEMP%, which break plain string trimming.
    foreach ($file in Get-ChildItem . -Recurse -File) {
        $entryName = (Resolve-Path -LiteralPath $file.FullName -Relative).Substring(2).Replace('\', '/')
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive, $file.FullName, $entryName, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
} finally {
    Pop-Location
    $archive.Dispose()
}
Remove-Item $staging -Recurse -Force

$mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "Created $zip ($mb MB)"
