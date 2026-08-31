param(
    [string]$Python = "python",
    [string]$Config = "",
    [string]$Device = "cuda",
    [int]$NumWorkers = 8,
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $PSScriptRoot
if (-not $Config) {
    $Config = Join-Path $PackageRoot "config\phase_b.json"
}

& $Python (Join-Path $PackageRoot "tools\audit_prerequisites.py") --config $Config
& $Python (Join-Path $PackageRoot "tools\generate_job_matrix.py") --config $Config --python $Python --device $Device --num-workers $NumWorkers

if ($Execute) {
    & $Python (Join-Path $PackageRoot "tools\run_job_matrix.py") --matrix (Join-Path $PackageRoot "scripts\phase_b_job_matrix.csv")
} else {
    Write-Host "Plan generated only. Add -Execute to start the resumable matrix."
}
