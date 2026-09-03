param(
    [string]$Python = "python",
    [string]$Device = "cuda",
    [int]$NumWorkers = 0,
    [int]$StartJob = 1
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $PSScriptRoot
$Config = Join-Path $PackageRoot "config\experiment.json"
$Matrix = Join-Path $PackageRoot "scripts\job_matrix.csv"

& $Python (Join-Path $PackageRoot "tools\generate_job_matrix.py") `
    --config $Config --device $Device --num-workers $NumWorkers --output $Matrix
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $PackageRoot "tools\run_job_matrix.py") `
    --matrix $Matrix --start-job $StartJob
exit $LASTEXITCODE
