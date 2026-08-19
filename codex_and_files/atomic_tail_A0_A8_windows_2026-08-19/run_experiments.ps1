param(
    [string]$Experiments = "A0,A1,A2,A3,A4,A5,A6,A7,A8",
    [string]$Participants = "A,D,J,M",
    [string]$Seeds = "1,2,42",
    [string]$Scopes = "all_runs",
    [string]$Config = "$PSScriptRoot\config\experiment_config.json",
    [switch]$DryRun,
    [switch]$ValidateOnly,
    [switch]$Overwrite,
    [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"
$resolvedConfig = (Resolve-Path -LiteralPath $Config).Path
$settings = Get-Content -LiteralPath $resolvedConfig -Raw | ConvertFrom-Json
$python = $settings.paths.python_executable
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable not found: $python`nEdit paths.python_executable in $resolvedConfig"
}
$arguments = @(
    "$PSScriptRoot\tools\run_grid.py",
    "--config", $resolvedConfig,
    "--experiments", $Experiments,
    "--participants", $Participants,
    "--seeds", $Seeds,
    "--scopes", $Scopes
)
if ($DryRun) { $arguments += "--dry-run" }
if ($ValidateOnly) { $arguments += "--validate-only" }
if ($Overwrite) { $arguments += "--overwrite" }
if ($ContinueOnError) { $arguments += "--continue-on-error" }
& $python @arguments
exit $LASTEXITCODE

