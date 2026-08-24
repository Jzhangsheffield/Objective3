param(
    [ValidateSet("Prepare", "Validate", "UpstreamDryRun", "UpstreamRun", "Smoke", "HistoryDryRun", "HistoryRun", "DryRun", "Run", "Full", "Coverage", "Summarize")]
    [string]$Action = "HistoryDryRun",
    [string]$Python = "python",
    [string]$Experiments = "",
    [string]$Participants = "",
    [string]$Seeds = "",
    [string]$DatasetRoot = "",
    [string]$Device = "",
    [switch]$Overwrite,
    [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"
$Config = Join-Path $PSScriptRoot "config\experiment_config.json"

if ($Action -eq "Prepare") {
    $Arguments = @((Join-Path $PSScriptRoot "tools\prepare_sequence_disjoint_protocols.py"), "--config", $Config)
    if ($Participants) { $Arguments += @("--participants", $Participants) }
    if ($Overwrite) { $Arguments += "--overwrite" }
    & $Python @Arguments
    exit $LASTEXITCODE
}

if ($Action -eq "Validate") {
    & $Python (Join-Path $PSScriptRoot "tools\validate_setup.py") --config $Config
    exit $LASTEXITCODE
}

if ($Action -eq "Smoke") {
    & $Python (Join-Path $PSScriptRoot "tools\smoke_test.py") --config $Config
    exit $LASTEXITCODE
}

if ($Action -eq "Coverage") {
    $Arguments = @((Join-Path $PSScriptRoot "tools\audit_augmentation_coverage.py"), "--config", $Config)
    if ($Participants) { $Arguments += @("--participants", $Participants) }
    if ($Experiments) { $Arguments += @("--experiments", $Experiments) }
    & $Python @Arguments
    exit $LASTEXITCODE
}

if ($Action -in @("UpstreamDryRun", "UpstreamRun", "Full")) {
    $UpstreamArguments = @((Join-Path $PSScriptRoot "tools\run_upstream_pipeline.py"), "--config", $Config)
    if ($Participants) { $UpstreamArguments += @("--participants", $Participants) }
    if ($Seeds) { $UpstreamArguments += @("--seeds", $Seeds) }
    if ($DatasetRoot) { $UpstreamArguments += @("--dataset-root", $DatasetRoot) }
    if ($Device) { $UpstreamArguments += @("--device", $Device) }
    if ($Action -eq "UpstreamDryRun") { $UpstreamArguments += "--dry-run" }
    if ($Overwrite) { $UpstreamArguments += "--overwrite" }
    if ($ContinueOnError) { $UpstreamArguments += "--continue-on-error" }
    & $Python @UpstreamArguments
    if ($LASTEXITCODE -ne 0 -or $Action -ne "Full") { exit $LASTEXITCODE }
}

if ($Action -eq "Summarize") {
    & $Python (Join-Path $PSScriptRoot "tools\summarize_results.py")
    exit $LASTEXITCODE
}

$GridArguments = @((Join-Path $PSScriptRoot "tools\run_grid.py"), "--config", $Config)
if ($Experiments) { $GridArguments += @("--experiments", $Experiments) }
if ($Participants) { $GridArguments += @("--participants", $Participants) }
if ($Seeds) { $GridArguments += @("--seeds", $Seeds) }
if ($Action -in @("DryRun", "HistoryDryRun")) { $GridArguments += "--dry-run" }
if ($Overwrite) { $GridArguments += "--overwrite" }
if ($ContinueOnError) { $GridArguments += "--continue-on-error" }
& $Python @GridArguments
exit $LASTEXITCODE
