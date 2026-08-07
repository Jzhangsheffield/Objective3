param(
    [string]$Config = "$PSScriptRoot\..\configs\base.json",
    [ValidateSet("prepare", "extract", "train", "evaluate", "end_to_end")][string]$Stage = "prepare",
    [ValidateSet("normal_only", "all_runs", "both")][string]$Scope = "both",
    [string]$PythonExe = "C:\Users\digit\anaconda3\envs\Pytorch\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..").Path
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python environment not found: $PythonExe"
}

if ($Stage -eq "prepare") {
    & $PythonExe "$Root\tools\prepare_protocols.py" --config $Config
    exit $LASTEXITCODE
}

$Scopes = if ($Scope -eq "both") { @("normal_only", "all_runs") } else { @($Scope) }
foreach ($Heldout in @("A", "D", "J", "M")) {
    foreach ($Seed in @(1, 2, 42)) {
        foreach ($TrainScope in $Scopes) {
            if ($Stage -eq "extract") {
                & $PythonExe "$Root\tools\extract_boundary_features.py" --config $Config --heldout $Heldout --seed $Seed --scope $TrainScope --splits train test_all
            } elseif ($Stage -eq "train") {
                & $PythonExe "$Root\tools\train_boundary.py" --config $Config --heldout $Heldout --seed $Seed --scope $TrainScope
            } elseif ($Stage -eq "evaluate") {
                & $PythonExe "$Root\tools\evaluate_boundary.py" --config $Config --heldout $Heldout --seed $Seed --scope $TrainScope
            } elseif ($Stage -eq "end_to_end") {
                & $PythonExe "$Root\tools\evaluate_end_to_end.py" --config $Config --heldout $Heldout --seed $Seed --scope $TrainScope
            }
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
}
