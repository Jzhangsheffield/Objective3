param(
    [ValidateSet('audit','signals','upstream','train','a2','stress','bootstrap','summary')]
    [string]$Stage,
    [ValidateSet('A','D','J','M')]
    [string]$Participant = 'A',
    [ValidateSet(1,2,42)]
    [int]$Seed = 1,
    [ValidateSet('A1','A2','A3','A4','A5','A6','A7')]
    [string]$Condition = 'A7',
    [string]$Python = 'python',
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$PackageRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $PackageRoot

switch ($Stage) {
    'audit' { & $Python tools/audit_dataset.py --load-tensors }
    'signals' { & $Python tools/build_signal_cache.py --participant $Participant }
    'upstream' { & $Python tools/prepare_secondary_camera.py --participant $Participant --seed $Seed --device $Device --execute }
    'train' { & $Python tools/train_condition.py --condition $Condition --participant $Participant --seed $Seed --device $Device }
    'a2' { & $Python tools/evaluate_a2_late_fusion.py --participant $Participant --seed $Seed }
    'stress' { & $Python tools/run_stress_tests.py --condition $Condition --participant $Participant --seed $Seed --device $Device }
    'bootstrap' { & $Python tools/paired_bootstrap.py --condition $Condition }
    'summary' { & $Python tools/summarize_phase_a.py }
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
