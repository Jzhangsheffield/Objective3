param(
    [string]$Python = 'python',
    [string]$Device = 'cuda',
    [int]$NumWorkers = 0,
    [ValidateSet('A','D','J','M')]
    [string[]]$Participants = @('A','D','J','M'),
    [ValidateSet(1,2,42)]
    [int[]]$Seeds = @(1,2,42),
    [ValidateSet('A1','A2','A3','A4','A5','A6','A7')]
    [string[]]$Conditions = @('A1','A2','A3','A4','A5','A6','A7'),
    [bool]$Resume = $true,
    [switch]$SkipTensorAudit,
    [switch]$SkipStress,
    [switch]$SkipLatency,
    [switch]$SkipBootstrap,
    [switch]$PlanOnly,
    [switch]$ContinueOnError
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$PackageRoot = Split-Path -Parent $PSScriptRoot
$ConfigPath = Join-Path $PackageRoot 'config\phase_a.json'
$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$OutputRoot = if ([System.IO.Path]::IsPathRooted([string]$Config.output_root)) {
    [string]$Config.output_root
} else {
    Join-Path $PackageRoot ([string]$Config.output_root)
}
$M2Root = [string]$Config.m2_project_root
$PrimaryCamera = [string]$Config.primary_camera_id
$SecondaryCamera = [string]$Config.secondary_camera_id
$RunStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogRoot = Join-Path $PackageRoot "logs\run_$RunStamp"
$StatusPath = Join-Path $LogRoot 'run_status.csv'
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Set-Location -LiteralPath $PackageRoot

$script:StatusRows = @()
$script:FailureCount = 0

function Save-Status {
    $script:StatusRows | Export-Csv -LiteralPath $StatusPath -NoTypeInformation -Encoding utf8
}

function Add-Status {
    param([string]$Task, [string]$Stage, [string]$Status, [double]$Seconds, [string]$Log, [string]$Message)
    $script:StatusRows += [pscustomobject]@{
        timestamp = (Get-Date -Format 's')
        task = $Task
        stage = $Stage
        status = $Status
        seconds = [math]::Round($Seconds, 2)
        log = $Log
        message = $Message
    }
    Save-Status
}

function Test-AllPaths {
    param([string[]]$Paths)
    if ($Paths.Count -eq 0) { return $false }
    foreach ($Path in $Paths) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    }
    return $true
}

function Invoke-PhasePython {
    param(
        [string]$Task,
        [string]$Stage,
        [string[]]$Arguments,
        [string[]]$CompletionPaths = @()
    )
    $SafeName = $Task -replace '[^A-Za-z0-9_.-]', '_'
    $LogPath = Join-Path $LogRoot "$SafeName.log"
    if ($Resume -and (Test-AllPaths -Paths $CompletionPaths)) {
        Write-Host "[SKIP] $Task" -ForegroundColor DarkGray
        Add-Status -Task $Task -Stage $Stage -Status 'SKIPPED_COMPLETE' -Seconds 0 -Log $LogPath -Message 'Completion files already exist.'
        return
    }

    if ($PlanOnly) {
        Write-Host "[PLAN] $Task :: $Python $($Arguments -join ' ')" -ForegroundColor Yellow
        Add-Status -Task $Task -Stage $Stage -Status 'PLANNED' -Seconds 0 -Log $LogPath -Message 'PlanOnly; command was not executed.'
        return
    }

    Write-Host "[RUN ] $Task" -ForegroundColor Cyan
    Write-Host ("      {0} {1}" -f $Python, ($Arguments -join ' '))
    $Started = Get-Date
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    $ExitCode = $LASTEXITCODE
    $Seconds = ((Get-Date) - $Started).TotalSeconds

    if ($ExitCode -ne 0) {
        $script:FailureCount += 1
        Add-Status -Task $Task -Stage $Stage -Status 'FAILED' -Seconds $Seconds -Log $LogPath -Message "Exit code $ExitCode"
        if (-not $ContinueOnError) {
            throw "Task failed: $Task. See $LogPath"
        }
        return
    }
    if ($CompletionPaths.Count -gt 0 -and -not (Test-AllPaths -Paths $CompletionPaths)) {
        $script:FailureCount += 1
        Add-Status -Task $Task -Stage $Stage -Status 'FAILED_MISSING_OUTPUT' -Seconds $Seconds -Log $LogPath -Message 'Command exited successfully but completion files are missing.'
        if (-not $ContinueOnError) {
            throw "Task did not create expected outputs: $Task"
        }
        return
    }
    Add-Status -Task $Task -Stage $Stage -Status 'COMPLETED' -Seconds $Seconds -Log $LogPath -Message ''
}

function Get-ProtocolRoot {
    param([string]$Participant)
    return Join-Path $M2Root "outputs\${Participant}_as_test\cam_$PrimaryCamera\protocols\all_runs"
}

function Get-ModelRoot {
    param([string]$Condition, [string]$Participant, [int]$Seed)
    return Join-Path $OutputRoot "$Condition\${Participant}_as_test\seed_$Seed"
}

Write-Host "Phase A automatic runner" -ForegroundColor Green
Write-Host "Package:       $PackageRoot"
Write-Host "Output:        $OutputRoot"
Write-Host "Participants:  $($Participants -join ',')"
Write-Host "Seeds:         $($Seeds -join ',')"
Write-Host "Conditions:    $($Conditions -join ',')"
Write-Host "Resume:        $Resume"
Write-Host "Plan only:     $PlanOnly"
Write-Host "Logs:          $LogRoot"

# Stage 0: runtime and immutable data/protocol audit.
Invoke-PhasePython -Task '00_runtime_check' -Stage 'runtime' -Arguments @(
    '-c', 'import torch,numpy; print("torch",torch.__version__); print("cuda",torch.cuda.is_available()); print("numpy",numpy.__version__)'
)
Invoke-PhasePython -Task '00_model_smoke' -Stage 'runtime' -Arguments @(
    'tools/smoke_test.py'
)

$AuditPath = Join-Path $PackageRoot 'audit\dataset_audit.json'
$TensorAuditComplete = $false
if ($Resume -and (Test-Path -LiteralPath $AuditPath -PathType Leaf)) {
    try {
        $AuditValue = Get-Content -LiteralPath $AuditPath -Raw | ConvertFrom-Json
        $TensorAuditComplete = ([string]$AuditValue.status -eq 'PASS' -and [string]$AuditValue.tensor_audit -eq 'PASS')
    } catch {
        $TensorAuditComplete = $false
    }
}
if ($SkipTensorAudit) {
    Write-Warning 'Full MindRove tensor audit was skipped by request.'
    Add-Status -Task '01_tensor_audit' -Stage 'audit' -Status 'SKIPPED_BY_USER' -Seconds 0 -Log '' -Message 'SkipTensorAudit was supplied.'
} elseif ($TensorAuditComplete) {
    Write-Host '[SKIP] 01_tensor_audit' -ForegroundColor DarkGray
    Add-Status -Task '01_tensor_audit' -Stage 'audit' -Status 'SKIPPED_COMPLETE' -Seconds 0 -Log '' -Message 'Existing tensor audit is PASS.'
} else {
    Invoke-PhasePython -Task '01_tensor_audit' -Stage 'audit' -Arguments @(
        'tools/audit_dataset.py', '--load-tensors'
    ) -CompletionPaths @($AuditPath)
}

# Stage 1: one leakage-free signal cache per LOSO fold.
foreach ($Participant in $Participants) {
    $SignalRoot = Join-Path $OutputRoot "signal_cache\${Participant}_as_test"
    Invoke-PhasePython -Task "10_signal_${Participant}" -Stage 'signal_cache' -Arguments @(
        'tools/build_signal_cache.py', '--participant', $Participant
    ) -CompletionPaths @(
        (Join-Path $SignalRoot 'train_right_signals.pt'),
        (Join-Path $SignalRoot 'test_right_signals.pt'),
        (Join-Path $SignalRoot 'right_signal_stats.json')
    )
}

$NeedsSecondary = @($Conditions | Where-Object { $_ -in @('A1','A2','A3','A7') }).Count -gt 0
if ($NeedsSecondary) {
    # Stage 2: independent second-camera backbone and feature caches for 12 fold x seed tasks.
    foreach ($Participant in $Participants) {
        foreach ($Seed in $Seeds) {
            $UpstreamRoot = Join-Path $OutputRoot "upstream\${Participant}_as_test\cam_$SecondaryCamera\seed_$Seed"
            Invoke-PhasePython -Task "20_upstream_${Participant}_s${Seed}" -Stage 'secondary_upstream' -Arguments @(
                'tools/prepare_secondary_camera.py', '--participant', $Participant,
                '--seed', [string]$Seed, '--device', $Device,
                '--num-workers', [string]$NumWorkers, '--execute'
            ) -CompletionPaths @(
                (Join-Path $UpstreamRoot 'backbone\last.pth'),
                (Join-Path $UpstreamRoot 'features\train_all.pt'),
                (Join-Path $UpstreamRoot 'features\test_all.pt')
            )
        }
    }
}

# Stage 3: A1 and then A2, because A2 consumes paired A0/A1 probabilities.
if ('A1' -in $Conditions -or 'A2' -in $Conditions) {
    foreach ($Participant in $Participants) {
        foreach ($Seed in $Seeds) {
            $A1Root = Get-ModelRoot -Condition 'A1' -Participant $Participant -Seed $Seed
            Invoke-PhasePython -Task "30_A1_${Participant}_s${Seed}" -Stage 'train_A1' -Arguments @(
                'tools/train_condition.py', '--condition', 'A1', '--participant', $Participant,
                '--seed', [string]$Seed, '--device', $Device, '--num-workers', [string]$NumWorkers
            ) -CompletionPaths @((Join-Path $A1Root 'completed.json'))

            if ('A2' -in $Conditions) {
                $A2Root = Get-ModelRoot -Condition 'A2' -Participant $Participant -Seed $Seed
                Invoke-PhasePython -Task "31_A2_${Participant}_s${Seed}" -Stage 'late_fusion_A2' -Arguments @(
                    'tools/evaluate_a2_late_fusion.py', '--participant', $Participant, '--seed', [string]$Seed
                ) -CompletionPaths @((Join-Path $A2Root 'completed.json'))
            }
        }
    }
}

# Stage 4: learned incremental adapters A3-A7.
$TrainableAdapters = @('A3','A4','A5','A6','A7') | Where-Object { $_ -in $Conditions }
foreach ($Condition in $TrainableAdapters) {
    foreach ($Participant in $Participants) {
        foreach ($Seed in $Seeds) {
            $ModelRoot = Get-ModelRoot -Condition $Condition -Participant $Participant -Seed $Seed
            Invoke-PhasePython -Task "40_${Condition}_${Participant}_s${Seed}" -Stage "train_$Condition" -Arguments @(
                'tools/train_condition.py', '--condition', $Condition, '--participant', $Participant,
                '--seed', [string]$Seed, '--device', $Device, '--num-workers', [string]$NumWorkers
            ) -CompletionPaths @((Join-Path $ModelRoot 'completed.json'))
        }
    }
}

# Stage 5: missing-modality and independent EMG/IMU offset stress tests.
if (-not $SkipStress) {
    foreach ($Condition in $TrainableAdapters) {
        foreach ($Participant in $Participants) {
            foreach ($Seed in $Seeds) {
                $ModelRoot = Get-ModelRoot -Condition $Condition -Participant $Participant -Seed $Seed
                Invoke-PhasePython -Task "50_stress_${Condition}_${Participant}_s${Seed}" -Stage 'stress' -Arguments @(
                    'tools/run_stress_tests.py', '--condition', $Condition, '--participant', $Participant,
                    '--seed', [string]$Seed, '--device', $Device, '--num-workers', [string]$NumWorkers
                ) -CompletionPaths @((Join-Path $ModelRoot 'stress_completed.json'))
            }
        }
    }
} else {
    Write-Warning 'Stress tests were skipped by request.'
}

# Stage 6: cached-feature latency on one representative fold/seed per trainable model.
if (-not $SkipLatency -and 'A' -in $Participants -and 1 -in $Seeds) {
    $LatencyConditions = @('A1','A3','A4','A5','A6','A7') | Where-Object { $_ -in $Conditions }
    foreach ($Condition in $LatencyConditions) {
        $ModelRoot = Get-ModelRoot -Condition $Condition -Participant 'A' -Seed 1
        Invoke-PhasePython -Task "60_latency_${Condition}_A_s1" -Stage 'latency' -Arguments @(
            'tools/benchmark_latency.py', '--condition', $Condition,
            '--participant', 'A', '--seed', '1', '--device', $Device
        ) -CompletionPaths @((Join-Path $ModelRoot 'latency_cached_feature_scope.json'))
    }
} elseif ($SkipLatency) {
    Write-Warning 'Latency tests were skipped by request.'
}

# Stage 7: paired clip bootstrap requires the complete 4-fold x 3-seed grid.
$HasAllParticipants = @(@('A','D','J','M') | Where-Object { $_ -notin $Participants }).Count -eq 0
$HasAllSeeds = @(@(1,2,42) | Where-Object { $_ -notin $Seeds }).Count -eq 0
if (-not $SkipBootstrap -and $HasAllParticipants -and $HasAllSeeds) {
    foreach ($Condition in $Conditions) {
        $BootstrapPath = Join-Path $OutputRoot "summary\paired_bootstrap_${Condition}_vs_A0.json"
        Invoke-PhasePython -Task "70_bootstrap_$Condition" -Stage 'bootstrap' -Arguments @(
            'tools/paired_bootstrap.py', '--condition', $Condition
        ) -CompletionPaths @($BootstrapPath)
    }
} elseif ($SkipBootstrap) {
    Write-Warning 'Paired bootstrap was skipped by request.'
} else {
    Write-Warning 'Paired bootstrap was automatically skipped because this is not the complete A/D/J/M x 1/2/42 grid.'
}

# Stage 8: always refresh the human- and machine-readable summary.
Invoke-PhasePython -Task '80_summary' -Stage 'summary' -Arguments @(
    'tools/summarize_phase_a.py'
) -CompletionPaths @(
    (Join-Path $OutputRoot 'summary\PHASE_A_RESULTS.md'),
    (Join-Path $OutputRoot 'summary\condition_summary.csv'),
    (Join-Path $OutputRoot 'summary\incremental_value_gates.json')
)

Write-Host "Run status: $StatusPath" -ForegroundColor Green
if ($PlanOnly) {
    Write-Host 'Plan generated; no Python command was executed.' -ForegroundColor Yellow
    exit 0
}
if ($script:FailureCount -gt 0) {
    Write-Error "Phase A finished with $script:FailureCount failed task(s)."
    exit 1
}
Write-Host 'Phase A completed successfully.' -ForegroundColor Green
