param(
    [string]$Python = 'python',
    [string]$Device = 'cuda',
    [int]$NumWorkers = 0,
    [ValidateSet('A','D','J','M')]
    [string[]]$Participants = @('A','D','J','M'),
    [ValidateSet(1,2,42)]
    [int[]]$Seeds = @(1,2,42),
    [ValidateSet('S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12')]
    [string[]]$Experiments = @('S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12'),
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
$ConfigPath = Join-Path $PackageRoot 'config\bilateral_supplementary_experiments.json'
$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$BaseConfigPath = Join-Path $PackageRoot 'config\phase_a.json'
$BaseConfig = Get-Content -LiteralPath $BaseConfigPath -Raw | ConvertFrom-Json
$BaseOutputRoot = if ([System.IO.Path]::IsPathRooted([string]$BaseConfig.output_root)) {
    [string]$BaseConfig.output_root
} else { Join-Path $PackageRoot ([string]$BaseConfig.output_root) }
$OutputRoot = Join-Path $BaseOutputRoot ([string]$Config.output_subdirectory)
$CacheRoot = Join-Path $BaseOutputRoot ([string]$Config.signal_data.cache_subdirectory)
$M2Root = [string]$BaseConfig.m2_project_root
$PrimaryCamera = [string]$BaseConfig.primary_camera_id
$Protocols = @($Config.signal_data.test_protocols.PSObject.Properties.Name)
$RunStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogRoot = Join-Path $PackageRoot "logs\bilateral_run_$RunStamp"
$StatusPath = Join-Path $LogRoot 'run_status.csv'
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Set-Location -LiteralPath $PackageRoot

$script:StatusRows = @()
$script:FailureCount = 0

function Save-Status { $script:StatusRows | Export-Csv -LiteralPath $StatusPath -NoTypeInformation -Encoding utf8 }
function Add-Status {
    param([string]$Task,[string]$Stage,[string]$Status,[double]$Seconds,[string]$Log,[string]$Message)
    $script:StatusRows += [pscustomobject]@{
        timestamp=(Get-Date -Format 's'); task=$Task; stage=$Stage; status=$Status
        seconds=[math]::Round($Seconds,2); log=$Log; message=$Message
    }
    Save-Status
}
function Test-AllPaths {
    param([string[]]$Paths)
    if ($Paths.Count -eq 0) { return $false }
    foreach ($Path in $Paths) { if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false } }
    return $true
}
function Invoke-ExperimentPython {
    param([string]$Task,[string]$Stage,[string[]]$Arguments,[string[]]$CompletionPaths=@(),[switch]$AlwaysRun)
    $SafeName = $Task -replace '[^A-Za-z0-9_.-]', '_'
    $LogPath = Join-Path $LogRoot "$SafeName.log"
    $StdoutPath = Join-Path $LogRoot "$SafeName.stdout.txt"
    $StderrPath = Join-Path $LogRoot "$SafeName.stderr.txt"
    if (-not $AlwaysRun -and $Resume -and (Test-AllPaths $CompletionPaths)) {
        Write-Host "[SKIP] $Task" -ForegroundColor DarkGray
        Add-Status $Task $Stage 'SKIPPED_COMPLETE' 0 $LogPath 'Completion files already exist.'
        return
    }
    if ($PlanOnly) {
        Write-Host "[PLAN] $Task :: $Python $($Arguments -join ' ')" -ForegroundColor Yellow
        Add-Status $Task $Stage 'PLANNED' 0 $LogPath 'PlanOnly; command was not executed.'
        return
    }
    Write-Host "[RUN ] $Task" -ForegroundColor Cyan
    Write-Host ("      {0} {1}" -f $Python,($Arguments -join ' '))
    $Started = Get-Date
    try {
        $Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $PackageRoot `
            -NoNewWindow -Wait -PassThru -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
        $ExitCode = $Process.ExitCode
    } catch {
        $Seconds = ((Get-Date)-$Started).TotalSeconds
        $Message = "Unable to start Python process: $($_.Exception.Message)"
        @("TASK: $Task","COMMAND: $Python $($Arguments -join ' ')","START_PROCESS_ERROR:",$Message) |
            Set-Content -LiteralPath $LogPath -Encoding utf8
        $script:FailureCount += 1
        Add-Status $Task $Stage 'FAILED_TO_START' $Seconds $LogPath $Message
        if (-not $ContinueOnError) { throw "Task failed to start: $Task. See $LogPath" }
        return
    }
    $Seconds = ((Get-Date)-$Started).TotalSeconds
    @("TASK: $Task","COMMAND: $Python $($Arguments -join ' ')","EXIT_CODE: $ExitCode","","===== STDOUT =====") |
        Set-Content -LiteralPath $LogPath -Encoding utf8
    if (Test-Path -LiteralPath $StdoutPath) {
        Get-Content -LiteralPath $StdoutPath | Add-Content -LiteralPath $LogPath -Encoding utf8
        Get-Content -LiteralPath $StdoutPath | ForEach-Object { Write-Host $_ }
    }
    "`n===== STDERR =====" | Add-Content -LiteralPath $LogPath -Encoding utf8
    if (Test-Path -LiteralPath $StderrPath) {
        Get-Content -LiteralPath $StderrPath | Add-Content -LiteralPath $LogPath -Encoding utf8
        Get-Content -LiteralPath $StderrPath | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    }
    if ($ExitCode -ne 0) {
        $script:FailureCount += 1
        Add-Status $Task $Stage 'FAILED' $Seconds $LogPath "Exit code $ExitCode"
        if (-not $ContinueOnError) { throw "Task failed: $Task. See $LogPath" }
        return
    }
    if ($CompletionPaths.Count -gt 0 -and -not (Test-AllPaths $CompletionPaths)) {
        $script:FailureCount += 1
        Add-Status $Task $Stage 'FAILED_MISSING_OUTPUT' $Seconds $LogPath 'Expected output is missing.'
        if (-not $ContinueOnError) { throw "Task did not create expected outputs: $Task" }
        return
    }
    Add-Status $Task $Stage 'COMPLETED' $Seconds $LogPath ''
}
function Get-ModelRoot {
    param([string]$Condition,[string]$Participant,[int]$Seed)
    return Join-Path $OutputRoot "$Condition\${Participant}_as_test\seed_$Seed"
}

Write-Host 'Bilateral S1-S12 automatic runner' -ForegroundColor Green
Write-Host "Package:      $PackageRoot"
Write-Host "Output:       $OutputRoot"
Write-Host "Cache:        $CacheRoot"
Write-Host "Participants: $($Participants -join ',')"
Write-Host "Seeds:        $($Seeds -join ',')"
Write-Host "Experiments:  $($Experiments -join ',')"
Write-Host "Protocols:    $($Protocols -join ',')"
Write-Host "Resume:       $Resume"
Write-Host "Plan only:    $PlanOnly"
Write-Host "Logs:         $LogRoot"

$DatasetRoot = [string]$BaseConfig.dataset_root
if (-not (Test-Path -LiteralPath $DatasetRoot -PathType Container)) {
    throw "dataset_root does not exist on this computer: $DatasetRoot. Edit config\phase_a.json before running."
}
if (-not (Test-Path -LiteralPath $M2Root -PathType Container)) {
    throw "m2_project_root does not exist on this computer: $M2Root. Edit config\phase_a.json before running."
}

Invoke-ExperimentPython '00_runtime_check' 'runtime' @('tools/check_runtime.py','--device',$Device)
Invoke-ExperimentPython '00_bilateral_model_smoke' 'runtime' @(
    'tools/smoke_test_supplementary.py',$ConfigPath
)

$AuditPath = Join-Path $PackageRoot 'audit\dataset_audit.json'
if ($SkipTensorAudit) {
    Add-Status '01_tensor_audit' 'audit' 'SKIPPED_BY_USER' 0 '' 'SkipTensorAudit was supplied.'
} else {
    Invoke-ExperimentPython '01_tensor_audit' 'audit' @('tools/audit_dataset.py','--load-tensors') @($AuditPath)
}

foreach ($Participant in $Participants) {
    $FoldCache = Join-Path $CacheRoot "${Participant}_as_test"
    $CacheCompletion = @(
        (Join-Path $FoldCache 'train_bilateral_signals.pt'),
        (Join-Path $FoldCache 'test_pooled_train_bilateral_signals.pt'),
        (Join-Path $FoldCache 'test_participant_calibrated_bilateral_signals.pt'),
        (Join-Path $FoldCache 'bilateral_signal_stats.json')
    )
    foreach ($Protocol in $Protocols) {
        $CacheCompletion += Join-Path $FoldCache "evaluation_protocols\$Protocol\test_all.jsonl"
        $CacheCompletion += Join-Path $FoldCache "evaluation_protocols\$Protocol\test_normal.jsonl"
        $CacheCompletion += Join-Path $FoldCache "evaluation_protocols\$Protocol\test_fault.jsonl"
    }
    Invoke-ExperimentPython "10_bilateral_signal_$Participant" 'signal_cache' @(
        'tools/build_bilateral_signal_cache.py','--config',$ConfigPath,'--participant',$Participant
    ) $CacheCompletion
}

$Dependencies = @{'S1'='S9';'S2'='S10';'S3'='S11';'S4'='S12'}
$Selected = @($Experiments | Select-Object -Unique)
$SelectedM2 = @('S1','S2','S3','S4') | Where-Object { $_ -in $Selected }
$SelectedNode = @('S5','S6','S7','S8') | Where-Object { $_ -in $Selected }
$RequiredTier3 = @(@('S9','S10','S11','S12') | Where-Object { $_ -in $Selected })
foreach ($Condition in $SelectedM2) { $RequiredTier3 += [string]$Dependencies[$Condition] }
$RequiredTier3 = @($RequiredTier3 | Select-Object -Unique)
$Available = @($Selected + $RequiredTier3 | Select-Object -Unique)

foreach ($Condition in $RequiredTier3) {
    foreach ($Participant in $Participants) { foreach ($Seed in $Seeds) {
        $ModelRoot = Get-ModelRoot $Condition $Participant $Seed
        Invoke-ExperimentPython "20_${Condition}_${Participant}_s$Seed" "train_$Condition" @(
            'tools/train_signal_direct.py','--config',$ConfigPath,'--condition',$Condition,
            '--participant',$Participant,'--seed',[string]$Seed,'--device',$Device,
            '--num-workers',[string]$NumWorkers
        ) @((Join-Path $ModelRoot 'completed.json'))
    }}
}

foreach ($Condition in $SelectedM2) {
    $Upstream = [string]$Dependencies[$Condition]
    foreach ($Participant in $Participants) { foreach ($Seed in $Seeds) {
        $FeatureRoot = Join-Path $OutputRoot "signal_features\$Upstream\${Participant}_as_test\seed_$Seed"
        $FeatureCompletion = @((Join-Path $FeatureRoot 'train_features.pt'),(Join-Path $FeatureRoot 'completed.json'))
        foreach ($Protocol in $Protocols) {
            $FeatureCompletion += Join-Path $FeatureRoot "test_${Protocol}_features.pt"
        }
        Invoke-ExperimentPython "21_features_${Upstream}_${Participant}_s$Seed" 'signal_feature_cache' @(
            'tools/extract_signal_features.py','--config',$ConfigPath,'--condition',$Upstream,
            '--participant',$Participant,'--seed',[string]$Seed,'--device',$Device,
            '--num-workers',[string]$NumWorkers
        ) $FeatureCompletion
        $ModelRoot = Get-ModelRoot $Condition $Participant $Seed
        Invoke-ExperimentPython "22_${Condition}_${Participant}_s$Seed" "train_$Condition" @(
            'tools/train_sensor_m2.py','--config',$ConfigPath,'--condition',$Condition,
            '--participant',$Participant,'--seed',[string]$Seed,'--device',$Device,
            '--num-workers',[string]$NumWorkers
        ) @((Join-Path $ModelRoot 'completed.json'))
    }}
}

foreach ($Condition in $SelectedNode) {
    foreach ($Participant in $Participants) { foreach ($Seed in $Seeds) {
        $ModelRoot = Get-ModelRoot $Condition $Participant $Seed
        Invoke-ExperimentPython "23_${Condition}_${Participant}_s$Seed" "train_$Condition" @(
            'tools/train_signal_direct.py','--config',$ConfigPath,'--condition',$Condition,
            '--participant',$Participant,'--seed',[string]$Seed,'--device',$Device,
            '--num-workers',[string]$NumWorkers
        ) @((Join-Path $ModelRoot 'completed.json'))
    }}
}

if (-not $SkipStress) {
    foreach ($Condition in $Selected) {
        foreach ($Participant in $Participants) { foreach ($Seed in $Seeds) {
            $ModelRoot = Get-ModelRoot $Condition $Participant $Seed
            Invoke-ExperimentPython "30_stress_${Condition}_${Participant}_s$Seed" 'stress' @(
                'tools/run_supplementary_stress.py','--config',$ConfigPath,'--condition',$Condition,
                '--participant',$Participant,'--seed',[string]$Seed,'--device',$Device,
                '--num-workers',[string]$NumWorkers
            ) @((Join-Path $ModelRoot 'stress_completed.json'))
        }}
    }
}

if (-not $SkipLatency -and 'A' -in $Participants -and 1 -in $Seeds) {
    foreach ($Condition in $Selected) {
        $ModelRoot = Get-ModelRoot $Condition 'A' 1
        Invoke-ExperimentPython "40_latency_${Condition}_A_s1" 'latency' @(
            'tools/benchmark_supplementary_latency.py','--config',$ConfigPath,'--condition',$Condition,
            '--participant','A','--seed','1','--device',$Device
        ) @((Join-Path $ModelRoot 'latency_end_to_end_signal_scope.json'))
    }
}

$HasAllParticipants = @(@('A','D','J','M') | Where-Object { $_ -notin $Participants }).Count -eq 0
$HasAllSeeds = @(@(1,2,42) | Where-Object { $_ -notin $Seeds }).Count -eq 0
if (-not $SkipBootstrap -and $HasAllParticipants -and $HasAllSeeds) {
    foreach ($Protocol in $Protocols) {
        foreach ($Comparison in $Config.paired_comparisons) {
            $Candidate = [string]$Comparison.candidate
            $Baseline = [string]$Comparison.baseline
            if ($Candidate -in $Available -and $Baseline -in $Available) {
                $BootstrapPath = Join-Path $OutputRoot "summary\$Protocol\paired_bootstrap_${Candidate}_vs_${Baseline}.json"
                Invoke-ExperimentPython "50_bootstrap_${Protocol}_${Candidate}_vs_$Baseline" 'bootstrap' @(
                    'tools/paired_bootstrap_supplementary.py','--config',$ConfigPath,
                    '--evaluation-protocol',$Protocol,'--candidate',$Candidate,'--baseline',$Baseline
                ) @($BootstrapPath)
            }
        }
    }
} elseif (-not $SkipBootstrap) {
    Write-Warning 'Bootstrap skipped because this is not the complete A/D/J/M x 1/2/42 grid.'
}

foreach ($Protocol in $Protocols) {
    $SummaryRoot = Join-Path $OutputRoot "summary\$Protocol"
    $SummaryArguments = @(
        'tools/summarize_supplementary.py','--config',$ConfigPath,
        '--evaluation-protocol',$Protocol,'--conditions'
    ) + $Selected
    Invoke-ExperimentPython -Task "60_summary_$Protocol" -Stage 'summary' -Arguments $SummaryArguments `
        -CompletionPaths @((Join-Path $SummaryRoot 'SUPPLEMENTARY_RESULTS.md'),
                           (Join-Path $SummaryRoot 'condition_summary.csv')) -AlwaysRun
}

Write-Host "Run status: $StatusPath" -ForegroundColor Green
if ($PlanOnly) { Write-Host 'Plan generated; no Python command was executed.' -ForegroundColor Yellow; exit 0 }
if ($script:FailureCount -gt 0) { Write-Error "Bilateral S1-S12 finished with $script:FailureCount failed task(s)."; exit 1 }
Write-Host 'Bilateral S1-S12 completed successfully.' -ForegroundColor Green
