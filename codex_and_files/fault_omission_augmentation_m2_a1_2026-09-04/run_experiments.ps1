param(
    [ValidateSet('prepare','check','audit','train','smoke','summarize')]
    [string]$Action = 'check',
    [string]$Python = 'C:\Users\digit\anaconda3\envs\Pytorch\python.exe',
    [string]$Config = '',
    [string[]]$Folds = @(),
    [int[]]$Seeds = @(),
    [string[]]$Groups = @(),
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
if (-not $Config) { $Config = Join-Path $PSScriptRoot 'config\experiment_config.json' }
$runArgs = @('-B', (Join-Path $PSScriptRoot 'tools\run_experiments.py'), $Action, '--config', $Config)
if ($Folds.Count) { $runArgs += '--folds'; $runArgs += $Folds }
if ($Seeds.Count) { $runArgs += '--seeds'; $runArgs += $Seeds | ForEach-Object { [string]$_ } }
if ($Groups.Count) { $runArgs += '--groups'; $runArgs += $Groups }
if ($Resume) { $runArgs += '--resume' }
& $Python @runArgs
if ($LASTEXITCODE -ne 0) { throw "Experiment command failed (exit $LASTEXITCODE)" }
