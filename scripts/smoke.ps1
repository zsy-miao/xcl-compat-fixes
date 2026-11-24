# Regression run of a title for a pull request: runs it for a fixed time, reads XCL's log and
# prints an `xcl-smoke` block to paste into the PR's "Regression run" section – once built
# from main, once from the PR's head commit (CONTRIBUTING.md, "What a PR needs to be merged").
#
#   .\scripts\smoke.ps1 -Game 'D:\Games\SomeTitle' [-Seconds 60] [-Xcl .\target\release\xcl.exe]
#
# Networking is off (XCL_NETWORK=0). Build first: cargo build --release -p xcl
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [int]$Seconds = 60,
    [string]$Xcl = (Join-Path $PSScriptRoot '..\target\release\xcl.exe')
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path $Xcl)) { throw "no $Xcl - build it first (cargo build --release -p xcl)" }
$root = Split-Path $PSScriptRoot -Parent
$commit = (git -C $root rev-parse HEAD).Trim()
$dirty = [bool](git -C $root status --porcelain --untracked-files=no -- crates Cargo.toml Cargo.lock)

# Use ProcessStartInfo instead of Start-Process: the latter cannot build its
# environment block on Windows PowerShell 5.1 when PATH is present twice with
# different casing (which is possible on Windows).
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Xcl
$psi.Arguments = 'run "' + $Game + '"'
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
$psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
$env:XCL_NETWORK = '0'
$env:XCL_RUN_SECONDS = "$Seconds"
$env:XCL_ACTIVITY = '1'
$null = $p.Start()
$stdoutTask = $p.StandardOutput.ReadToEndAsync()
$stderrTask = $p.StandardError.ReadToEndAsync()
if (-not $p.WaitForExit(($Seconds + 120) * 1000)) {
    if (-not $p.HasExited) { $p.Kill() }
    $timedOut = $true
}
$p.WaitForExit()
$stdout = $stdoutTask.Result
$null = $stderrTask.Result
Remove-Item Env:XCL_RUN_SECONDS, Env:XCL_ACTIVITY

$logLine = ($stdout -split "`n" | Where-Object { $_ -match '^Log:\s+(.+)$' } | Select-Object -Last 1)
if (-not $logLine) { throw "XCL printed no log path; its output:`n$stdout" }
$log = ($logLine -replace '^Log:\s+', '').Trim()
$text = Get-Content $log -Raw -Encoding utf8

$frames = if ($text -match '\((\d+) frames shown, (\d+) PresentX calls\)') { "$($Matches[1]) (PresentX $($Matches[2]))" } else { 'none logged (the run did not reach its end)' }
$presents = [regex]::Matches($text, 'activity \(2 s\): (\d+) presents') | ForEach-Object { [int]$_.Groups[1].Value }
$lastPresents = ($presents | Select-Object -Last 3) -join ' '
$commits = [regex]::Matches($text, 'memory: (\d+) MB committed') | ForEach-Object { [int]$_.Groups[1].Value }
$peak = if ($commits) { ($commits | Measure-Object -Maximum).Maximum } else { 'n/a' }
$removals = ([regex]::Matches($text, 'device removed')).Count
$crashes = ([regex]::Matches($text, 'unhandled exception|process ends with status')).Count
# the adapter XCL ran on (report.json next to the log; XCL removes the last run's at start),
# not the first one Windows lists – on a machine with an iGPU that is often not the one used
$gpu = $null
$report = Join-Path (Split-Path $log -Parent) 'report.json'
if (Test-Path $report) {
    try { $gpu = ((Get-Content $report -Raw -Encoding utf8 | ConvertFrom-Json).d3d12.adapter -replace '\s*\([^()]*VRAM\)$', '') } catch { }
}
if (-not $gpu) { $gpu = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -notmatch 'Basic' } | Select-Object -First 1).Name }
$code = if ($timedOut) { 'killed (no exit)' } else { '0x{0:x}' -f $p.ExitCode }

@"
``````
xcl-smoke v1
commit: $commit$(if ($dirty) { ' (uncommitted changes)' })
gpu: $gpu
title: $(Split-Path $Game -Leaf)
seconds: $Seconds
exit: $code
frames shown: $frames
presents per 2 s (last 3): $lastPresents
device removals: $removals
crashes: $crashes
peak commit: $peak MB
``````
"@
