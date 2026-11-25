# Run with powershell.exe or pwsh.exe. Tests scripts/smoke.ps1 -OutDir artifact saving.
$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'smoke.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-test-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$mockXcl = Join-Path $fixture 'mock_xcl.cmd'
$gameDir = Join-Path $fixture 'MockGame'
$null = New-Item -ItemType Directory -Path $gameDir
$runDir = Join-Path $fixture 'run'
$null = New-Item -ItemType Directory -Path $runDir
$mockLog = Join-Path $runDir 'mock.log'
$mockReport = Join-Path $runDir 'report.json'
$outDir1 = Join-Path $fixture 'artifacts1'
$outDir2 = Join-Path $fixture 'artifacts2'

try {
    # Generate mock log and report
    @"
[00:00:01] activity (2 s): 30 presents
[00:00:03] activity (2 s): 30 presents
[00:00:05] activity (2 s): 30 presents
[00:00:05] memory: 128 MB committed
(90 frames shown, 90 PresentX calls)
"@ | Set-Content -LiteralPath $mockLog -Encoding utf8

    '{"d3d12": {"adapter": "Mock Synthetic Adapter (16384 MB VRAM)"}}' |
        Set-Content -LiteralPath $mockReport -Encoding utf8

    # Mock XCL script outputs Log: path
    @"
@echo off
echo Running mock game...
echo Log: $mockLog
exit /b 0
"@ | Set-Content -LiteralPath $mockXcl -Encoding Ascii

    # Run 1: with report.json present
    $output1 = & $runner -Game $gameDir -Xcl $mockXcl -Seconds 5 -OutDir $outDir1 -Title 'MockRun1'
    if ($LASTEXITCODE -ne 0) { throw "smoke.ps1 exited with code $LASTEXITCODE on run 1" }

    if (-not (Test-Path (Join-Path $outDir1 'xcl.log'))) { throw 'run 1 did not preserve xcl.log' }
    if (-not (Test-Path (Join-Path $outDir1 'report.json'))) { throw 'run 1 did not preserve report.json' }
    if (-not (Test-Path (Join-Path $outDir1 'manifest.json'))) { throw 'run 1 did not write manifest.json' }

    $manifest1 = Get-Content (Join-Path $outDir1 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ($manifest1.gpu -ne 'Mock Synthetic Adapter') { throw "unexpected gpu: $($manifest1.gpu)" }
    if ($manifest1.seconds -ne 5) { throw "unexpected seconds: $($manifest1.seconds)" }
    if ($manifest1.frames -ne '90 (PresentX 90)') { throw "unexpected frames: $($manifest1.frames)" }
    if ($manifest1.title -ne 'MockRun1') { throw "unexpected title: $($manifest1.title)" }

    $smokeBlock = $output1 -join "`n"
    if ($smokeBlock -notmatch 'xcl-smoke v1') { throw "smoke block missing header: $smokeBlock" }
    if ($smokeBlock -notmatch 'gpu: Mock Synthetic Adapter') { throw "smoke block missing mock gpu: $smokeBlock" }

    # Run 2: remove report.json to test missing report case; reuse $outDir1 to verify stale report.json is purged
    Remove-Item -LiteralPath $mockReport -Force
    $output2 = & $runner -Game $gameDir -Xcl $mockXcl -Seconds 10 -OutDir $outDir1 -Title 'MockRun2'
    if ($LASTEXITCODE -ne 0) { throw "smoke.ps1 exited with code $LASTEXITCODE on run 2" }

    if (-not (Test-Path (Join-Path $outDir1 'xcl.log'))) { throw 'run 2 did not preserve xcl.log' }
    if (Test-Path (Join-Path $outDir1 'report.json')) { throw 'run 2 did not clean up stale report.json' }
    if (-not (Test-Path (Join-Path $outDir1 'manifest.json'))) { throw 'run 2 did not write manifest.json' }

    $manifest2 = Get-Content (Join-Path $outDir1 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ($manifest2.seconds -ne 10) { throw "unexpected seconds: $($manifest2.seconds)" }
    if ($manifest2.title -ne 'MockRun2') { throw "unexpected title: $($manifest2.title)" }

    Write-Host "test-smoke-artifacts: all assertions passed successfully."
} finally {
    if (Test-Path -LiteralPath $fixture) {
        Remove-Item -LiteralPath $fixture -Recurse -Force
    }
}
