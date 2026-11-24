$ErrorActionPreference = 'Stop'
try {
    if (Get-Process -Name xcl -ErrorAction SilentlyContinue) {
        throw 'XCL is already running. Close its game window before starting another copy.'
    }
    $executable = Join-Path $PSScriptRoot 'target\XCL-Windows-x64\xcl.exe'
    $selection = Join-Path $env:LOCALAPPDATA 'XCL\last-game.txt'
    if (-not (Test-Path -LiteralPath $executable)) { throw 'The rebuilt XCL executable is missing.' }
    if (-not (Test-Path -LiteralPath $selection)) { throw 'Open xcl.exe once and select the game first.' }
    $game = (Get-Content -LiteralPath $selection -Raw).Trim()
    if (-not (Test-Path -LiteralPath $game)) { throw 'The previously selected game could not be found.' }
    foreach ($setting in @('XCL_RUN_SECONDS', 'XCL_FRAME_DUMP', 'XCL_FRAME_DUMP_FULL', 'XCL_PRESS_AT', 'XCL_PRESS_A_AT', 'XCL_THREAD_DUMP_AT', 'XCL_TRACE_MEM', 'XCL_ACTIVITY', 'XCL_VRAM_ARENAS')) {
        Remove-Item -LiteralPath "Env:$setting" -ErrorAction SilentlyContinue
    }
    $env:XCL_NETWORK = '0'
    $env:XCL_DXC_THREADS = '2'
    $env:XCL_LAZY_PSO = '1'
    $env:XCL_LAZY_COMPUTE = '1'
    $env:XCL_HEAP_WINDOWS = '1'
    $ErrorActionPreference = 'Continue'
    & $executable run $game
    exit $LASTEXITCODE
}
catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
