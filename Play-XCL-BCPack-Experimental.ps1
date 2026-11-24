$ErrorActionPreference = 'Stop'
$decoder = Join-Path $PSScriptRoot 'target\bcpack-research\external-oracle\build\release\bcpack-external-stdio.exe'
if (-not (Test-Path -LiteralPath $decoder -PathType Leaf)) {
    Write-Host 'The experimental BCPack decoder has not been built in this copy.' -ForegroundColor Red
    exit 1
}
$hadPreviousDecoder = Test-Path -LiteralPath Env:XCL_BCPACK_EXTERNAL_DECODER
$previousDecoder = $env:XCL_BCPACK_EXTERNAL_DECODER
try {
    $env:XCL_BCPACK_EXTERNAL_DECODER = $decoder
    Write-Host 'Starting XCL with the experimental BCPack decoder.'
    & (Join-Path $PSScriptRoot 'Play-XCL-Offline.ps1')
}
finally {
    if ($hadPreviousDecoder) {
        [Environment]::SetEnvironmentVariable('XCL_BCPACK_EXTERNAL_DECODER', $previousDecoder, 'Process')
    }
    else {
        Remove-Item -LiteralPath Env:XCL_BCPACK_EXTERNAL_DECODER -ErrorAction SilentlyContinue
    }
}
