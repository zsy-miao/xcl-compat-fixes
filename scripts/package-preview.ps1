param(
    [Parameter(Mandatory)][string]$Executable,
    [Parameter(Mandatory)][string]$DxcArchive,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [string]$SourceDirectory = (Split-Path $PSScriptRoot -Parent)
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $SourceDirectory).Path
$revision = git -C $root rev-parse --verify HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify the source revision.' }
$Executable = (Resolve-Path -LiteralPath $Executable).Path
$DxcArchive = (Resolve-Path -LiteralPath $DxcArchive).Path
$compilerHash = 'ad31b1fc8443175d204f77a611fdb3ef2ec42759bdc2f1167368de24a4a7e7f1'
if ((Get-FileHash -LiteralPath $DxcArchive).Hash -ne $compilerHash) {
    throw 'Expected the Microsoft v1.9.2609 compiler distribution.'
}

$out = New-Item -ItemType Directory -Force -Path $OutputDirectory
$stage = Join-Path $out.FullName 'XCL-Windows-x64'
$zip = Join-Path $out.FullName 'XCL-Windows-x64.zip'
if ((Test-Path -LiteralPath $stage) -or (Test-Path -LiteralPath $zip)) {
    throw 'Use an empty staging directory so previous builds are preserved.'
}
New-Item -ItemType Directory -Path $stage | Out-Null
$licenses = New-Item -ItemType Directory -Path (Join-Path $stage 'licenses/DXC') -Force
Copy-Item -LiteralPath $Executable -Destination (Join-Path $stage 'xcl.exe')
foreach ($file in @('LICENSE', 'THIRD_PARTY.md')) {
    Copy-Item -LiteralPath (Join-Path $root $file) -Destination $stage
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$distribution = [System.IO.Compression.ZipFile]::OpenRead($DxcArchive)
try {
    $entries = @{
        'bin/x64/dxcompiler.dll' = (Join-Path $stage 'dxcompiler.dll')
        'bin/x64/dxil.dll' = (Join-Path $stage 'dxil.dll')
        'LICENCE-MIT.txt' = (Join-Path $licenses.FullName 'LICENCE-MIT.txt')
        'LICENSE-LLVM.txt' = (Join-Path $licenses.FullName 'LICENSE-LLVM.txt')
        'LICENSE-MS.txt' = (Join-Path $licenses.FullName 'LICENSE-MS.txt')
    }
    foreach ($entry in $distribution.Entries) {
        $key = $entry.FullName.Replace('\', '/')
        if ($entries.ContainsKey($key)) {
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $entries[$key], $false)
            $entries.Remove($key)
        }
    }
    if ($entries.Count) { throw "Missing files in the compiler distribution: $($entries.Keys -join ', ')" }
} finally {
    $distribution.Dispose()
}

$version = & (Join-Path $stage 'xcl.exe') --version
if ($LASTEXITCODE -ne 0 -or "$version" -notmatch '^xcl ') { throw 'The staged executable failed its version check.' }
$payload = @('xcl.exe','dxcompiler.dll','dxil.dll')
$onDemandLauncher = Join-Path $root 'scripts/Play-On-Demand.cmd'
if (Test-Path -LiteralPath $onDemandLauncher -PathType Leaf) {
    Copy-Item -LiteralPath $onDemandLauncher -Destination $stage
    $payload += 'Play-On-Demand.cmd'
}
$hashes = [ordered]@{}
foreach ($filename in $payload) {
    $hashes[$filename] = (Get-FileHash -LiteralPath (Join-Path $stage $filename)).Hash.ToLowerInvariant()
}
$utf8 = [System.Text.UTF8Encoding]::new($false)
$build = [ordered]@{
    revision = $revision
    binary_version = "$version"
    compiler_distribution = 'microsoft/DirectXShaderCompiler v1.9.2609'
    compiler_distribution_sha256 = $compilerHash
    payload_sha256 = $hashes
}
[IO.File]::WriteAllText((Join-Path $stage 'BUILD-INFO.json'), ($build | ConvertTo-Json -Depth 3), $utf8)
$start = "Extract the whole ZIP and start xcl.exe. Select your unpacked game's executable or MicrosoftGame.config. Keep both compiler DLLs beside xcl.exe. See the release notes for tested hardware, settings and known limitations."
if ($hashes.Contains('Play-On-Demand.cmd')) {
    $start = "Extract the whole ZIP. Start Play-On-Demand.cmd to use the offline settings tested on the RTX 3060 Ti. Select your unpacked game in XCL. These settings build pipelines when the game asks for them and disable speculative preloading. For the normal XCL settings, start xcl.exe. See the release notes for tested hardware and how far the game ran."
}
[IO.File]::WriteAllText((Join-Path $stage 'START-HERE.txt'), $start, $utf8)
$lines = foreach ($filename in $hashes.Keys) { "$($hashes[$filename])  $filename" }
[IO.File]::WriteAllLines((Join-Path $stage 'SHA256SUMS.txt'), [string[]]$lines, $utf8)
Compress-Archive -LiteralPath $stage -DestinationPath $zip -CompressionLevel Optimal
$archiveHash = (Get-FileHash -LiteralPath $zip).Hash.ToLowerInvariant()
[IO.File]::WriteAllText("$zip.sha256", "$archiveHash  XCL-Windows-x64.zip`n", $utf8)
Write-Host "Packaged $revision SHA256 $archiveHash"
