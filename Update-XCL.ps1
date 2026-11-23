[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$Rebuild,
    [switch]$BuildCurrent
)

$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$targetDirectory = Join-Path $projectDirectory 'target'
$releaseDirectory = Join-Path $targetDirectory 'release'
$packageDirectory = Join-Path $targetDirectory 'XCL-Windows-x64'
$packageArchive = Join-Path $targetDirectory 'XCL-Windows-x64.zip'
$sourceUrl = 'https://github.com/Prime1997/xcl-compat-fixes.git'
$patchBranch = 'local/game-compat'
$legacyPatchBranch = 'local/pr151-cpu-compat'
$pr151Revision = '236cc74954798356426d1d7bb1232cb135d1c123'
$pr162Revision = '641f55143864349fa3a5dd02557512af8fe0e43d'
$pr163Revision = '595b2b7a8446d9bf912ae602547476adcb21ef50'
$pr164Revision = 'ce0cb0a8244bf58f5a7e94ea8a4812d498eb4e2d'
$pr168Revision = '070a7378543b8efac3496e6a344393efadbe30b7'
$pr170Revision = 'aeadb23f0a4c29c0297392266e3de1cd9453251b'
$pr187Revision = '8c0346da21c0275fc7fbdb942887dc37ea18ab0f'
$pr192Revision = '50cff5dc2a7146e3e99bbfcbe1822627e06d62d8'
$pr195Revision = 'bf26412d368fcf6aecf7cc9ba9804aaf2ad3fc92'
$pr203Revision = '7cbd0088cc40a6456ffa3939c5bb307a23105cc1'
$textureAllocatorRevision = '38b05b7b6954cfa5bd21cec72f68823603ca2227'
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$testPassed = 0
$testIgnored = 0
$testSkips = @()
$transcribing = $false
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Read-Git {
    param([string[]]$GitArguments)
    $ErrorActionPreference = 'Continue'
    $result = & $script:gitCommand -C $projectDirectory @GitArguments 2>&1
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "Git failed ($code): $($result -join [Environment]::NewLine)" }
    return ($result | ForEach-Object { "$_" })
}

function Invoke-Checked {
    param([string]$Command, [string[]]$CommandArguments, [switch]$CountTests)
    # Windows PowerShell otherwise treats cargo's normal stderr as terminating errors.
    $ErrorActionPreference = 'Continue'
    $passedBefore = $script:testPassed
    & $Command @CommandArguments 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Host $line
        if ($CountTests -and $line -match '^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;') {
            $script:testPassed += [int]$Matches[1]
            $script:testIgnored += [int]$Matches[3]
        }
        if ($CountTests -and $line -match 'SKIPPED \S+: (.+)') {
            $script:testSkips += $Matches[1]
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Command $($CommandArguments -join ' ')" }
    if ($CountTests -and $script:testPassed -le $passedBefore) {
        throw "The test command reported no passing tests. Check that its filter still matches: $($CommandArguments -join ' ')"
    }
}

try {
    Set-Location -LiteralPath $projectDirectory
    $logDirectory = Join-Path $targetDirectory 'update-logs'
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $logPath = Join-Path $logDirectory "$stamp.log"
    Start-Transcript -LiteralPath $logPath | Out-Null
    $transcribing = $true

    $gitCommand = (Get-Command git.exe -ErrorAction Stop).Source
    if (-not (Test-Path -LiteralPath (Join-Path $projectDirectory '.git'))) {
        throw 'This folder has not been connected to the upstream XCL repository.'
    }
    $remoteUrl = Read-Git -GitArguments @('remote', 'get-url', 'upstream')
    if ($remoteUrl -notmatch '^https://github\.com/ivall/xcl(?:\.git)?/?$') {
        throw "Unexpected upstream repository: $remoteUrl"
    }
    $branch = Read-Git -GitArguments @('branch', '--show-current')
    $patchedBuild = $branch -in @($patchBranch, $legacyPatchBranch)
    if ($branch -ne 'main' -and -not $patchedBuild) { throw "Automatic updating supports main or the local compatibility branches (current: $branch)." }
    $changes = Read-Git -GitArguments @('status', '--porcelain', '--untracked-files=no')
    if ($changes) { throw 'There are local source changes. Save or reconcile them before updating; nothing was discarded.' }

    if (-not $BuildCurrent) {
        Invoke-Checked -Command $gitCommand -CommandArguments @('-C', $projectDirectory, 'fetch', 'upstream', 'refs/heads/main:refs/remotes/upstream/main')
    }
    $previousRevision = Read-Git -GitArguments @('rev-parse', 'HEAD')
    $upstreamLatestSeen = Read-Git -GitArguments @('rev-parse', 'refs/remotes/upstream/main')
    $upstreamRevision = $upstreamLatestSeen
    if ($BuildCurrent) {
        $upstreamRevision = Read-Git -GitArguments @('merge-base', 'HEAD', 'refs/remotes/upstream/main')
    }
    Write-Host "Installed source: $previousRevision"
    Write-Host "Upstream main:   $upstreamLatestSeen"
    if ($BuildCurrent) { Write-Host "Building pinned HEAD without fetching or updating source (upstream base: $upstreamRevision)." }
    if ($patchedBuild) { Write-Host 'This copy includes local compatibility fixes; they will be retained.' }
    & $gitCommand -C $projectDirectory merge-base --is-ancestor refs/remotes/upstream/main HEAD
    $sourceUpToDate = $LASTEXITCODE -eq 0
    if ($CheckOnly) {
        if ($sourceUpToDate) { Write-Host 'Your source already includes the current upstream main.' }
        else { Write-Host 'An upstream update is available. Run Update-XCL.cmd to build and install it.' }
        exit 0
    }

    if (-not $BuildCurrent -and -not $patchedBuild) {
        & $gitCommand -C $projectDirectory merge-base --is-ancestor HEAD refs/remotes/upstream/main
        if ($LASTEXITCODE -ne 0) { throw 'The source history has diverged. Automatic updating stopped without changing your files.' }
    }
    $buildInfoPath = Join-Path $packageDirectory 'BUILD-INFO.json'
    if (-not $BuildCurrent -and -not $Rebuild -and $sourceUpToDate -and (Test-Path -LiteralPath $buildInfoPath)) {
        $installedInfo = Get-Content -LiteralPath $buildInfoPath -Raw | ConvertFrom-Json
        if ($installedInfo.revision -eq $previousRevision -and (Test-Path -LiteralPath (Join-Path $packageDirectory 'xcl.exe'))) {
            Write-Host 'XCL is already built and up to date. Use -Rebuild to build again.'
            exit 0
        }
    }

    $projectPrefix = $projectDirectory.TrimEnd('\') + '\'
    foreach ($process in @(Get-Process -Name xcl -ErrorAction SilentlyContinue)) {
        $processPath = $process.Path
        if ($processPath -and $processPath.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Close XCL and its game before updating this copy.'
        }
    }
    $cargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
    if (Test-Path -LiteralPath $cargoBin) { $env:PATH = "$cargoBin;$env:PATH" }
    $cargoCommand = (Get-Command cargo.exe -ErrorAction Stop).Source
    $vswherePath = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswherePath) {
        $linkers = & $vswherePath -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'VC\Tools\MSVC\**\bin\Hostx64\x64\link.exe'
        if (-not $linkers) { throw 'Microsoft C++ build tools were not found. Restore them before rebuilding XCL.' }
    }

    $dxcSources = @{}
    foreach ($file in @('dxcompiler.dll', 'dxil.dll')) {
        foreach ($directory in @($releaseDirectory, $packageDirectory)) {
            $candidate = Join-Path $directory $file
            if (Test-Path -LiteralPath $candidate) { $dxcSources[$file] = $candidate; break }
        }
        if (-not $dxcSources.ContainsKey($file)) { throw "Missing $file. Restore it beside the current XCL executable before updating." }
    }

    $backupDirectory = Join-Path $targetDirectory "update-backups\$stamp"
    New-Item -ItemType Directory -Path $backupDirectory -Force | Out-Null
    if (Test-Path -LiteralPath $packageDirectory) { Copy-Item -LiteralPath $packageDirectory -Destination $backupDirectory -Recurse }
    if (Test-Path -LiteralPath $packageArchive) { Copy-Item -LiteralPath $packageArchive -Destination $backupDirectory }
    [IO.File]::WriteAllText((Join-Path $backupDirectory 'SOURCE-REVISION.txt'), "$previousRevision`r`n", $utf8)
    Invoke-Checked -Command $gitCommand -CommandArguments @('-C', $projectDirectory, 'update-ref', "refs/xcl/update-backups/$stamp", $previousRevision)
    Write-Host "Previous runnable package saved in $backupDirectory"

    if ($BuildCurrent) {
        Write-Host "Source remains pinned at $previousRevision."
    }
    elseif ($patchedBuild) {
        if (-not $sourceUpToDate) {
            try {
                Invoke-Checked -Command $gitCommand -CommandArguments @('-C', $projectDirectory, 'rebase', 'refs/remotes/upstream/main')
            }
            catch {
                $rebaseFailure = $_.Exception.Message
                try { Invoke-Checked -Command $gitCommand -CommandArguments @('-C', $projectDirectory, 'rebase', '--abort') }
                catch { Write-Host 'Rebase cleanup could not complete. Review the retained update log before continuing.' }
                throw "The local compatibility fixes could not be reconciled with upstream. The previous package was retained. $rebaseFailure"
            }
        }
    }
    else { Invoke-Checked -Command $gitCommand -CommandArguments @('-C', $projectDirectory, 'merge', '--ff-only', 'refs/remotes/upstream/main') }
    Invoke-Checked -Command $cargoCommand -CommandArguments @('build', '--locked', '--release', '--workspace')
    foreach ($file in @('dxcompiler.dll', 'dxil.dll')) {
        $destination = Join-Path $releaseDirectory $file
        if ($dxcSources[$file] -ne $destination) { Copy-Item -LiteralPath $dxcSources[$file] -Destination $destination -Force }
    }
    $env:XCL_DXC_DIR = $releaseDirectory
    $env:XCL_DXC_THREADS = '2'
    $env:XCL_NETWORK = '0'
    $savedLazyCompute = [Environment]::GetEnvironmentVariable('XCL_LAZY_COMPUTE', 'Process')
    try {
    # Baseline checks exercise eager compute; the opt-in mode gets its own check below.
    $env:XCL_LAZY_COMPUTE = '0'
    if ($patchedBuild) {
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'insn::tests', '--', '--test-threads=1', '--nocapture') -CountTests
    }
    if ($branch -eq $patchBranch) {
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'clzero_batch', '--', '--test-threads=1', '--nocapture') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'sse4a', '--', '--test-threads=1', '--nocapture') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'dstorage', '--', '--test-threads=1', '--nocapture', '--skip', 'external_pipe_parent', '--skip', 'external_pipe_descendant') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'hle::d3d12x::title_textures::mirror_tests', '--', '--test-threads=1', '--nocapture') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'console_depth_buffers_become_pc_ones', '--', '--test-threads=1', '--nocapture') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'real::depth_regression_tests', '--', '--test-threads=1', '--nocapture') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'typeless_depth_resource_accepts_typed_dsv_and_shader_views', '--', '--test-threads=1', '--nocapture') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'real::texture_regression_tests', '--', '--test-threads=1', '--nocapture') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'real::mesh_refusal_tests', '--', '--test-threads=1', '--nocapture') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'compute_first_use_is_a_persistent_per_title_opt_in', '--', '--test-threads=1', '--nocapture') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'pso_async::scheduling_tests', '--', '--test-threads=1', '--nocapture') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'hle::xg::tests', '--', '--test-threads=1', '--nocapture') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'console_mesh_streams_become_pc_ones', '--', '--test-threads=1', '--nocapture') -CountTests
        Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', 'gpu_mesh_mode_two_passes', '--', '--test-threads=1', '--nocapture') -CountTests
        foreach ($filter in @('clwb', 'real::this_first_tests', 'gpu::tiling::tests', 'gpu_memory_windows_from_the_heap', 'gpu_memory_windows_above_4_gib', 'gpu_raw_window_view_reaches_its_top', 'sparse::tests', 'vram::tests', 'vram::gpu_regression_tests', 'kernelx::lazy::tests', 'physical_pages_in_plain_reservations_are_lazy_and_round_trip', 'physical_pages_reject_an_unreserved_plain_destination', 'read_file_commits_a_lazy_destination_before_kernel_io')) {
            Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--lib', $filter, '--', '--test-threads=1', '--nocapture') -CountTests
        }
    }
    Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--test', 'disasm_e2e', '--test', 'e2e', '--test', 'rpf_e2e', '--test', 'gpu_compute_e2e', '--', '--test-threads=1', '--nocapture') -CountTests
    Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--test', 'bcpack_controls', 'independent_synthetic_corpus', '--', '--nocapture') -CountTests
    if ($branch -eq $patchBranch) {
        try {
            $env:XCL_LAZY_COMPUTE = '1'
            Invoke-Checked -Command $cargoCommand -CommandArguments @('test', '--locked', '--release', '-p', 'xcl', '--test', 'gpu_compute_e2e', 'console_wave64_compute_through_the_pso_hook_dispatch_and_read_back', '--', '--test-threads=1', '--nocapture') -CountTests
        }
        finally { $env:XCL_LAZY_COMPUTE = '0' }
    }
    }
    finally {
        if ($null -eq $savedLazyCompute) { Remove-Item -LiteralPath Env:\XCL_LAZY_COMPUTE -ErrorAction SilentlyContinue }
        else { $env:XCL_LAZY_COMPUTE = $savedLazyCompute }
    }
    if ($testPassed -eq 0) { throw 'The test command did not report any passing tests. The existing portable package was retained.' }

    $stageParent = Join-Path $targetDirectory "update-staging\$stamp"
    $stageDirectory = Join-Path $stageParent 'XCL-Windows-x64'
    New-Item -ItemType Directory -Path $stageDirectory -Force | Out-Null
    foreach ($file in @('xcl.exe', 'dxcompiler.dll', 'dxil.dll')) {
        Copy-Item -LiteralPath (Join-Path $releaseDirectory $file) -Destination $stageDirectory
    }
    foreach ($file in @('LICENSE', 'THIRD_PARTY.md')) { Copy-Item -LiteralPath (Join-Path $projectDirectory $file) -Destination $stageDirectory }
    $licenses = Join-Path $packageDirectory 'licenses'
    if (Test-Path -LiteralPath $licenses) { Copy-Item -LiteralPath $licenses -Destination $stageDirectory -Recurse }
    $revision = Read-Git -GitArguments @('rev-parse', 'HEAD')
    $info = [ordered]@{
        revision = $revision
        source_url = $sourceUrl
        upstream_url = $remoteUrl
        branch = $branch
        upstream_revision = $upstreamRevision
        upstream_latest_seen = $upstreamLatestSeen
        build_current = [bool]$BuildCurrent
        applied_patches = @()
        upstream_integrations = @()
        local_fixes = @()
        built_utc = [DateTime]::UtcNow.ToString('o')
        build_command = 'cargo build --locked --release --workspace'
        tests_passed = $testPassed
        tests_ignored = $testIgnored
        gpu_skip_reasons = @($testSkips)
        test_suites = @('disasm_e2e', 'e2e', 'rpf_e2e', 'gpu_compute_e2e')
    }
    $patchNote = ''
    if ($patchedBuild) {
        $info.applied_patches = @([ordered]@{ pull_request = 151; url = 'https://github.com/ivall/xcl/pull/151'; source_revision = $pr151Revision })
        $info.test_suites = @('insn::tests') + $info.test_suites
        $patchNote = "Includes PR #151 CPU compatibility fix ($pr151Revision).`r`nhttps://github.com/ivall/xcl/pull/151"
    }
    if ($branch -eq $patchBranch) {
        $info.local_fixes = @([ordered]@{
            name = 'clzero_loop_emulation'
            files = @('crates/xcl/src/eh.rs', 'crates/xcl/src/insn.rs', 'crates/xcl/src/gpu/shader.rs')
            scope = 'Batch recognized CLZERO/add/compare/back-jump loops with checked bounds and native writable-range validation; resume at the native compare to preserve flags and branch behavior. Synthetic title checks cover executed loops and memory boundaries.'
        }, [ordered]@{
            name = 'sse4a_emulation'
            files = @('crates/xcl/src/eh.rs', 'crates/xcl/src/insn.rs')
            scope = 'Use community SSE4a bit-field instruction emulation with saved XMM state validation and bounds checks; preserve the existing local CLZERO batching and other CPU handlers.'
        }, [ordered]@{
            name = 'bounded_bcpack_bc7'
            files = @('crates/xcl/src/hle/bcpack.rs', 'crates/xcl/src/hle/bcpack_bc7_general.rs', 'crates/xcl/src/hle/dstorage_bcpack.rs', 'crates/xcl/src/hle/dstorage.rs')
            scope = 'MIT entropy and independently reconstructed BC7 modes; unsupported initial/cross-mode prediction refuses atomically. XCL_BCPACK_EXTERNAL_DECODER enables a separate bounded research process for fitted history, with strict diagnostics, exact block sizes and a deadline covering inherited pipes. Fitted output is provisional; the external program is not included in the portable archive.'
        })
        $info.applied_patches += @(
            [ordered]@{ pull_request = 162; url = 'https://github.com/ivall/xcl/pull/162'; source_revision = $pr162Revision; scope = 'Graphics pipelines compile at first use with urgent compilation; compute first-use compilation is opt-in per title via lazy_compute_pipelines or XCL_LAZY_COMPUTE.' },
            [ordered]@{ pull_request = 163; url = 'https://github.com/ivall/xcl/pull/163'; source_revision = $pr163Revision },
            [ordered]@{ pull_request = 164; url = 'https://github.com/ivall/xcl/pull/164'; source_revision = $pr164Revision; scope = 'Crash and allocation diagnostics; existing PR #195 debug-message handling retained.' },
            [ordered]@{ pull_request = 168; url = 'https://github.com/ivall/xcl/pull/168'; source_revision = $pr168Revision; scope = 'Texture-address functionality included through PR #195 and local placed/committed texture-address and lifetime fixes.' },
            [ordered]@{ pull_request = 170; url = 'https://github.com/ivall/xcl/pull/170'; source_revision = $pr170Revision },
            [ordered]@{ pull_request = 187; url = 'https://github.com/ivall/xcl/pull/187'; source_revision = $pr187Revision },
            [ordered]@{ pull_request = 192; url = 'https://github.com/ivall/xcl/pull/192'; source_revision = $pr192Revision },
            [ordered]@{ pull_request = 195; url = 'https://github.com/ivall/xcl/pull/195'; source_revision = $pr195Revision; scope = 'Four graphics files; PR #192 replaces its XG implementation; local allocation and address regressions are recorded by the source revision.' },
            [ordered]@{ pull_request = 203; url = 'https://github.com/ivall/xcl/pull/203'; source_revision = $pr203Revision; scope = 'Memory-window and sparse-page integration from the user-specified source combination; compute first-use remains a per-title opt-in.' }
        )
        $info.applied_patches += @(
            [ordered]@{ pull_request = 225; url = 'https://github.com/ivall/xcl/pull/225'; source_revision = '327c24c98ea883f9fa586d647b5898e073ca6d34'; scope = 'Fallback texture arenas use aligned resource sizes and retire unused backing; existing shared target pages are retained.' },
            [ordered]@{ pull_request = 230; url = 'https://github.com/ivall/xcl/pull/230'; source_revision = 'b85602215a2f616f734187504d30279e4c249df4'; scope = 'Aligned sparse tile hash distribution and its bucket regression only; PR #203 sparse-page behavior retained.' },
            [ordered]@{ pull_request = 252; url = 'https://github.com/ivall/xcl/pull/252'; source_revision = '862a0d488d6ba1e7b3ffcfb10f8419251df2648b'; scope = 'Write-watch GPU-visible Windows fallback reservations; forced-fallback regression covers all three flag combinations.' },
            [ordered]@{ pull_request = 283; url = 'https://github.com/ivall/xcl/pull/283'; source_revision = 'a2bfbed181159950526ba4377fbe2873e87fc672'; initial_revision = '94bfcf00332e3270611067cc242bf186074d3e3b'; scope = 'Pair GPU-only reservations with CPU views and retarget existing textures after lazy physical-page map/unmap changes; preserve GPU title addresses, local Entry.state and lazy mappings. Actual GPU readback checks CPU rewrites and later mappings before first touch.' }
        )
        $info.upstream_integrations = @([ordered]@{
            source_commit = $textureAllocatorRevision
            url = "https://github.com/ivall/xcl/commit/$textureAllocatorRevision"
            files = @('crates/xcl/src/hle/d3d12x/vram.rs', 'crates/xcl/src/hle/d3d12x/device_x.rs', 'crates/xcl/src/hle/d3d12x/title_textures.rs', 'crates/xcl/src/hle/d3d12x/real.rs', 'crates/xcl/src/hle/kernelx.rs', 'crates/xcl/src/hle/kernelx/lazy.rs', 'crates/xcl/src/hle/dstorage.rs', 'crates/xcl/src/vfs_hooks.rs', 'crates/xcl/src/hle/d3d12x/cpu_writes.rs')
            scope = 'Complete VRAM texture-page allocator and console texture-size argument; per-frame refresh upload wait; unsupported mesh-layout refusal; typeless depth resource families with typed pipeline/clear/depth/stencil views; compatible lazy physical-page reservations and deferred-page handling for DirectStorage, file reads, and CPU completion writes. Local first-touch concurrency/failure/cleanup regressions and PR #187/#203 UMA/sparse changes retained.'
        })
        $info.upstream_integrations += [ordered]@{
            source_commit = '0cd0a53b584a8be0fd779d16e5826d97a0e66344'
            related_commit = 'af20b4b38e5a8003234f28091fb45e03aca68167'
            url = 'https://github.com/ivall/xcl/commit/af20b4b38e5a8003234f28091fb45e03aca68167'
            files = @('crates/xcl/src/insn.rs', 'crates/xcl/src/eh.rs')
            scope = 'SSE4a immediate/register INSERTQ and EXTRQ, including hardware-verified prefix behavior and upper-quadword results. Local floating-point context validation and the prior CLZERO batching remain; unrelated CPU hot-patching is not imported.'
        }
        $info.upstream_integrations += [ordered]@{
            source_commit = '7a033c8cd13194900cd49c01ed60a59c6c9559c7'
            regression_commit = '2f5979ecdf686436c90c479727f2dcc16973072d'
            url = 'https://github.com/ivall/xcl/pull/263'
            scope = 'Disambiguate the hidden return buffer from the device when both arguments contain an adopted vtable pointer.'
        }
        $info.local_fixes += [ordered]@{
            name = 'clwb_loop_emulation'
            files = @('crates/xcl/src/eh.rs', 'crates/xcl/src/insn.rs')
            scope = 'Batch the repeated cache writebacks observed after account selection and resume at the native comparison. The game instruction bytes stay intact.'
            related_upstream_commit = 'b25582a39244fb9576f2a751ba4603b521f10c20'
        }
        $info.upstream_integrations += [ordered]@{
            contributor = 'gw1337x'
            url = 'https://github.com/ivall/xcl/pull/283#issuecomment-7116115692'
            files = @('crates/xcl/src/gpu/tiling.rs')
            scope = 'Copy contiguous texture rows and reuse per-block offsets for tiled data. Differential tests cover partial elements, uncommon base swizzles, mip tails and array slices.'
        }
        $info.test_suites = @('clwb', 'real::this_first_tests', 'gpu::tiling::tests') + $info.test_suites
        $info.test_suites = @('console_depth_buffers_become_pc_ones', 'real::depth_regression_tests', 'typeless_depth_resource_accepts_typed_dsv_and_shader_views', 'real::texture_regression_tests', 'real::mesh_refusal_tests', 'compute_first_use_is_a_persistent_per_title_opt_in', 'pso_async::scheduling_tests', 'hle::xg::tests', 'console_mesh_streams_become_pc_ones', 'gpu_mesh_mode_two_passes', 'gpu_memory_windows_from_the_heap', 'gpu_memory_windows_above_4_gib', 'gpu_raw_window_view_reaches_its_top', 'sparse::tests', 'vram::tests', 'vram::gpu_regression_tests', 'kernelx::lazy::tests', 'physical_pages_in_plain_reservations_are_lazy_and_round_trip', 'physical_pages_reject_an_unreserved_plain_destination', 'read_file_commits_a_lazy_destination_before_kernel_io', 'gpu_compute_e2e::console_wave64_compute_through_the_pso_hook_dispatch_and_read_back (XCL_LAZY_COMPUTE=1)') + $info.test_suites
        $info.test_suites = @('dstorage (subprocess fixture entry points excluded; inherited-pipe regression executed)', 'hle::d3d12x::title_textures::mirror_tests', 'bcpack_controls::independent_synthetic_corpus (2049 controls, 2066 blocks)') + $info.test_suites
        $patchNote += "`r`nIncludes PR #163 depth fixes ($pr163Revision) and PR #170 mesh pipeline fixes ($pr170Revision)."
        $patchNote += "`r`nIncludes PR #192 texture tiling ($pr192Revision) and PR #195 graphics changes ($pr195Revision)."
        $patchNote += "`r`nIncludes PR #162 first-use pipelines ($pr162Revision); compute first-use is disabled by default and can be enabled per title with lazy_compute_pipelines. XCL_LAZY_COMPUTE overrides that setting."
        $patchNote += "`r`nIncludes PR #187 memory windows ($pr187Revision), PR #203 integration ($pr203Revision), and PR #164 diagnostics ($pr164Revision)."
        $patchNote += "`r`nIncludes selected texture allocator/upload, mesh-layout refusal, typeless depth/view consistency, and lazy physical-page changes from upstream $textureAllocatorRevision; BUILD-INFO.json records the nine-file integration scope."
        $patchNote += "`r`nThe XG implementation is from #192; the source revision also records local texture-address, allocation cleanup and regression-test changes."
        $info.test_suites = @('clzero_batch', 'sse4a') + $info.test_suites
        $patchNote += "`r`nIncludes local CLZERO loop emulation that avoids repeating a Windows memory query for every 64-byte zero operation."
        $patchNote += "`r`nIncludes community SSE4a immediate/register bit-field emulation (0cd0a53 and hardware follow-up af20b4b), with local saved-context validation."
        $patchNote += "`r`nIncludes PR #225 fallback texture-arena memory improvements and PR #230 aligned sparse tile hash fix, found by the community-update watcher."
    }
    [IO.File]::WriteAllText((Join-Path $stageDirectory 'BUILD-INFO.json'), ($info | ConvertTo-Json -Depth 5), $utf8)
    $startText = @"
XCL - Windows x64 release build
Source revision: $revision
Source: https://github.com/Prime1997/xcl-compat-fixes/commit/$revision
Upstream project: https://github.com/ivall/xcl
$patchNote

Extract the whole ZIP, then double-click xcl.exe. Choose your unpacked game's
executable or MicrosoftGame.config. Keep both compiler DLLs beside xcl.exe.
No games are included. Saves and logs go to %LOCALAPPDATA%\XCL\<title>\.

The bundled Microsoft DXC compiler has its own terms in licenses\DXC. Upstream's
THIRD_PARTY.md describes the source repository; this local package also includes
dxcompiler.dll and dxil.dll from Microsoft's official DXC release.

Validation: $testPassed tests passed; $($testSkips.Count) GPU skip messages.
Retail games were not tested by the updater. See BUILD-INFO.json for details.

To update again, open Update-XCL.cmd in the source folder:
$projectDirectory
The updater follows upstream main, retains this branch's local fixes, and backs up the
previous portable package before building. It does not run on a schedule.
To rebuild the pinned source without fetching or rebasing, run Update-XCL.cmd -BuildCurrent.
"@
    [IO.File]::WriteAllText((Join-Path $stageDirectory 'START-HERE.txt'), $startText, $utf8)
    $hashLines = foreach ($file in @('xcl.exe', 'dxcompiler.dll', 'dxil.dll')) {
        $hash = Get-FileHash -LiteralPath (Join-Path $stageDirectory $file) -Algorithm SHA256
        "$($hash.Hash.ToLower())  $file"
    }
    [IO.File]::WriteAllLines((Join-Path $stageDirectory 'SHA256SUMS.txt'), [string[]]$hashLines, $utf8)
    Invoke-Checked -Command (Join-Path $stageDirectory 'xcl.exe') -CommandArguments @('--version')
    $stagedArchive = Join-Path $stageParent 'XCL-Windows-x64.zip'
    Compress-Archive -LiteralPath $stageDirectory -DestinationPath $stagedArchive -CompressionLevel Optimal
    New-Item -ItemType Directory -Path $packageDirectory -Force | Out-Null
    Get-ChildItem -LiteralPath $stageDirectory -Force | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $packageDirectory -Recurse -Force }
    Copy-Item -LiteralPath $stagedArchive -Destination $packageArchive -Force
    Write-Host "Updated XCL to $revision. $testPassed tests passed."
    Write-Host "Open: $(Join-Path $packageDirectory 'xcl.exe')"
    Write-Host "ZIP:  $packageArchive"
    exit 0
}
catch {
    Write-Host "XCL update stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'The saved backup and update log are retained in the target folder.'
    exit 1
}
finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
}
