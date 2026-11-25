# Run with powershell.exe or pwsh.exe. Checks the `title:` line of smoke.ps1's block (#677): the
# hash of the game folder's name, as Python's hashlib.sha256(name.strip().encode()).hexdigest()[:8]
# gives it, or the -Title label. A fixture stands in for xcl.exe.
$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-title-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
try {
    $xcl = Join-Path $fixture 'xcl.cmd'
    [IO.File]::WriteAllText($xcl, "@echo off`r`necho Log: %~dp0xcl.log`r`n")
    [IO.File]::WriteAllText((Join-Path $fixture 'xcl.log'), "(1 frames shown, 1 PresentX calls)`n")
    function Get-SmokeTitle([hashtable]$Arguments) {
        $block = & $smoke -Seconds 1 -Xcl $xcl @Arguments
        $line = $block -split "`n" | Where-Object { $_ -like 'title:*' }
        $line.Substring('title: '.Length).Trim()
    }
    # names built from character codes: this file stays ASCII for Windows PowerShell 5.1
    $cases = @(
        @('SomeTitle', '27693731'),
        @(('Caf' + [char]0xE9 + ' Folder'), '5256b5ea'),
        @(([string][char]0x65E5 + [char]0x672C + ' pack ' + [char]::ConvertFromUtf32(0x1F3AE)), 'dbd98f4b'),
        @('  Padded Name  ', 'f8b1d479')
    )
    foreach ($case in $cases) {
        foreach ($game in @("D:\Games\$($case[0])", "D:\Games\$($case[0])\")) {
            $got = Get-SmokeTitle @{ Game = $game }
            if ($got -ne $case[1]) { throw "title of '$game' is '$got', expected '$($case[1])'" }
        }
    }
    $got = Get-SmokeTitle @{ Game = 'D:\Games\SomeTitle'; Title = '  my label: one  ' }
    if ($got -ne 'my label: one') { throw "-Title printed '$got'" }
    foreach ($bad in @('', '   ', "a`nb", 'a`b', ('a' + [char]0x2028 + 'commit: 0123456'), ('a' + [char]0x85 + 'b'))) {
        $threw = $false
        try { $null = & $smoke -Game 'D:\Games\SomeTitle' -Seconds 1 -Xcl $xcl -Title $bad } catch { $threw = $true }
        if (-not $threw) { throw "-Title '$bad' was accepted" }
    }
    'smoke-title: ok'
} finally {
    Remove-Item -LiteralPath $fixture -Recurse -Force
}
