param(
    [string]$BuildDir = "build_ninja",
    [string]$OutputDir = "",
    [string]$MakensisPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($MakensisPath)) {
    $candidate = Join-Path $PSScriptRoot "..\tools\nsis\portable\nsis-3.11\makensis.exe"
    $resolvedCandidate = Resolve-Path -Path $candidate -ErrorAction SilentlyContinue
    if ($null -ne $resolvedCandidate) {
        $MakensisPath = $resolvedCandidate.Path
    }
}

if ([string]::IsNullOrWhiteSpace($MakensisPath)) {
    throw "makensis.exe not found. Pass -MakensisPath or install/extract NSIS under tools\\nsis\\portable."
}

$resolvedMakensis = Resolve-Path -Path $MakensisPath -ErrorAction Stop
$nsisRoot = Split-Path -Path $resolvedMakensis.Path -Parent
$buildRoot = Resolve-Path -Path $BuildDir -ErrorAction Stop
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $buildRoot.Path "dist-nsis"
}

$env:NSISDIR = $nsisRoot
$env:Path = "$nsisRoot;$nsisRoot\Bin;$env:Path"

& cmake -S . -B $buildRoot.Path "-DJAKAL_CORE_MAKENSIS_EXECUTABLE=$($resolvedMakensis.Path)"
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed."
}

& cpack --config (Join-Path $buildRoot.Path "CPackConfig.cmake") -G NSIS -B $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "CPack NSIS generation failed."
}

Write-Host ("Generated NSIS package under {0}" -f $OutputDir)
