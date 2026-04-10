param(
    [string]$BuildDir = "build_ninja",
    [string]$OutputDir = "",
    [string]$MakensisPath = "",
    [string]$SignToolPath = "",
    [string]$CodeSignCertSha1 = "",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [switch]$SkipChecksum
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
$artifactScript = Join-Path $PSScriptRoot "sign-and-verify-artifact.ps1"

$env:NSISDIR = $nsisRoot
$env:Path = "$nsisRoot;$nsisRoot\Bin;$env:Path"

$cmakeArguments = @(
    "-S", ".",
    "-B", $buildRoot.Path,
    "-DJAKAL_CORE_MAKENSIS_EXECUTABLE=$($resolvedMakensis.Path)"
)

if (-not [string]::IsNullOrWhiteSpace($CodeSignCertSha1)) {
    $cmakeArguments += @(
        "-DJAKAL_CORE_ENABLE_CODE_SIGNING=ON",
        "-DJAKAL_CORE_CODESIGN_CERT_SHA1=$CodeSignCertSha1",
        "-DJAKAL_CORE_CODESIGN_TIMESTAMP_URL=$TimestampUrl"
    )
    if (-not [string]::IsNullOrWhiteSpace($SignToolPath)) {
        $resolvedSignTool = Resolve-Path -Path $SignToolPath -ErrorAction Stop
        $SignToolPath = $resolvedSignTool.Path
        $cmakeArguments += "-DJAKAL_CORE_SIGNTOOL_PATH=$SignToolPath"
    }
}

& cmake @cmakeArguments
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed."
}

& cpack --config (Join-Path $buildRoot.Path "CPackConfig.cmake") -G NSIS -B $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "CPack NSIS generation failed."
}

if (-not (Test-Path -Path $artifactScript)) {
    throw "Artifact signing helper not found: $artifactScript"
}

$packagedArtifacts = @(
    Get-ChildItem -Path $OutputDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension.ToLowerInvariant() -in @(".exe", ".msi", ".zip") }
)
if ($packagedArtifacts.Count -eq 0) {
    $packagedArtifacts = @(
        Get-ChildItem -Path $OutputDir -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Extension.ToLowerInvariant() -in @(".exe", ".msi", ".zip") -and
                $_.FullName -notmatch '[\\/]\_CPack_Packages[\\/]'
            }
    )
}

if ($packagedArtifacts.Count -eq 0) {
    throw "No packaged installer artifacts were found under $OutputDir"
}

foreach ($artifact in $packagedArtifacts) {
    $artifactArguments = @{
        ArtifactPath = $artifact.FullName
    }

    if (-not $SkipChecksum) {
        $artifactArguments.WriteChecksum = $true
        $artifactArguments.VerifyChecksum = $true
    }

    if ($artifact.Extension.ToLowerInvariant() -in @(".exe", ".msi") -and -not [string]::IsNullOrWhiteSpace($CodeSignCertSha1)) {
        $artifactArguments.Sign = $true
        $artifactArguments.RequireSignature = $true
        $artifactArguments.UseSignToolVerification = $true
        $artifactArguments.CertificateThumbprint = $CodeSignCertSha1
        $artifactArguments.ExpectedThumbprint = $CodeSignCertSha1
        $artifactArguments.TimestampUrl = $TimestampUrl
        if (-not [string]::IsNullOrWhiteSpace($SignToolPath)) {
            $artifactArguments.SignToolPath = $SignToolPath
        }
    }

    & $artifactScript @artifactArguments
}

Write-Host ("Generated NSIS package under {0}" -f $OutputDir)
