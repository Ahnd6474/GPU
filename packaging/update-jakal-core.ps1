param(
    [Parameter(Mandatory = $true)]
    [string]$InstallerPath,
    [string]$TargetDir = "",
    [switch]$Quiet,
    [switch]$SkipVerification,
    [string]$ExpectedSignerThumbprint = "",
    [switch]$RequireChecksum
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolvedInstaller = Resolve-Path -Path $InstallerPath -ErrorAction Stop
$extension = [System.IO.Path]::GetExtension($resolvedInstaller.Path).ToLowerInvariant()
$artifactScript = Join-Path $PSScriptRoot "sign-and-verify-artifact.ps1"

if (-not $SkipVerification) {
    if (-not (Test-Path -Path $artifactScript)) {
        throw "Artifact verification helper not found: $artifactScript"
    }

    $verificationArguments = @{
        ArtifactPath = $resolvedInstaller.Path
    }

    if (-not [string]::IsNullOrWhiteSpace($ExpectedSignerThumbprint)) {
        $verificationArguments.ExpectedThumbprint = $ExpectedSignerThumbprint
    }

    if ($RequireChecksum) {
        $verificationArguments.RequireChecksum = $true
    }

    if (Test-Path -Path "$($resolvedInstaller.Path).sha256") {
        $verificationArguments.VerifyChecksum = $true
    }

    if ($extension -in @(".exe", ".msi")) {
        $verificationArguments.RequireSignature = $true
    }

    & $artifactScript @verificationArguments
}

switch ($extension) {
    ".msi" {
        $arguments = @("/i", "`"$($resolvedInstaller.Path)`"")
        if ($Quiet) {
            $arguments += "/qn"
        }
        Start-Process -FilePath "msiexec.exe" -ArgumentList $arguments -Wait
        break
    }
    ".exe" {
        $arguments = @()
        if ($Quiet) {
            $arguments += "/S"
        }
        Start-Process -FilePath $resolvedInstaller.Path -ArgumentList $arguments -Wait
        break
    }
    ".zip" {
        if ([string]::IsNullOrWhiteSpace($TargetDir)) {
            throw "ZIP updates require -TargetDir."
        }
        $resolvedTarget = New-Item -ItemType Directory -Force -Path $TargetDir
        Expand-Archive -Path $resolvedInstaller.Path -DestinationPath $resolvedTarget.FullName -Force
        break
    }
    default {
        throw "Unsupported installer type: $extension"
    }
}
