param(
    [Parameter(Mandatory = $true)]
    [string]$ArtifactPath,
    [string]$SignToolPath = "",
    [string]$CertificateThumbprint = "",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [string]$ExpectedThumbprint = "",
    [switch]$Sign,
    [switch]$RequireSignature,
    [switch]$UseSignToolVerification,
    [switch]$WriteChecksum,
    [switch]$VerifyChecksum,
    [switch]$RequireChecksum
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-SignToolExecutable {
    param([string]$Candidate)

    if (-not [string]::IsNullOrWhiteSpace($Candidate)) {
        return (Resolve-Path -Path $Candidate -ErrorAction Stop).Path
    }

    $command = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    throw "signtool.exe not found. Pass -SignToolPath."
}

function Normalize-Thumbprint {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ""
    }

    return ($Value -replace '\s+', '').ToUpperInvariant()
}

function Get-ChecksumPath {
    param([string]$Path)

    return "{0}.sha256" -f $Path
}

function Write-ChecksumSidecar {
    param([string]$Path)

    $resolved = Resolve-Path -Path $Path -ErrorAction Stop
    $hash = Get-FileHash -Path $resolved.Path -Algorithm SHA256
    $content = "{0} *{1}" -f $hash.Hash.ToLowerInvariant(), [System.IO.Path]::GetFileName($resolved.Path)
    $checksumPath = Get-ChecksumPath -Path $resolved.Path
    Set-Content -Path $checksumPath -Value $content -Encoding ascii
    return $checksumPath
}

function Test-ChecksumSidecar {
    param(
        [string]$Path,
        [bool]$RequireSidecar
    )

    $resolved = Resolve-Path -Path $Path -ErrorAction Stop
    $checksumPath = Get-ChecksumPath -Path $resolved.Path
    if (-not (Test-Path -Path $checksumPath)) {
        if ($RequireSidecar) {
            throw "Checksum sidecar missing: $checksumPath"
        }
        return $false
    }

    $line = (Get-Content -Path $checksumPath -Encoding ascii | Select-Object -First 1).Trim()
    if ([string]::IsNullOrWhiteSpace($line)) {
        throw "Checksum sidecar is empty: $checksumPath"
    }

    $parts = $line -split '\s+', 2
    if ($parts.Count -lt 1 -or [string]::IsNullOrWhiteSpace($parts[0])) {
        throw "Checksum sidecar is malformed: $checksumPath"
    }

    $expectedHash = $parts[0].ToLowerInvariant()
    $actualHash = (Get-FileHash -Path $resolved.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($expectedHash -ne $actualHash) {
        throw "SHA256 mismatch for $($resolved.Path). Expected $expectedHash but got $actualHash"
    }

    return $true
}

function Test-AuthenticodeSignatureState {
    param(
        [string]$Path,
        [bool]$RequireValidSignature,
        [string]$RequiredThumbprint
    )

    $resolved = Resolve-Path -Path $Path -ErrorAction Stop
    $signature = Get-AuthenticodeSignature -FilePath $resolved.Path
    if ($RequireValidSignature -and $signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
        throw "Authenticode signature is not valid for $($resolved.Path): $($signature.Status)"
    }

    if (-not [string]::IsNullOrWhiteSpace($RequiredThumbprint)) {
        if ($null -eq $signature.SignerCertificate) {
            throw "Expected signer thumbprint $RequiredThumbprint but no signer certificate was found for $($resolved.Path)"
        }

        $actualThumbprint = Normalize-Thumbprint -Value $signature.SignerCertificate.Thumbprint
        if ($actualThumbprint -ne (Normalize-Thumbprint -Value $RequiredThumbprint)) {
            throw "Signer thumbprint mismatch for $($resolved.Path). Expected $RequiredThumbprint but got $actualThumbprint"
        }
    }

    return $signature
}

function Invoke-Signing {
    param(
        [string]$Path,
        [string]$ToolPath,
        [string]$Thumbprint,
        [string]$Rfc3161Url
    )

    if ([string]::IsNullOrWhiteSpace($Thumbprint)) {
        throw "Signing requested but -CertificateThumbprint is empty."
    }

    & $ToolPath sign /fd SHA256 /sha1 (Normalize-Thumbprint -Value $Thumbprint) /tr $Rfc3161Url /td SHA256 $Path
    if ($LASTEXITCODE -ne 0) {
        throw "signtool sign failed for $Path"
    }
}

function Invoke-SignatureVerification {
    param(
        [string]$Path,
        [string]$ToolPath
    )

    & $ToolPath verify /pa /v $Path
    if ($LASTEXITCODE -ne 0) {
        throw "signtool verify failed for $Path"
    }
}

$resolvedArtifact = Resolve-Path -Path $ArtifactPath -ErrorAction Stop
$extension = [System.IO.Path]::GetExtension($resolvedArtifact.Path).ToLowerInvariant()
$supportsAuthenticode = $extension -in @(".exe", ".msi", ".ps1", ".psm1", ".psd1", ".cat")
$resolvedSignTool = $null

if ($Sign) {
    if (-not $supportsAuthenticode) {
        throw "Signing is not supported for $extension artifacts: $($resolvedArtifact.Path)"
    }

    $resolvedSignTool = Resolve-SignToolExecutable -Candidate $SignToolPath
    Invoke-Signing -Path $resolvedArtifact.Path -ToolPath $resolvedSignTool -Thumbprint $CertificateThumbprint -Rfc3161Url $TimestampUrl
}

if ($WriteChecksum) {
    Write-ChecksumSidecar -Path $resolvedArtifact.Path | Out-Null
}

if ($supportsAuthenticode -and ($Sign -or $RequireSignature -or -not [string]::IsNullOrWhiteSpace($ExpectedThumbprint))) {
    Test-AuthenticodeSignatureState -Path $resolvedArtifact.Path -RequireValidSignature:$true -RequiredThumbprint $ExpectedThumbprint | Out-Null
    if ($UseSignToolVerification) {
        if ($null -eq $resolvedSignTool) {
            $resolvedSignTool = Resolve-SignToolExecutable -Candidate $SignToolPath
        }
        Invoke-SignatureVerification -Path $resolvedArtifact.Path -ToolPath $resolvedSignTool
    }
}

if ($VerifyChecksum -or $RequireChecksum) {
    Test-ChecksumSidecar -Path $resolvedArtifact.Path -RequireSidecar:$RequireChecksum | Out-Null
}
