param(
    [string]$InstallRoot = "",
    [string]$RuntimeRoot = "",
    [ValidateSet("interactive", "auto", "cpu-only", "intel-level-zero", "vulkan-runtime", "opencl-fallback")]
    [string]$Backend = "interactive",
    [ValidateSet("interactive", "recommended", "none", "all")]
    [string]$Prerequisites = "interactive",
    [switch]$Quiet
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-DefaultInstallRoot {
    $root = Split-Path -Path $PSScriptRoot -Parent
    $root = Split-Path -Path $root -Parent
    return Split-Path -Path $root -Parent
}

function Resolve-JakalInstallRoot {
    param([string]$Candidate)

    if (-not [string]::IsNullOrWhiteSpace($Candidate)) {
        return (Resolve-Path -Path $Candidate).Path
    }

    return (Resolve-Path -Path (Get-DefaultInstallRoot)).Path
}

function Get-DoctorReport {
    param(
        [string]$CliPath,
        [string]$InstallRootPath
    )

    $arguments = @("doctor", "--json", "--runtime-root", $InstallRootPath)
    $json = & $CliPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "jakal_core_cli doctor failed with exit code $LASTEXITCODE"
    }

    return $json | ConvertFrom-Json
}

function Write-SectionLine {
    param(
        [System.Collections.Generic.List[string]]$Lines,
        [string]$Section,
        [string]$Key,
        [int]$Value
    )

    if ($Lines.Count -gt 0 -and $Lines[$Lines.Count - 1] -ne "") {
        $Lines.Add("")
    }
    if ($Lines.Count -eq 0 -or $Lines[$Lines.Count - 1] -ne "[$Section]") {
        $Lines.Add("[$Section]")
    }
    $Lines.Add("$Key=$Value")
}

function Write-BackendPresetConfig {
    param(
        [string]$ConfigPath,
        [string]$BackendId
    )

    $configDir = Split-Path -Path $ConfigPath -Parent
    New-Item -ItemType Directory -Force -Path $configDir | Out-Null

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("[runtime]")
    $lines.Add("")
    $lines.Add("[paths]")
    $lines.Add("")
    $lines.Add("[backends]")

    $backendMap = @{
        "auto" = @{
            host = 1; opencl = 1; level_zero = 1; vulkan_probe = 1; vulkan_status = 1; cuda = 1; rocm = 1; prefer_level_zero_over_opencl = 1
        }
        "cpu-only" = @{
            host = 1; opencl = 0; level_zero = 0; vulkan_probe = 0; vulkan_status = 0; cuda = 0; rocm = 0; prefer_level_zero_over_opencl = 1
        }
        "intel-level-zero" = @{
            host = 1; opencl = 0; level_zero = 1; vulkan_probe = 0; vulkan_status = 1; cuda = 0; rocm = 0; prefer_level_zero_over_opencl = 1
        }
        "vulkan-runtime" = @{
            host = 1; opencl = 0; level_zero = 0; vulkan_probe = 1; vulkan_status = 1; cuda = 0; rocm = 0; prefer_level_zero_over_opencl = 1
        }
        "opencl-fallback" = @{
            host = 1; opencl = 1; level_zero = 0; vulkan_probe = 0; vulkan_status = 1; cuda = 0; rocm = 0; prefer_level_zero_over_opencl = 0
        }
    }

    if (-not $backendMap.ContainsKey($BackendId)) {
        throw "Unknown backend preset: $BackendId"
    }

    foreach ($pair in $backendMap[$BackendId].GetEnumerator() | Sort-Object Name) {
        $lines.Add("$($pair.Name)=$($pair.Value)")
    }

    $lines.Add("")
    $lines.Add("[doctor]")
    $lines.Add("host_only=0")

    Set-Content -Path $ConfigPath -Value $lines -Encoding ascii
}

function Find-EntryById {
    param($Items, [string]$Id)
    foreach ($item in $Items) {
        if ($item.id -eq $Id) {
            return $item
        }
    }
    return $null
}

function Select-BackendOption {
    param(
        $Doctor,
        [string]$RequestedBackend,
        [bool]$UseQuiet
    )

    $options = @($Doctor.recommendations | Where-Object { $_.available })
    if ($options.Count -eq 0) {
        throw "No backend recommendations are available."
    }

    if ($RequestedBackend -ne "interactive") {
        $selected = Find-EntryById -Items $Doctor.recommendations -Id $RequestedBackend
        if ($null -eq $selected) {
            throw "Unknown backend recommendation: $RequestedBackend"
        }
        return $selected
    }

    if ($UseQuiet) {
        return Find-EntryById -Items $Doctor.recommendations -Id $Doctor.recommended_backend_id
    }

    Write-Host ""
    Write-Host "Detected backend recommendations:"
    for ($index = 0; $index -lt $options.Count; ++$index) {
        $item = $options[$index]
        $recommended = if ($item.recommended) { " [recommended]" } else { "" }
        Write-Host ("  {0}. {1}{2}" -f ($index + 1), $item.label, $recommended)
        Write-Host ("     {0}" -f $item.reason)
    }

    $default = Find-EntryById -Items $Doctor.recommendations -Id $Doctor.recommended_backend_id
    $choice = Read-Host ("Select backend recommendation [{0}]" -f $default.id)
    if ([string]::IsNullOrWhiteSpace($choice)) {
        return $default
    }

    if ($choice -match '^\d+$') {
        $position = [int]$choice - 1
        if ($position -ge 0 -and $position -lt $options.Count) {
            return $options[$position]
        }
    }

    $selected = Find-EntryById -Items $Doctor.recommendations -Id $choice
    if ($null -eq $selected) {
        throw "Unknown backend selection: $choice"
    }
    return $selected
}

function Resolve-PrerequisiteSelection {
    param(
        $Doctor,
        [string]$BackendId,
        [string]$RequestedMode,
        [bool]$UseQuiet
    )

    $choices = @($Doctor.prerequisite_choices | Where-Object { $_.available -and $_.id -ne "skip-existing-drivers" })
    $pickRelevant = {
        param([string[]]$Ids)
        return @($choices | Where-Object { ($Ids -contains $_.id) -and $_.recommended } | ForEach-Object { $_.id })
    }
    $recommendedIds = switch ($BackendId) {
        "cpu-only" { @() }
        "intel-level-zero" { & $pickRelevant @("intel-level-zero-runtime") }
        "vulkan-runtime" { & $pickRelevant @("vulkan-support") }
        "opencl-fallback" { & $pickRelevant @("opencl-runtime") }
        default { @($choices | Where-Object { $_.recommended } | ForEach-Object { $_.id }) }
    }
    if ($RequestedMode -eq "none") {
        return @()
    }
    if ($RequestedMode -eq "recommended" -or $UseQuiet) {
        return @($choices | Where-Object { $recommendedIds -contains $_.id })
    }
    if ($RequestedMode -eq "all") {
        return $choices
    }

    Write-Host ""
    Write-Host "Optional prerequisite actions:"
    Write-Host "  0. Skip and use existing drivers"
    for ($index = 0; $index -lt $choices.Count; ++$index) {
        $item = $choices[$index]
        $recommended = if ($item.recommended) { " [recommended]" } else { "" }
        Write-Host ("  {0}. {1}{2}" -f ($index + 1), $item.label, $recommended)
        Write-Host ("     {0}" -f $item.reason)
    }
    $selection = Read-Host "Select prerequisite actions (comma separated numbers, empty = recommended)"
    if ([string]::IsNullOrWhiteSpace($selection)) {
        return @($choices | Where-Object { $recommendedIds -contains $_.id })
    }
    if ($selection.Trim() -eq "0") {
        return @()
    }

    $selected = @()
    foreach ($token in ($selection -split ",")) {
        $trimmed = $token.Trim()
        if ($trimmed -notmatch '^\d+$') {
            continue
        }
        $position = [int]$trimmed - 1
        if ($position -ge 0 -and $position -lt $choices.Count) {
            $selected += $choices[$position]
        }
    }
    return $selected
}

function Invoke-PrerequisiteAction {
    param(
        [string]$InstallerAssetRootPath,
        [string]$UpdateScriptPath,
        $ManifestEntry,
        [bool]$UseQuiet
    )

    if ($ManifestEntry.id -eq "skip-existing-drivers") {
        Write-Host "Skipping prerequisite installers and keeping existing drivers."
        return
    }

    $bundledTools = @()
    foreach ($pattern in @($ManifestEntry.runtime_tool_globs)) {
        $fullPattern = Join-Path $InstallerAssetRootPath $pattern
        $bundledTools += @(Get-ChildItem -Path $fullPattern -File -ErrorAction SilentlyContinue)
    }

    $installerPath = $null
    foreach ($pattern in @($ManifestEntry.local_installer_globs)) {
        $fullPattern = Join-Path $InstallerAssetRootPath $pattern
        $matches = @(Get-ChildItem -Path $fullPattern -File -ErrorAction SilentlyContinue)
        if ($matches.Count -gt 0) {
            $installerPath = $matches[0].FullName
            break
        }
    }

    if ($null -ne $installerPath) {
        Write-Host ("Running prerequisite installer for {0}: {1}" -f $ManifestEntry.label, $installerPath)
        & $UpdateScriptPath -InstallerPath $installerPath -Quiet:$UseQuiet
        if ($LASTEXITCODE -ne 0) {
            throw "Prerequisite installer failed for $($ManifestEntry.id)"
        }
        return
    }

    Write-Host ("No bundled installer found for {0}." -f $ManifestEntry.label)
    if ($bundledTools.Count -gt 0) {
        Write-Host ("Bundled runtime tools detected for {0}:" -f $ManifestEntry.label)
        foreach ($tool in $bundledTools) {
            Write-Host ("  Tool asset: {0}" -f $tool.FullName)
        }
        Write-Host "The Jakal runtime will probe these tool locations automatically after install."
    }
    foreach ($url in @($ManifestEntry.support_urls)) {
        Write-Host ("  Support URL: {0}" -f $url)
        if (-not $UseQuiet) {
            Start-Process $url | Out-Null
        }
    }
}

function Show-DoctorSummary {
    param($Doctor)

    Write-Host ""
    Write-Host "Detected devices:"
    foreach ($device in @($Doctor.devices)) {
        Write-Host ("  - [{0}] {1}" -f $device.probe, $device.name)
    }

    Write-Host ""
    Write-Host "Backend statuses:"
    foreach ($backend in @($Doctor.backends)) {
        Write-Host ("  - {0}: {1}" -f $backend.backend_name, $backend.code)
    }

    Write-Host ""
    Write-Host ("Recommended backend: {0}" -f $Doctor.recommended_backend_id)
}

$resolvedInstallRoot = Resolve-JakalInstallRoot -Candidate $InstallRoot
$cliPath = Join-Path $resolvedInstallRoot "bin\\jakal_core_cli.exe"
$manifestPath = Join-Path $PSScriptRoot "jakal-prerequisites.json"
$installerAssetRoot = $PSScriptRoot
$updateScript = Join-Path $resolvedInstallRoot "share\\jakal-core\\update\\update-jakal-core.ps1"

if (-not [string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    $env:JAKAL_RUNTIME_HOME = $RuntimeRoot
}

if (-not (Test-Path $cliPath)) {
    throw "Could not find jakal_core_cli.exe under $resolvedInstallRoot"
}
if (-not (Test-Path $manifestPath)) {
    throw "Could not find prerequisite manifest at $manifestPath"
}
if (-not (Test-Path $updateScript)) {
    throw "Could not find update helper at $updateScript"
}

$manifest = Get-Content -Path $manifestPath -Raw | ConvertFrom-Json
$doctor = Get-DoctorReport -CliPath $cliPath -InstallRootPath $resolvedInstallRoot

Show-DoctorSummary -Doctor $doctor

$selectedBackend = Select-BackendOption -Doctor $doctor -RequestedBackend $Backend -UseQuiet:$Quiet
$selectedPrerequisites = Resolve-PrerequisiteSelection -Doctor $doctor -BackendId $selectedBackend.id -RequestedMode $Prerequisites -UseQuiet:$Quiet

Write-Host ""
Write-Host ("Selected backend preset: {0}" -f $selectedBackend.id)

foreach ($choice in @($selectedPrerequisites)) {
    $manifestEntry = Find-EntryById -Items $manifest.prerequisites -Id $choice.id
    if ($null -eq $manifestEntry) {
        Write-Host ("Skipping unknown prerequisite manifest entry: {0}" -f $choice.id)
        continue
    }
    Invoke-PrerequisiteAction -InstallerAssetRootPath $installerAssetRoot -UpdateScriptPath $updateScript -ManifestEntry $manifestEntry -UseQuiet:$Quiet
}

$configPath = Join-Path $doctor.paths.config_dir "jakal-runtime-config.ini"
Write-BackendPresetConfig -ConfigPath $configPath -BackendId $selectedBackend.id
Write-Host ("Wrote runtime config preset to {0}" -f $configPath)

Write-Host ""
Write-Host "Post-install doctor:"
& $cliPath doctor --runtime-root $resolvedInstallRoot
if ($LASTEXITCODE -ne 0) {
    throw "Post-install doctor reported an error."
}
