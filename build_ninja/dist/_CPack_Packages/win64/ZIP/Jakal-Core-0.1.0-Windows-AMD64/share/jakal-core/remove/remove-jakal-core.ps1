param(
    [string]$InstallRoot = "",
    [switch]$ForceRemove
)

if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $InstallRoot = Split-Path -Path $PSScriptRoot -Parent
    $InstallRoot = Split-Path -Path $InstallRoot -Parent
}

$nsisUninstaller = Join-Path $InstallRoot "Uninstall.exe"

if (Test-Path $nsisUninstaller) {
    Start-Process -FilePath $nsisUninstaller -Wait
    exit 0
}

if (-not $ForceRemove) {
    throw "No packaged uninstaller found under $InstallRoot. Re-run with -ForceRemove to remove the install tree directly."
}

Remove-Item -Path $InstallRoot -Recurse -Force
