param(
    [Parameter(Mandatory = $true)]
    [string]$InstallerPath,
    [string]$TargetDir = "",
    [switch]$Quiet
)

$resolvedInstaller = Resolve-Path -Path $InstallerPath -ErrorAction Stop
$extension = [System.IO.Path]::GetExtension($resolvedInstaller.Path).ToLowerInvariant()

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
