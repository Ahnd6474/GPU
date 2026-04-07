param(
    [string]$Model = "qwen2.5:0.5b",
    [int]$Passes = 3,
    [switch]$ShowOps
)

$ErrorActionPreference = "Stop"

function Start-SampledRun {
    param(
        [string]$Label,
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    $stdoutPath = Join-Path $env:TEMP ("jakal-" + $Label + "-" + [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() + ".out.txt")
    $stderrPath = Join-Path $env:TEMP ("jakal-" + $Label + "-" + [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() + ".err.txt")

    $proc = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    $cpuSamples = @()
    $gpuSamples = @()
    $maxGpuEngine = @()
    $gpuLocalMemoryMb = @()
    $lastWall = Get-Date
    $lastCpu = 0.0
    try {
        $initial = Get-Process -Id $proc.Id -ErrorAction Stop
        $lastCpu = [double]$initial.CPU
    } catch {
        $lastCpu = 0.0
    }

    while (-not $proc.HasExited) {
        Start-Sleep -Milliseconds 500
        $now = Get-Date
        $intervalSec = [Math]::Max(0.001, ($now - $lastWall).TotalSeconds)
        $lastWall = $now

        $cpuPct = 0.0
        $gpuPct = 0.0
        $gpuEngineMax = 0.0
        $gpuLocalMb = 0.0

        $current = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if ($null -ne $current) {
            $currentCpu = [double]$current.CPU
            $cpuPct = [Math]::Max(0.0, (($currentCpu - $lastCpu) / $intervalSec / [Environment]::ProcessorCount) * 100.0)
            $lastCpu = $currentCpu
        }

        $samples = @()
        try {
            $samples = Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 1 -ErrorAction Stop |
                Select-Object -ExpandProperty CounterSamples |
                Where-Object {
                    $_.InstanceName -match ("pid_" + $proc.Id + "_") -and
                    ($_.Status -eq 0 -or $_.Status -eq 'Success')
                }
        } catch {
            $samples = @()
        }

        foreach ($sample in $samples) {
            $value = [double]$sample.CookedValue
            $gpuPct += $value
            if ($value -gt $gpuEngineMax) {
                $gpuEngineMax = $value
            }
        }

        try {
            $memorySamples = Get-Counter '\GPU Process Memory(*)\Local Usage' -SampleInterval 1 -MaxSamples 1 -ErrorAction Stop |
                Select-Object -ExpandProperty CounterSamples |
                Where-Object {
                    $_.InstanceName -match ("pid_" + $proc.Id + "_") -and
                    ($_.Status -eq 0 -or $_.Status -eq 'Success')
                }

            foreach ($sample in $memorySamples) {
                $gpuLocalMb += ([double]$sample.CookedValue / 1MB)
            }
        } catch {
            $gpuLocalMb = 0.0
        }

        $cpuSamples += $cpuPct
        $gpuSamples += $gpuPct
        $maxGpuEngine += $gpuEngineMax
        $gpuLocalMemoryMb += $gpuLocalMb
        $proc.Refresh()
    }

    $stdout = if (Test-Path $stdoutPath) { Get-Content -Path $stdoutPath -Raw } else { "" }
    $stderr = if (Test-Path $stderrPath) { Get-Content -Path $stderrPath -Raw } else { "" }

    $wallMatch = [regex]::Matches($stdout, 'total_us=([0-9]+(?:\.[0-9]+)?)')
    $totals = @()
    foreach ($match in $wallMatch) {
        $totals += [double]$match.Groups[1].Value
    }

    [pscustomobject]@{
        Label = $Label
        ExitCode = $proc.ExitCode
        AvgCpuPct = if ($cpuSamples.Count -gt 0) { ($cpuSamples | Measure-Object -Average).Average } else { 0.0 }
        MaxCpuPct = if ($cpuSamples.Count -gt 0) { ($cpuSamples | Measure-Object -Maximum).Maximum } else { 0.0 }
        AvgGpuPct = if ($gpuSamples.Count -gt 0) { ($gpuSamples | Measure-Object -Average).Average } else { 0.0 }
        MaxGpuPct = if ($gpuSamples.Count -gt 0) { ($gpuSamples | Measure-Object -Maximum).Maximum } else { 0.0 }
        MaxGpuEnginePct = if ($maxGpuEngine.Count -gt 0) { ($maxGpuEngine | Measure-Object -Maximum).Maximum } else { 0.0 }
        AvgGpuLocalMb = if ($gpuLocalMemoryMb.Count -gt 0) { ($gpuLocalMemoryMb | Measure-Object -Average).Average } else { 0.0 }
        MaxGpuLocalMb = if ($gpuLocalMemoryMb.Count -gt 0) { ($gpuLocalMemoryMb | Measure-Object -Maximum).Maximum } else { 0.0 }
        AvgTotalUs = if ($totals.Count -gt 0) { ($totals | Measure-Object -Average).Average } else { 0.0 }
        RawOutput = $stdout
        RawError = $stderr
    }
}

$root = Split-Path -Parent $PSScriptRoot
$profileExe = Join-Path $root "build\Debug\jakal_profile_manifest.exe"
$directmlExe = Join-Path $root "build\Debug\jakal_directml_manifest_bench.exe"

if (-not (Test-Path $profileExe)) {
    throw "Missing executable: $profileExe"
}
if (-not (Test-Path $directmlExe)) {
    throw "Missing executable: $directmlExe"
}

$runtimeArgs = @("--ollama-model", $Model, "--passes", $Passes.ToString(), "--level-zero-only")
$dmlArgs = @("--ollama-model", $Model, "--passes", $Passes.ToString())
if ($ShowOps) {
    $runtimeArgs += "--show-ops"
    $dmlArgs += "--show-ops"
}

$runtime = Start-SampledRun -Label "runtime-levelzero" -FilePath $profileExe -ArgumentList $runtimeArgs
$directml = Start-SampledRun -Label "directml-standalone" -FilePath $directmlExe -ArgumentList $dmlArgs

"Comparison summary"
[pscustomobject]@{
    Label = $runtime.Label
    ExitCode = $runtime.ExitCode
    AvgTotalUs = [Math]::Round($runtime.AvgTotalUs, 3)
    AvgCpuPct = [Math]::Round($runtime.AvgCpuPct, 2)
    MaxCpuPct = [Math]::Round($runtime.MaxCpuPct, 2)
    AvgGpuPct = [Math]::Round($runtime.AvgGpuPct, 2)
    MaxGpuPct = [Math]::Round($runtime.MaxGpuPct, 2)
    MaxGpuEnginePct = [Math]::Round($runtime.MaxGpuEnginePct, 2)
    AvgGpuLocalMb = [Math]::Round($runtime.AvgGpuLocalMb, 1)
    MaxGpuLocalMb = [Math]::Round($runtime.MaxGpuLocalMb, 1)
} | Format-Table -AutoSize

[pscustomobject]@{
    Label = $directml.Label
    ExitCode = $directml.ExitCode
    AvgTotalUs = [Math]::Round($directml.AvgTotalUs, 3)
    AvgCpuPct = [Math]::Round($directml.AvgCpuPct, 2)
    MaxCpuPct = [Math]::Round($directml.MaxCpuPct, 2)
    AvgGpuPct = [Math]::Round($directml.AvgGpuPct, 2)
    MaxGpuPct = [Math]::Round($directml.MaxGpuPct, 2)
    MaxGpuEnginePct = [Math]::Round($directml.MaxGpuEnginePct, 2)
    AvgGpuLocalMb = [Math]::Round($directml.AvgGpuLocalMb, 1)
    MaxGpuLocalMb = [Math]::Round($directml.MaxGpuLocalMb, 1)
} | Format-Table -AutoSize

""
"Runtime output"
$runtime.RawOutput
if ($runtime.RawError) {
    "Runtime stderr"
    $runtime.RawError
}

""
"DirectML output"
$directml.RawOutput
if ($directml.RawError) {
    "DirectML stderr"
    $directml.RawError
}
