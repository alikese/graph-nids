param(
    [string[]]$Days = @('Thursday', 'Friday'),
    [int]$WindowSeconds = 10
)

$ErrorActionPreference = 'Stop'

if ($WindowSeconds -le 0) {
    throw 'WindowSeconds must be greater than 0.'
}

$inputDir = Join-Path $PSScriptRoot 'data'
$outputDir = Join-Path $PSScriptRoot 'Windows divide'

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

foreach ($day in $Days) {
    $inputPath = Join-Path $inputDir "$day-ip.csv"

    if (-not (Test-Path $inputPath)) {
        throw "Input file not found: $inputPath"
    }

    Write-Host "Splitting $day from $inputPath"

    $reader = [System.IO.StreamReader]::new($inputPath, [System.Text.Encoding]::UTF8, $true, 1048576)
    $writer = $null
    $currentWindowStart = $null
    $windowsWritten = 0

    try {
        $header = $reader.ReadLine()
        if ([string]::IsNullOrWhiteSpace($header)) {
            throw "Input CSV is empty: $inputPath"
        }

        $columns = $header.Split(',')
        $timestampIndex = [Array]::IndexOf($columns, 'timestamp')
        if ($timestampIndex -lt 0) {
            throw "CSV must contain a 'timestamp' column: $inputPath"
        }

        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }

            $parts = $line.Split(',')
            if ($parts.Length -le $timestampIndex) {
                continue
            }

            $timestamp = [double]::Parse($parts[$timestampIndex], [System.Globalization.CultureInfo]::InvariantCulture)
            $windowStart = [int64]([math]::Floor($timestamp / $WindowSeconds) * $WindowSeconds)
            $windowEnd = $windowStart + $WindowSeconds

            if ($null -ne $currentWindowStart -and $windowStart -lt $currentWindowStart) {
                throw "Timestamps must be sorted by ascending window in $inputPath."
            }

            if ($windowStart -ne $currentWindowStart) {
                if ($writer) {
                    $writer.Close()
                    $writer.Dispose()
                }

                $fileName = "$day-ip_${windowStart}_${windowEnd}.csv"
                $outPath = Join-Path $outputDir $fileName
                $writer = [System.IO.StreamWriter]::new($outPath, $false, [System.Text.Encoding]::UTF8, 1048576)
                $writer.WriteLine($header)
                $currentWindowStart = $windowStart
                $windowsWritten++
            }

            $writer.WriteLine($line)
        }

        Write-Host "Finished ${day}: $windowsWritten windows"
    }
    finally {
        if ($reader) {
            $reader.Close()
            $reader.Dispose()
        }

        if ($writer) {
            $writer.Close()
            $writer.Dispose()
        }
    }
}
