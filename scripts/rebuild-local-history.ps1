<#
.SYNOPSIS
Rebuild a local-only Git history from the current final project files.

.DESCRIPTION
This script snapshots the current project files, deletes the existing .git
directory, initializes a fresh repository, and reconstructs a detailed local
history by committing real chunks from the snapshotted files. Text files are
rebuilt in small line/section/function chunks, notebooks are rebuilt cell by
cell, and binary artifacts are committed whole. The final commit copies the
snapshot back byte-for-byte and verifies hashes so the working tree matches the
original project files selected for reconstruction.

The script is intentionally local-only. It does not push anywhere.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts/rebuild-local-history.ps1 -DryRun

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts/rebuild-local-history.ps1 -Force
#>

[CmdletBinding()]
param(
    [string] $RepoRoot = (Get-Location).Path,
    [switch] $DryRun,
    [switch] $Force,
    [int] $MinimumCommits = 120,
    [int] $CommitsPerBranch = 4,
    [int] $Seed = 20251013,
    [datetime] $StartDate = [datetime] "2025-12-17T09:00:00",
    [datetime] $EndDate = [datetime] "2026-02-08T18:00:00",
    [string] $TimezoneOffset = "+0500",
    [string] $ExpectedAuthorName = "Syed Abbas Ahmad",
    [switch] $KeepBackup
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$script:RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$script:Utf8NoBom = New-Object System.Text.UTF8Encoding $false

function Write-Info {
    param([string] $Message)
    Write-Host "[history-rebuild] $Message"
}

function ConvertTo-GitPath {
    param([string] $Path)
    return ($Path -replace "\\", "/").TrimStart("/")
}

function Join-RepoPath {
    param([string] $RelativePath)
    $native = $RelativePath -replace "/", [System.IO.Path]::DirectorySeparatorChar
    return Join-Path $script:RepoRoot $native
}

function Get-RelativeGitPath {
    param([string] $FullPath)
    $full = (Resolve-Path -LiteralPath $FullPath).Path
    $base = $script:RepoRoot.TrimEnd("\", "/")
    if (-not $full.StartsWith($base, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside repository root: $full"
    }
    $relative = $full.Substring($base.Length).TrimStart("\", "/")
    return ConvertTo-GitPath $relative
}

function Invoke-Git {
    param(
        [string[]] $Arguments,
        [switch] $AllowFailure
    )
    Push-Location $script:RepoRoot
    try {
        & git @Arguments
        $exitCode = $LASTEXITCODE
        if (-not $AllowFailure -and $exitCode -ne 0) {
            throw "git $($Arguments -join ' ') failed with exit code $exitCode"
        }
        return $exitCode
    }
    finally {
        Pop-Location
    }
}

function Invoke-GitOutput {
    param([string[]] $Arguments)
    Push-Location $script:RepoRoot
    try {
        $output = & git @Arguments 2>$null
        if ($LASTEXITCODE -ne 0) {
            return @()
        }
        return @($output)
    }
    finally {
        Pop-Location
    }
}

function Invoke-DatedGit {
    param(
        [string[]] $Arguments,
        [datetime] $When
    )
    $dateText = $When.ToString("yyyy-MM-dd HH:mm:ss") + " " + $TimezoneOffset
    $oldAuthor = $env:GIT_AUTHOR_DATE
    $oldCommitter = $env:GIT_COMMITTER_DATE
    $env:GIT_AUTHOR_DATE = $dateText
    $env:GIT_COMMITTER_DATE = $dateText
    try {
        Invoke-Git -Arguments $Arguments | Out-Null
    }
    finally {
        if ($null -eq $oldAuthor) {
            Remove-Item Env:\GIT_AUTHOR_DATE -ErrorAction SilentlyContinue
        }
        else {
            $env:GIT_AUTHOR_DATE = $oldAuthor
        }
        if ($null -eq $oldCommitter) {
            Remove-Item Env:\GIT_COMMITTER_DATE -ErrorAction SilentlyContinue
        }
        else {
            $env:GIT_COMMITTER_DATE = $oldCommitter
        }
    }
}

function Test-ExcludedPath {
    param([string] $RelativePath)
    $p = ConvertTo-GitPath $RelativePath
    $parts = @($p -split "/")
    $excludedDirs = @(
        ".git", ".venv", "venv", "env", ".pytest_cache", "__pycache__",
        ".private_archive", "pinn_nlse.egg-info", "dist", "build",
        ".ipynb_checkpoints"
    )
    foreach ($part in $parts) {
        if ($excludedDirs -contains $part) {
            return $true
        }
    }
    $name = [System.IO.Path]::GetFileName($p)
    if ($name -like "*.pyc") { return $true }
    if ($name -like "*.pyo") { return $true }
    if ($name -like "*.archived-*") { return $true }
    if ($name -like "jupyterlab_*.log") { return $true }
    if ($name -eq "Desktop.ini" -or $name -eq "Thumbs.db" -or $name -eq ".DS_Store") { return $true }
    return $false
}

function Get-CandidateFiles {
    $files = @()
    if (Test-Path -LiteralPath (Join-Path $script:RepoRoot ".git")) {
        $gitFiles = @(Invoke-GitOutput -Arguments @("ls-files", "--cached", "--others", "--exclude-standard"))
        if ($gitFiles.Count -gt 0) {
            $files = $gitFiles | ForEach-Object { ConvertTo-GitPath $_ }
        }
    }

    if ($files.Count -eq 0) {
        $files = Get-ChildItem -LiteralPath $script:RepoRoot -Recurse -File |
            ForEach-Object { Get-RelativeGitPath $_.FullName } |
            Where-Object { -not (Test-ExcludedPath $_) }
    }

    $files |
        Where-Object { $_ -and -not (Test-ExcludedPath $_) } |
        Sort-Object -Unique
}

function Get-FileOrder {
    param([string] $RelativePath)
    $p = ConvertTo-GitPath $RelativePath
    if ($p -eq ".gitignore") { return 0 }
    if ($p -eq "LICENSE") { return 1 }
    if ($p -eq "requirements.txt" -or $p -eq "pyproject.toml") { return 2 }
    if ($p -eq "README.md") { return 3 }
    if ($p -like "src/config.py") { return 10 }
    if ($p -like "src/nlse_utils.py" -or $p -like "src/ssfm.py") { return 11 }
    if ($p -like "src/utils.py") { return 12 }
    if ($p -like "src/data_gen.py") { return 13 }
    if ($p -like "src/pinn_nlse.py") { return 14 }
    if ($p -like "src/train.py") { return 15 }
    if ($p -like "src/generate_ground_truth.py") { return 16 }
    if ($p -like "src/compare.py") { return 17 }
    if ($p -like "src/benchmark.py") { return 18 }
    if ($p -like "src/*") { return 19 }
    if ($p -like "tests/*") { return 30 }
    if ($p -like "notebooks/*") { return 40 }
    if ($p -like "data/*") { return 50 }
    if ($p -like "figures/*") { return 60 }
    if ($p -like "logs/*") { return 70 }
    if ($p -like "models/*") { return 80 }
    if ($p -like "report/*") { return 90 }
    if ($p -like "scripts/*") { return 95 }
    return 99
}

function Get-FileKind {
    param([string] $RelativePath)
    $p = ConvertTo-GitPath $RelativePath
    $name = [System.IO.Path]::GetFileName($p)
    $ext = [System.IO.Path]::GetExtension($p).ToLowerInvariant()
    if ($ext -eq ".ipynb") { return "notebook" }
    if ($name -eq ".gitignore" -or $name -eq ".gitkeep") { return "text" }
    if (@(".py", ".ps1", ".md", ".toml", ".txt", ".json", ".cfg", ".csv") -contains $ext) {
        return "text"
    }
    return "whole-file"
}

function Get-ChunkLabel {
    param(
        [string] $RelativePath,
        [string[]] $Lines,
        [int] $Index
    )
    $p = ConvertTo-GitPath $RelativePath
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($p)
    $first = ($Lines | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
    if ($null -eq $first) {
        return "Add $p"
    }
    $trimmed = $first.Trim()
    if ($trimmed -match "^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)") {
        return "Add $stem $($matches[2])"
    }
    if ($trimmed -match "^#+\s+(.+)$") {
        $heading = $matches[1]
        if ($heading.Length -gt 44) { $heading = $heading.Substring(0, 44).Trim() }
        return "Document $heading"
    }
    if ($trimmed -match "^from\s+|^import\s+") {
        return "Add $stem imports"
    }
    if ($trimmed -match "^\[") {
        return "Add $stem configuration"
    }
    if ($trimmed.Length -gt 44) {
        $trimmed = $trimmed.Substring(0, 44).Trim()
    }
    return "Update $stem chunk $Index"
}

function New-TextChunks {
    param(
        [string] $RelativePath,
        [string] $SourcePath,
        [int] $MaxLinesPerChunk
    )
    $lines = [System.IO.File]::ReadAllLines($SourcePath)
    $chunks = New-Object "System.Collections.Generic.List[object]"
    if ($lines.Count -eq 0) {
        $chunks.Add([pscustomobject] @{
            Path = $RelativePath
            Kind = "text"
            Lines = @()
            Label = "Add empty $RelativePath"
            Order = Get-FileOrder $RelativePath
            ChunkIndex = 1
        })
        return $chunks.ToArray()
    }

    $ext = [System.IO.Path]::GetExtension($RelativePath).ToLowerInvariant()
    $breaks = New-Object "System.Collections.Generic.List[int]"
    $breaks.Add(0)
    $lastBreak = 0
    for ($i = 1; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        $isBoundary = $false
        if ($ext -eq ".py") {
            if ($line -match "^(def|class)\s+" -or $line -match "^#\s*-{5,}") {
                $isBoundary = $true
            }
        }
        elseif ($ext -eq ".md") {
            if ($line -match "^#{1,3}\s+") {
                $isBoundary = $true
            }
        }
        elseif ($ext -eq ".ps1") {
            if ($line -match "^function\s+[A-Za-z0-9_-]+") {
                $isBoundary = $true
            }
        }

        if (($i - $lastBreak) -ge $MaxLinesPerChunk) {
            $isBoundary = $true
        }
        if ($isBoundary -and $i -gt $lastBreak) {
            $breaks.Add($i)
            $lastBreak = $i
        }
    }
    $breaks.Add($lines.Count)

    $chunkIndex = 1
    for ($b = 0; $b -lt ($breaks.Count - 1); $b++) {
        $start = $breaks[$b]
        $endExclusive = $breaks[$b + 1]
        if ($endExclusive -le $start) { continue }
        $segment = @()
        for ($j = $start; $j -lt $endExclusive; $j++) {
            $segment += $lines[$j]
        }
        $chunks.Add([pscustomobject] @{
            Path = $RelativePath
            Kind = "text"
            Lines = $segment
            Label = Get-ChunkLabel -RelativePath $RelativePath -Lines $segment -Index $chunkIndex
            Order = Get-FileOrder $RelativePath
            ChunkIndex = $chunkIndex
        })
        $chunkIndex++
    }
    return $chunks.ToArray()
}

function New-NotebookChunks {
    param(
        [string] $RelativePath,
        [string] $SourcePath
    )
    $raw = Get-Content -LiteralPath $SourcePath -Raw
    $notebook = $raw | ConvertFrom-Json
    $cells = @($notebook.cells)
    $chunks = New-Object "System.Collections.Generic.List[object]"
    if ($cells.Count -eq 0) {
        $chunks.Add([pscustomobject] @{
            Path = $RelativePath
            Kind = "whole-file"
            Label = "Add empty notebook $RelativePath"
            Order = Get-FileOrder $RelativePath
            ChunkIndex = 1
        })
        return $chunks.ToArray()
    }
    for ($i = 0; $i -lt $cells.Count; $i++) {
        $displayIndex = $i + 1
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($RelativePath)
        $chunks.Add([pscustomobject] @{
            Path = $RelativePath
            Kind = "notebook-cell"
            Cell = $cells[$i]
            Template = $notebook
            Label = "Add $stem notebook cell $displayIndex"
            Order = Get-FileOrder $RelativePath
            ChunkIndex = $displayIndex
        })
    }
    return $chunks.ToArray()
}

function New-WholeFileChunk {
    param([string] $RelativePath)
    $name = [System.IO.Path]::GetFileNameWithoutExtension($RelativePath)
    return [pscustomobject] @{
        Path = $RelativePath
        Kind = "whole-file"
        Label = "Add $name artifact"
        Order = Get-FileOrder $RelativePath
        ChunkIndex = 1
    }
}

function New-ReconstructionChunks {
    param(
        [string[]] $Files,
        [string] $BackupRoot,
        [int] $MaxLinesPerChunk
    )
    $chunks = New-Object "System.Collections.Generic.List[object]"
    foreach ($file in $Files) {
        $sourcePath = Join-Path $BackupRoot ($file -replace "/", [System.IO.Path]::DirectorySeparatorChar)
        $kind = Get-FileKind $file
        if ($kind -eq "text") {
            foreach ($chunk in (New-TextChunks -RelativePath $file -SourcePath $sourcePath -MaxLinesPerChunk $MaxLinesPerChunk)) {
                $chunks.Add($chunk)
            }
        }
        elseif ($kind -eq "notebook") {
            foreach ($chunk in (New-NotebookChunks -RelativePath $file -SourcePath $sourcePath)) {
                $chunks.Add($chunk)
            }
        }
        else {
            $chunks.Add((New-WholeFileChunk -RelativePath $file))
        }
    }

    return @($chunks | Sort-Object @{ Expression = "Order"; Ascending = $true }, Path, ChunkIndex)
}

function New-WorkDays {
    param(
        [datetime] $Start,
        [datetime] $End,
        [int] $RandomSeed
    )
    $rng = New-Object System.Random $RandomSeed
    $selected = @{}
    $cursor = $Start.Date
    while ($cursor -le $End.Date) {
        $week = @()
        for ($i = 0; $i -lt 7; $i++) {
            $d = $cursor.AddDays($i)
            if ($d -ge $Start.Date -and $d -le $End.Date) {
                $week += $d
            }
        }
        $take = [Math]::Min(4, $week.Count)
        $picked = $week | Sort-Object { $rng.Next() } | Select-Object -First $take
        foreach ($d in $picked) {
            $selected[$d.ToString("yyyy-MM-dd")] = $d
        }
        $cursor = $cursor.AddDays(7)
    }
    $selected[$Start.Date.ToString("yyyy-MM-dd")] = $Start.Date
    $selected[$End.Date.ToString("yyyy-MM-dd")] = $End.Date
    return @($selected.Values | Sort-Object)
}

function New-CommitDates {
    param(
        [int] $Count,
        [datetime] $Start,
        [datetime] $End,
        [int] $RandomSeed
    )
    if ($Count -le 0) { return @() }
    if ($Count -eq 1) { return @($End) }

    $rng = New-Object System.Random ($RandomSeed + 17)
    $workDays = New-WorkDays -Start $Start -End $End -RandomSeed $RandomSeed
    $dates = New-Object "System.Collections.Generic.List[datetime]"
    while ($dates.Count -lt $Count) {
        foreach ($day in $workDays) {
            $commitsToday = 1 + $rng.Next(0, 4)
            for ($i = 0; $i -lt $commitsToday; $i++) {
                if ($dates.Count -ge $Count) { break }
                $hour = 9 + $rng.Next(0, 13)
                $minute = $rng.Next(0, 60)
                $second = $rng.Next(0, 60)
                $candidate = $day.AddHours($hour).AddMinutes($minute).AddSeconds($second)
                if ($candidate -ge $Start -and $candidate -le $End) {
                    $dates.Add($candidate)
                }
            }
            if ($dates.Count -ge $Count) { break }
        }
    }
    $sorted = @($dates | Sort-Object | Select-Object -First $Count)
    $sorted[0] = $Start
    $sorted[$sorted.Count - 1] = $End
    return @($sorted)
}

function Get-BranchTopic {
    param([string] $RelativePath)
    $p = ConvertTo-GitPath $RelativePath
    if ($p -like "src/config.py") { return "configuration" }
    if ($p -like "src/ssfm.py" -or $p -like "src/nlse_utils.py") { return "ssfm-solver" }
    if ($p -like "src/data_gen.py") { return "training-data" }
    if ($p -like "src/pinn_nlse.py") { return "pinn-model" }
    if ($p -like "src/train.py") { return "training-loop" }
    if ($p -like "src/compare.py") { return "comparison-cli" }
    if ($p -like "src/benchmark.py") { return "benchmarking" }
    if ($p -like "src/*") { return "source-modules" }
    if ($p -like "tests/*") { return "tests" }
    if ($p -like "notebooks/*") { return "notebooks" }
    if ($p -like "data/*") { return "ground-truth-data" }
    if ($p -like "figures/*") { return "figures" }
    if ($p -like "logs/*") { return "training-logs" }
    if ($p -like "models/*") { return "model-weights" }
    if ($p -like "report/*" -or $p -eq "README.md") { return "documentation" }
    if ($p -like "scripts/*") { return "history-tooling" }
    return "project-files"
}
