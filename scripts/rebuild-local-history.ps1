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