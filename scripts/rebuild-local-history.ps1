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
