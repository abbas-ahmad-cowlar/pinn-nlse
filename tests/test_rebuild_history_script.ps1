$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scriptPath = Join-Path $repoRoot "scripts/rebuild-local-history.ps1"

function Assert-True {
    param(
        [bool] $Condition,
        [string] $Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

Assert-True (Test-Path -LiteralPath $scriptPath) "Expected script to exist: $scriptPath"

$tokens = $null
$parseErrors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    $scriptPath,
    [ref] $tokens,
    [ref] $parseErrors
) | Out-Null
Assert-True ($parseErrors.Count -eq 0) ("PowerShell parser errors: " + ($parseErrors | Out-String))

$dryRunOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File $scriptPath `
    -RepoRoot $repoRoot `
    -DryRun `
    -MinimumCommits 120 `
    -Seed 1234 2>&1
Assert-True ($LASTEXITCODE -eq 0) ("Dry run failed:`n" + ($dryRunOutput | Out-String))
$dryRunText = $dryRunOutput | Out-String
Assert-True ($dryRunText -match "DRY RUN") "Dry run output should identify itself"
Assert-True ($dryRunText -match "Planned chunk commits") "Dry run should report chunk commits"
Assert-True ($dryRunText -match "Planned branch merges") "Dry run should report merge commits"
Assert-True ($dryRunText -match "Total planned commits") "Dry run should report total commits"
Assert-True ($dryRunText -match "Timeline start: 2025-12-17") "Default timeline should start on 2025-12-17"
Assert-True ($dryRunText -match "Timeline end:\s+2026-02-08") "Default timeline should end on 2026-02-08"

$refusalOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File $scriptPath `
    -RepoRoot $repoRoot `
    -MinimumCommits 120 `
    -Seed 1234 2>&1
Assert-True ($LASTEXITCODE -ne 0) "Script must refuse destructive execution without -Force"
$refusalText = $refusalOutput | Out-String
Assert-True ($refusalText -match "Refusing to rebuild") "Refusal output should explain that -Force is required"

Write-Host "rebuild-local-history script tests passed"