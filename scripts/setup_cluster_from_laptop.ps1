# Wires the cluster to the repo, from Windows, in one go. Idempotent.
#
#   .\scripts\setup_cluster_from_laptop.ps1 user@your-login-node
#
# Same thing as the .sh of the same name, for machines without bash on the path.
# Windows ships an OpenSSH client, so ssh and scp are there; gh does the rest.
#
# What it does:
#   1. copies bootstrap_cluster.sh over
#   2. runs it, which generates a deploy key there and prints the public half
#   3. registers that half on the repo through the GitHub API
#   4. runs it again, which clones the repo and checks that pushing works

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Target,

    [string]$Repo = $(if ($env:SCENPORT_REPO) { $env:SCENPORT_REPO } else { "ralorin/scenport" })
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "gh is not installed"
}
gh auth status | Out-Null
if (-not $?) {
    Write-Error "gh is not logged in; run 'gh auth login'"
}
foreach ($tool in @("ssh", "scp")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Error "$tool not found. Install the Windows OpenSSH client from Optional Features."
    }
}

Write-Host "cluster : $Target"
Write-Host "repo    : $Repo"
Write-Host ""

# The cluster runs the script with sh, so it has to arrive with Unix line endings.
# A CRLF checkout produces 'bad interpreter' or a silent '\r' in every path.
$local = "scripts/bootstrap_cluster.sh"
$bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $local))
if ($bytes -contains 13) {
    $temp = Join-Path $env:TEMP "bootstrap_cluster.sh"
    $text = [System.IO.File]::ReadAllText((Resolve-Path $local)) -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($temp, $text, (New-Object System.Text.UTF8Encoding $false))
    $local = $temp
    Write-Host "converted line endings for transfer"
}

Write-Host "== copying the bootstrap script =="
scp -q $local "${Target}:~/bootstrap_cluster.sh"
if ($LASTEXITCODE -ne 0) { Write-Error "scp failed" }

Write-Host "== first run: generate the key =="
$first = ssh $Target 'bash ~/bootstrap_cluster.sh'
$first | ForEach-Object { "    $_" }

$pubkey = $first | Where-Object { $_ -match '^ssh-ed25519 ' } | Select-Object -First 1

if (-not $pubkey) {
    if ($first -match "deploy key works") {
        Write-Host ""
        Write-Host "already wired up, nothing to register."
        exit 0
    }
    Write-Error "could not find a public key in the output above"
}

Write-Host ""
Write-Host "== registering the deploy key =="

$title = ($Target -replace '[^A-Za-z0-9._@-]', '') + " " + (Get-Date -Format "yyyy-MM-dd")
$existing = gh api "repos/$Repo/keys" --jq '.[].key'
$body = ($pubkey -split ' ')[0..1] -join ' '

if ($existing -and ($existing | Where-Object { $_.StartsWith($body) })) {
    Write-Host "    already registered"
} else {
    # No jq formatting here. PowerShell takes the quotes off the expression before
    # gh sees it, so a jq string literal arrives as bare tokens and gh fails to
    # parse it. That failure came after the POST had already gone through, so the
    # key was registered and the script reported it as an error anyway.
    $out = gh api "repos/$Repo/keys" -f "title=$title" -f "key=$pubkey" -F read_only=false
    if ($LASTEXITCODE -eq 0) {
        $id = ($out | ConvertFrom-Json).id
        Write-Host "    added key id $id, write access"
    } else {
        # Re-read rather than trust the exit code: GitHub rejects a key that is
        # already on the repo, and that is a success for our purposes.
        $now = gh api "repos/$Repo/keys" --jq '.[].key'
        if ($now -and ($now | Where-Object { $_.StartsWith($body) })) {
            Write-Host "    already present on the repo"
        } else {
            Write-Error "could not register the deploy key: $out"
        }
    }
}

Write-Host ""
Write-Host "== second run: clone and verify =="
ssh $Target 'bash ~/bootstrap_cluster.sh' | ForEach-Object { "    $_" }

Write-Host ""
Write-Host "Deploy keys for this repo: https://github.com/$Repo/settings/keys"
Write-Host "Revoke there if the cluster is ever rebuilt again."
