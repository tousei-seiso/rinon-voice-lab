[CmdletBinding()]
param(
  # いまの内訳を見るだけで、dwm には触らない。
  [switch] $MeasureOnly,
  # 確認プロンプトを省く（毎回の定型作業として回すとき用）。
  [switch] $Force,
  # 一覧に載せる下限(MiB)。小さなプロセスを省いて読みやすくする。
  [int] $MinimumMiB = 100,
  # dwm が戻ってくるまで待つ上限(秒)。
  [int] $RestartTimeoutSeconds = 20
)

# dwm(Desktop Window Manager) が抱える VRAM はログオンしたままだと積み上がり、自力では
# 戻らない。高解像度・多画面・HDR ではコンポジション用サーフェスが FP16(8 バイト/px) に
# なるため 4K 1 枚で 66MiB あり、ウィンドウを開くほど増える。
#
# これが問題になるのは GPU を推論サーバと分け合っているとき。2 つの合計が物理 VRAM を
# 超えると、推論が GPU を飽和させた瞬間に WDDM が dwm のサーフェスをシステムメモリへ
# 追い出す。以後は描画のたびに PCIe 越しで戻す必要があり、画面更新もマウスカーソルも
# 止まる（推論そのものは正常な速度で走っているのに操作できなくなる）。
#
# サインアウトや再起動をせずに返させる手段は「dwm を落として作り直させる」ことだけ。
# winlogon が dwm を見張っていて即座に起動し直すので、セッションと開いているウィンドウは
# そのまま残る。作り直しのあいだ数秒だけ画面が暗転する。
#
# 文字コード: 日本語コメントを含むので UTF-8(BOM 付き)で保存する。BOM が無いと
# PowerShell 5.1 が cp932 として読み、param ブロックごと壊れる（pwsh 7 なら BOM 不要）。

$ErrorActionPreference = 'Stop'

function Test-Administrator {
  $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
  return ([Security.Principal.WindowsPrincipal] $identity).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
  )
}

function Get-GpuTotalUsedMiB {
  # GPU 全体の使用量を nvidia-smi から取る。NVIDIA 以外や未インストールなら $null。
  if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    return $null
  }
  try {
    $line = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits |
      Select-Object -First 1)
    $parts = $line -split ','
    return [pscustomobject]@{
      UsedMiB  = [int][double]$parts[0].Trim()
      TotalMiB = [int][double]$parts[1].Trim()
    }
  } catch {
    return $null
  }
}

function Get-VramBreakdown {
  # プロセス別の専用 VRAM を MiB で返す。GeForce の nvidia-smi は Windows(WDDM) では
  # プロセス別を報告しないため、WDDM のパフォーマンスカウンタから読む。
  # カウンタが取れない環境では空を返し、呼び出し側は全体値だけで進む。
  param([int] $Threshold = 100)

  $samples = $null
  try {
    $samples = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples
  } catch {
    Write-Host "GPU per-process counters are unavailable: $($_.Exception.Message)"
    return @()
  }

  # インスタンス名は pid_<PID>_luid_... の形。1 プロセスが複数へ割れるので束ねる。
  $byProcess = @{}
  foreach ($sample in $samples) {
    if ($sample.InstanceName -notmatch 'pid_(\d+)') { continue }
    $processId = [int] $Matches[1]
    $byProcess[$processId] = [double] ($byProcess[$processId]) + $sample.CookedValue
  }

  $rows = foreach ($entry in $byProcess.GetEnumerator()) {
    $mib = [int] ($entry.Value / 1MB)
    if ($mib -lt $Threshold) { continue }
    $process = Get-Process -Id $entry.Key -ErrorAction SilentlyContinue
    [pscustomobject]@{
      Name      = if ($process) { $process.Name } else { 'gone' }
      ProcessId = $entry.Key
      VramMiB   = $mib
    }
  }
  return @($rows | Sort-Object VramMiB -Descending)
}

function Write-VramReport {
  param(
    [string] $Title,
    [object[]] $Rows,
    [object] $Total,
    [int] $Threshold
  )
  Write-Host ''
  Write-Host "== $Title =="
  if ($Rows.Count -gt 0) {
    $Rows | Format-Table -AutoSize | Out-String | Write-Host -NoNewline
    Write-Host ("per-process total (>= $Threshold MiB): {0} MiB" -f (
      ($Rows | Measure-Object VramMiB -Sum).Sum
    ))
  }
  if ($null -ne $Total) {
    Write-Host ("nvidia-smi: {0} / {1} MiB used" -f $Total.UsedMiB, $Total.TotalMiB)
  }
}

# --- 1) 掃除前の状態を記録する ------------------------------------------------
$before = Get-VramBreakdown -Threshold $MinimumMiB
$beforeTotal = Get-GpuTotalUsedMiB
Write-VramReport -Title 'Before' -Rows $before -Total $beforeTotal -Threshold $MinimumMiB

$dwmBefore = $before | Where-Object { $_.Name -eq 'dwm' } | Select-Object -First 1
if ($dwmBefore) {
  Write-Host ("dwm is holding {0} MiB" -f $dwmBefore.VramMiB)
}

if ($MeasureOnly) {
  Write-Host ''
  Write-Host 'Measure-only mode: nothing was restarted.'
  return
}

# --- 2) 落とせる状態か確かめる ------------------------------------------------
if (-not (Test-Administrator)) {
  Write-Host ''
  Write-Host 'This needs an elevated shell (dwm.exe runs as SYSTEM).'
  Write-Host 'Reopen PowerShell with "Run as administrator", or pass -MeasureOnly to just look.'
  exit 1
}

# 自分と同じセッションの dwm だけを対象にする（他セッションを巻き込まないため）。
$sessionId = (Get-Process -Id $PID).SessionId
$target = Get-Process dwm -ErrorAction SilentlyContinue |
  Where-Object { $_.SessionId -eq $sessionId } |
  Select-Object -First 1
if (-not $target) {
  Write-Host "No dwm process found for session $sessionId."
  exit 1
}

if (-not $Force) {
  Write-Host ''
  Write-Host "About to restart dwm (pid $($target.Id), session $sessionId)."
  Write-Host 'The screen goes black for a few seconds. Windows and apps stay open.'
  $answer = Read-Host 'Continue? [y/N]'
  if ($answer -notmatch '^(y|yes)$') {
    Write-Host 'Cancelled.'
    return
  }
}

# --- 3) dwm を落として作り直させる --------------------------------------------
# Stop-Process は SYSTEM のプロセスに拒否されるので taskkill を通す。
Write-Host "Restarting dwm (pid $($target.Id))..."
$killOutput = & taskkill.exe /F /PID $target.Id 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host "taskkill failed: $killOutput"
  Write-Host 'If access was denied, dwm is protected here. Alternatives without a reboot:'
  Write-Host '  - change the display resolution and change it back (a mode set recreates surfaces)'
  Write-Host '  - lock the session with Win+L, then sign back in'
  exit 1
}

# 復帰を待つ。落とした直後は一時的にプロセスが居らず、カウンタも読めない。
$deadline = (Get-Date).AddSeconds($RestartTimeoutSeconds)
$revived = $null
while ((Get-Date) -lt $deadline) {
  Start-Sleep -Milliseconds 500
  $revived = Get-Process dwm -ErrorAction SilentlyContinue |
    Where-Object { $_.SessionId -eq $sessionId -and $_.Id -ne $target.Id } |
    Select-Object -First 1
  if ($revived) { break }
}
if (-not $revived) {
  Write-Host ''
  Write-Host "dwm did not come back within $RestartTimeoutSeconds s. Sign out and back in to recover."
  exit 1
}
Write-Host "dwm is back (pid $($revived.Id))."

# サーフェスを作り直し終えるまで少し待つ（直後に測ると回復途中の値を拾う）。
Start-Sleep -Seconds 3

# --- 4) 掃除後の状態と差分 ----------------------------------------------------
$after = Get-VramBreakdown -Threshold $MinimumMiB
$afterTotal = Get-GpuTotalUsedMiB
Write-VramReport -Title 'After' -Rows $after -Total $afterTotal -Threshold $MinimumMiB

$dwmAfter = $after | Where-Object { $_.Name -eq 'dwm' } | Select-Object -First 1
$dwmBeforeMiB = if ($dwmBefore) { $dwmBefore.VramMiB } else { 0 }
$dwmAfterMiB = if ($dwmAfter) { $dwmAfter.VramMiB } else { 0 }
Write-Host ''
Write-Host ("dwm: {0} -> {1} MiB (freed {2} MiB)" -f
  $dwmBeforeMiB, $dwmAfterMiB, ($dwmBeforeMiB - $dwmAfterMiB))
if ($null -ne $beforeTotal -and $null -ne $afterTotal) {
  Write-Host ("gpu total: {0} -> {1} MiB (freed {2} MiB)" -f
    $beforeTotal.UsedMiB, $afterTotal.UsedMiB, ($beforeTotal.UsedMiB - $afterTotal.UsedMiB))
}
