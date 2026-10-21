$ErrorActionPreference = "Stop"
$Port = 8765

function Say([string]$Text, [string]$Color = "Cyan") {
  Write-Host $Text -ForegroundColor $Color
}

$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
$ownerIds = @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)

if ($ownerIds.Count -eq 0) {
  Say "后台未运行：127.0.0.1:$Port 没有被占用。" "Yellow"
  exit 0
}

foreach ($ownerId in $ownerIds) {
  $process = Get-Process -Id $ownerId -ErrorAction SilentlyContinue
  if (-not $process) { continue }

  $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $ownerId" -ErrorAction SilentlyContinue).CommandLine
  if ($commandLine -and $commandLine -notmatch "uvicorn") {
    Say "端口 $Port 被其他程序占用（PID $ownerId），未关闭：" "Red"
    Say $commandLine "DarkGray"
    exit 1
  }

  Stop-Process -Id $ownerId -Force
  Say "已关闭后台：$($process.ProcessName) (PID $ownerId)" "Green"
}

$deadline = (Get-Date).AddSeconds(3)
while ((Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) {
  Start-Sleep -Milliseconds 200
}

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
  Say "端口 $Port 仍被占用，请以管理员身份重新运行 stop.bat。" "Red"
  exit 1
}

Say "端口 $Port 已释放，两个窗口会显示连接断开。" "DarkGray"
