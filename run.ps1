param(
  [string]$Python = "",
  [switch]$Rebuild,
  [switch]$SwapScreens
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Frontend = Join-Path $Root "frontend"
$Backend = Join-Path $Root "backend"
$Port = 8765
$Base = "http://127.0.0.1:$Port"

function Say([string]$Text, [string]$Color = "Cyan") {
  Write-Host $Text -ForegroundColor $Color
}

function Test-Health {
  try {
    $result = Invoke-RestMethod "$Base/api/health" -TimeoutSec 2
    return [bool]$result.ok
  } catch {
    return $false
  }
}

function Resolve-Python {
  if ($Python) {
    if (Test-Path $Python) { return (Resolve-Path $Python).Path }
    throw "指定的 Python 不存在: $Python"
  }
  if ($env:YOLO_PY -and (Test-Path $env:YOLO_PY)) {
    return (Resolve-Path $env:YOLO_PY).Path
  }

  $patterns = @(
    (Join-Path $env:USERPROFILE ".conda\envs\*\python.exe"),
    (Join-Path $env:USERPROFILE "miniconda3\envs\*\python.exe"),
    (Join-Path $env:USERPROFILE "anaconda3\envs\*\python.exe"),
    (Join-Path $env:LOCALAPPDATA "miniconda3\envs\*\python.exe"),
    (Join-Path $env:LOCALAPPDATA "anaconda3\envs\*\python.exe"),
    (Join-Path $env:USERPROFILE "miniforge3\envs\*\python.exe")
  )
  $candidates = @()
  foreach ($pattern in $patterns) {
    $candidates += Get-ChildItem $pattern -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
  }
  $candidates = $candidates | Sort-Object @{ Expression = { if ($_ -match "yolo") { 0 } else { 1 } } }, @{ Expression = { $_ } }
  $command = Get-Command python -ErrorAction SilentlyContinue
  if ($command) { $candidates += $command.Source }
  $candidates = $candidates | Select-Object -Unique

  foreach ($candidate in $candidates) {
    & $candidate -c "import torch, ultralytics, fastapi, uvicorn, onnxruntime, cv2" 2>$null
    if ($LASTEXITCODE -eq 0) { return $candidate }
  }
  throw "没有找到可用的 YOLO Python 环境。请用 -Python 指定解释器路径，例如：.\run.ps1 -Python C:\Users\you\.conda\envs\yolo26\python.exe"
}

function Ensure-Frontend {
  $dist = Join-Path $Frontend "dist\index.html"
  $needBuild = $Rebuild -or -not (Test-Path $dist)
  if (-not $needBuild) {
    $distTime = (Get-Item $dist).LastWriteTimeUtc
    $srcFiles = Get-ChildItem (Join-Path $Frontend "src") -Recurse -File
    if ($srcFiles | Where-Object { $_.LastWriteTimeUtc -gt $distTime } | Select-Object -First 1) {
      $needBuild = $true
    }
    foreach ($name in @("index.html", "package.json", "vite.config.ts", "tsconfig.json")) {
      $file = Join-Path $Frontend $name
      if ((Test-Path $file) -and (Get-Item $file).LastWriteTimeUtc -gt $distTime) { $needBuild = $true }
    }
  }
  if (-not $needBuild) {
    Say "前端已是最新构建"
    return
  }
  $npm = Get-Command npm -ErrorAction SilentlyContinue
  if (-not $npm) { throw "未找到 npm，无法构建前端。请先安装 Node.js。" }
  Push-Location $Frontend
  try {
    if (-not (Test-Path (Join-Path $Frontend "node_modules"))) {
      Say "安装前端依赖 (npm install)..."
      & npm install
      if ($LASTEXITCODE -ne 0) { throw "npm install 失败" }
    }
    Say "构建前端 (npm run build)..."
    & npm run build
    if ($LASTEXITCODE -ne 0) { throw "前端构建失败" }
  } finally {
    Pop-Location
  }
}

function Ensure-Backend([string]$PythonPath) {
  if (Test-Health) {
    Say "后端已在 $Base 运行" "Green"
    return
  }
  $logDir = Join-Path $Root "state"
  New-Item -ItemType Directory -Force -Path $logDir | Out-Null
  $outLog = Join-Path $logDir "backend.log"
  $errLog = Join-Path $logDir "backend.err.log"
  Say "启动后端并加载模型（首次约需 10-40 秒）..."
  Start-Process -FilePath $PythonPath `
    -ArgumentList @("-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "$Port", "--log-level", "warning") `
    -WorkingDirectory $Backend `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -WindowStyle Hidden | Out-Null

  $deadline = (Get-Date).AddSeconds(180)
  while (-not (Test-Health)) {
    if ((Get-Date) -gt $deadline) {
      throw "后端启动超时。请查看 state\backend.err.log"
    }
    Start-Sleep -Milliseconds 700
  }
  Say "后端就绪：$Base" "Green"
}

function Get-WindowRect([System.Windows.Forms.Screen]$Screen, [int]$X, [int]$Y, [int]$Width, [int]$Height) {
  return [pscustomobject]@{ X = $X; Y = $Y; Width = $Width; Height = $Height }
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class YoloWin32 {
  [DllImport("user32.dll")]
  public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);
  [DllImport("user32.dll")]
  public static extern bool SetForegroundWindow(IntPtr hWnd);
}
"@

function Find-AppWindow([string]$Needle) {
  return Get-Process -Name msedge, chrome -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -like "*$Needle*" } |
    Select-Object -First 1
}

function Place-Window([string]$Needle, $Rect) {
  $process = Find-AppWindow $Needle
  if (-not $process) { return $false }
  [YoloWin32]::MoveWindow($process.MainWindowHandle, $Rect.X, $Rect.Y, $Rect.Width, $Rect.Height, $true) | Out-Null
  return $true
}

function Wait-AppWindow([string]$Needle, [int]$TimeoutMs = 10000) {
  $deadline = (Get-Date).AddMilliseconds($TimeoutMs)
  while ((Get-Date) -lt $deadline) {
    $process = Find-AppWindow $Needle
    if ($process) { return $process }
    Start-Sleep -Milliseconds 300
  }
  return $null
}

function Open-AppWindow([string]$Browser, [string]$Url, $Rect, [string]$ProfileDir, [string]$Needle) {
  if (Place-Window $Needle $Rect) {
    Say "复用已打开的窗口：$Needle"
    return $true
  }
  New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null
  $arguments = @(
    "--app=$Url",
    "--user-data-dir=$ProfileDir",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-session-crashed-bubble",
    "--window-size=$($Rect.Width),$($Rect.Height)",
    "--window-position=$($Rect.X),$($Rect.Y)"
  )
  for ($attempt = 1; $attempt -le 3; $attempt++) {
    Start-Process -FilePath $Browser -ArgumentList $arguments | Out-Null
    if (Wait-AppWindow $Needle 9000) { return $true }
    # 上一个窗口刚关闭时浏览器还在退出，新启动会被吞掉；等一会再试。
    Start-Sleep -Seconds 2
  }
  return $false
}

function Find-Browser {
  $paths = @(
    (Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe"),
    (Join-Path $env:ProgramFiles "Microsoft\Edge\Application\msedge.exe"),
    (Join-Path $env:LOCALAPPDATA "Google\Chrome\Application\chrome.exe"),
    (Join-Path $env:ProgramFiles "Google\Chrome\Application\chrome.exe")
  )
  foreach ($path in $paths) {
    if ($path -and (Test-Path $path)) { return $path }
  }
  return $null
}

try {
  Say "YOLO26 神经显微镜 · 双屏启动器" "White"
  Say "项目目录：$Root" "DarkGray"

  $pythonPath = Resolve-Python
  Say "Python：$pythonPath" "DarkGray"

  Ensure-Frontend
  Ensure-Backend $pythonPath

  $screens = @([System.Windows.Forms.Screen]::AllScreens)
  $primary = [System.Windows.Forms.Screen]::PrimaryScreen
  $others = @($screens | Where-Object { $_.DeviceName -ne $primary.DeviceName } | Sort-Object { $_.Bounds.Width * $_.Bounds.Height } -Descending)
  if ($SwapScreens -and $others.Count -gt 0) {
    $secondary = $primary
    $primary = $others[0]
    $others = @($screens | Where-Object { $_.DeviceName -ne $primary.DeviceName })
  } else {
    $secondary = $others | Select-Object -First 1
  }

  if ($secondary) {
    $liveRect = Get-WindowRect $primary $primary.WorkingArea.X $primary.WorkingArea.Y $primary.WorkingArea.Width $primary.WorkingArea.Height
    $neuralRect = Get-WindowRect $secondary $secondary.WorkingArea.X $secondary.WorkingArea.Y $secondary.WorkingArea.Width $secondary.WorkingArea.Height
    Say "检测到 $($screens.Count) 块屏幕：左屏 -> $($primary.DeviceName)，右屏 -> $($secondary.DeviceName)"
  } else {
    $area = $primary.WorkingArea
    $offset = 44
    $liveRect = Get-WindowRect $primary $area.X $area.Y $area.Width $area.Height
    $neuralRect = Get-WindowRect $primary ($area.X + $offset) ($area.Y + $offset) ($area.Width - $offset) ($area.Height - $offset)
    Say "只检测到一块屏幕：两个窗口全尺寸叠放，可用 Alt+Tab 切换；接上外接显示器后可把右屏拖过去。" "Yellow"
  }

  $browser = Find-Browser
  $profileRoot = Join-Path $env:LOCALAPPDATA "Yolo26Scope"
  if ($browser) {
    $liveOk = Open-AppWindow $browser "$Base/live" $liveRect (Join-Path $profileRoot "window-live") "观察者"
    $neuralOk = Open-AppWindow $browser "$Base/neural" $neuralRect (Join-Path $profileRoot "window-neural") "神经显微镜"
    Start-Sleep -Seconds 1
    Place-Window "观察者" $liveRect | Out-Null
    Place-Window "神经显微镜" $neuralRect | Out-Null
    if ($liveOk -and $neuralOk) {
      Say "已打开两个独立窗口：左屏观察者 / 右屏神经显微镜" "Green"
    } else {
      $missing = @()
      if (-not $liveOk) { $missing += "观察者" }
      if (-not $neuralOk) { $missing += "神经显微镜" }
      Say ("未能打开：{0}。通常是旧窗口还没完全退出，请关掉残留窗口后再运行一次。" -f ($missing -join " / ")) "Yellow"
    }
  } else {
    Say "未找到 Edge 或 Chrome，改用默认浏览器打开两个标签页。" "Yellow"
    Start-Process "$Base/live"
    Start-Process "$Base/neural"
  }

  Say ""
  Say "左屏： http://127.0.0.1:$Port/live" "DarkGray"
  Say "右屏： http://127.0.0.1:$Port/neural" "DarkGray"
  Say "后端日志： state\backend.log" "DarkGray"
  Say "启动器可以关闭；后端会继续在后台运行。" "DarkGray"
} catch {
  Write-Host ""
  Write-Host "启动失败：$($_.Exception.Message)" -ForegroundColor Red
  exit 1
}
