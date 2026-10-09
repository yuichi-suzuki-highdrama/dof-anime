<#
 DoF Anime プラグインのビルドスクリプト (Windows / VS2022 BuildTools / Ninja)

 使い方:
   .\plugin\build.ps1                      # Release ビルド
   .\plugin\build.ps1 -Install             # ビルド後に AE プラグインフォルダへコピー (管理者権限が必要)
   .\plugin\build.ps1 -SdkRoot C:\dev\AE_SDK

 深度の自動生成は Relight Anime と同じ仕組み (relight_depth.exe / 深度サーバー) を使い、
 設定 (%APPDATA%\SRLM\RelightAnime.ini) とキャッシュも共有する。DoF Anime だけ変えたい項目は
 %APPDATA%\SRLM\DoFAnime.ini に同じ形 (key=value) で書く。
#>
param(
  [string]$SdkRoot = $(if ($env:AE_SDK_ROOT) { $env:AE_SDK_ROOT } else { "C:\dev\AE_SDK" }),
  [string]$Config = "Release",
  [switch]$Install,
  [switch]$Clean
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pluginDir = Join-Path $root "plugin"
$buildDir = Join-Path $pluginDir "build"

# 展開したフォルダの中にもう 1 段 (AfterEffectsSDK_xx_win など) あれば、その中を使う
if ((Test-Path $SdkRoot) -and -not (Test-Path (Join-Path $SdkRoot "Examples\Headers")) -and -not (Test-Path (Join-Path $SdkRoot "Headers"))) {
  $sub = Get-ChildItem -Path $SdkRoot -Directory | Where-Object { Test-Path (Join-Path $_.FullName "Examples\Headers") } | Select-Object -First 1
  if ($sub) { $SdkRoot = $sub.FullName }
}
if (-not (Test-Path $SdkRoot)) {
  Write-Host "AE SDK が見つかりません: $SdkRoot" -ForegroundColor Red
  Write-Host "https://developer.adobe.com/after-effects/ から SDK を入手し、展開先を -SdkRoot で指定してください"
  exit 1
}

# MSVC 環境 (vcvars64) を取り込む
$vsBase = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022"
$vcvars = Get-ChildItem -Path $vsBase -Recurse -Filter vcvars64.bat -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $vcvars) { Write-Host "vcvars64.bat が見つかりません (VS2022 BuildTools の C++ ワークロードが必要)" -ForegroundColor Red; exit 1 }
$envDump = cmd /c "`"$($vcvars.FullName)`" >nul 2>&1 && set"
foreach ($line in $envDump) {
  if ($line -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process") }
}

# CMake / Ninja の所在
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
  $pipCmake = python -c "import cmake,os;print(cmake.CMAKE_BIN_DIR)" 2>$null
  if ($pipCmake -and (Test-Path $pipCmake)) { $env:PATH = "$pipCmake;$env:PATH" } else { Write-Host "cmake がありません (pip install cmake)" -ForegroundColor Red; exit 1 }
}
$ninja = Get-Command ninja -ErrorAction SilentlyContinue
if (-not $ninja) {
  $n = Get-ChildItem -Path $vsBase -Recurse -Filter ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($n) { $env:PATH = "$($n.DirectoryName);$env:PATH" } else { Write-Host "ninja がありません" -ForegroundColor Red; exit 1 }
}

if ($Clean -and (Test-Path $buildDir)) { Remove-Item -Recurse -Force $buildDir }

cmake -S $pluginDir -B $buildDir -G Ninja "-DCMAKE_BUILD_TYPE=$Config" "-DAE_SDK_ROOT=$SdkRoot"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build $buildDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$aex = Join-Path $buildDir "DoFAnime.aex"
Write-Host "ビルド完了: $aex" -ForegroundColor Green

if (-not (Test-Path (Join-Path $env:APPDATA "SRLM\RelightAnime.ini"))) {
  Write-Host "注意: %APPDATA%\SRLM\RelightAnime.ini がありません。深度の自動生成には Relight Anime の深度推定 (relight_depth.exe) の設定が必要です" -ForegroundColor Yellow
}

if ($Install) {
  $dest = "C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\SRLM"
  try {
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    # エフェクト本体と補助 AEGP (アイドルフック担当) の 2 つを入れる
    Copy-Item $aex, (Join-Path $buildDir "DoFAnimeHelper.aex") -Destination $dest -Force
    Write-Host "インストール完了: $dest (AE を再起動してください)" -ForegroundColor Green
  } catch {
    Write-Host "コピーに失敗しました。管理者権限の PowerShell で再実行してください: $_" -ForegroundColor Yellow
  }
}
