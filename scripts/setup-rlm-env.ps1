<#
Creates (or reuses) a Python virtual environment in the project root, installs
PyTorch (optionally), installs project requirements, and can optionally run a
CUDA sanity check.

Examples:
  .\scripts\setup-rlm-env.ps1
  .\scripts\setup-rlm-env.ps1 -EnvName "venv"
  .\scripts\setup-rlm-env.ps1 -SkipTorchInstall
  .\scripts\setup-rlm-env.ps1 -RunCudaCheck
#>

param(
  # Folder name for the virtual environment (created under the project root).
  [Parameter(Mandatory = $false)]
  [string] $EnvName = "rlm_env",

  # Python executable to use. Can be "python" (from PATH) or a full path to python.exe.
  [Parameter(Mandatory = $false)]
  [string] $PythonExe = "python",

  # Requirements filename located in the project root (e.g. requirements.txt).
  [Parameter(Mandatory = $false)]
  [string] $RequirementsFile = "requirements.txt",

  # pip index URL used to download PyTorch wheels (e.g. CUDA-enabled wheels).
  [Parameter(Mandatory = $false)]
  [string] $TorchIndexUrl = "https://download.pytorch.org/whl/cu130",

  # If passed, skips installing torch/torchvision/torchaudio.
  [Parameter(Mandatory = $false)]
  [switch] $SkipTorchInstall,

  # If passed, runs check_cuda.py after installs finish.
  [Parameter(Mandatory = $false)]
  [switch] $RunCudaCheck
)

# Fail fast: if any command errors, stop the script immediately.
# This prevents half-completed environment setups.
$ErrorActionPreference = "Stop"

# Prints a consistent step header in the terminal for easier reading.
function Write-Step([string] $msg) {
  Write-Host ""
  Write-Host "==> $msg" -ForegroundColor Cyan
}

# Ensures a command exists before we try to run it (helps with clearer errors).
function Require-Command([string] $cmd) {
  $found = Get-Command $cmd -ErrorAction SilentlyContinue
  if (-not $found) {
    throw "Command not found: '$cmd'. Install it or pass -PythonExe with a full path (e.g. C:\Users\ranga\python312\python.exe)."
  }
}

Write-Step "Resolving project root"
# $PSScriptRoot points to this script's directory (...\RLM\scripts).
$scriptDir = $PSScriptRoot
# Project root is assumed to be the parent folder (...\RLM).
$projectRoot = Split-Path -Parent $scriptDir
# Run from project root so relative paths like .\check_cuda.py resolve correctly.
Set-Location $projectRoot
Write-Host "Project root: $projectRoot"

Write-Step "Checking Python"
# Verify the chosen Python command exists and can run.
Require-Command $PythonExe
& $PythonExe --version

# Virtual environment path and its activation script path (Windows venv layout).
$envPath = Join-Path $projectRoot $EnvName
$activatePath = Join-Path $envPath "Scripts\Activate.ps1"

Write-Step "Creating virtual environment ($EnvName) if missing"
# Create the venv only if it doesn't already exist.
if (-not (Test-Path $envPath)) {
  & $PythonExe -m venv $envPath
  Write-Host "Created venv at: $envPath"
} else {
  Write-Host "Venv already exists at: $envPath"
}

# Ensure activation script exists (sanity check: venv created correctly).
if (-not (Test-Path $activatePath)) {
  throw "Activation script not found at: $activatePath"
}

Write-Step "Activating venv"
# Dot-source activation script so it modifies the current PowerShell session.
. $activatePath
# Confirm we're using the venv's Python.
python --version

Write-Step "Upgrading pip"
# Upgrade pip inside the venv to improve install compatibility with wheels.
python -m pip install --upgrade pip

if (-not $SkipTorchInstall) {
  Write-Step "Installing PyTorch (CUDA) from $TorchIndexUrl"
  # Install PyTorch packages. The index URL controls which build (e.g. CUDA) is used.
  pip install "torch>=2.9.0" "torchvision>=0.24.0" "torchaudio>=2.9.0" --index-url $TorchIndexUrl
} else {
  Write-Host "Skipping torch install (per -SkipTorchInstall)."
}

Write-Step "Installing requirements ($RequirementsFile)"
# Resolve requirements path and verify it exists before installing.
$reqPath = Join-Path $projectRoot $RequirementsFile
if (-not (Test-Path $reqPath)) {
  throw "requirements file not found: $reqPath"
}
# Install the rest of the project dependencies.
pip install -r $reqPath

if ($RunCudaCheck) {
  Write-Step "Running CUDA check (check_cuda.py)"
  # Optional sanity check to verify torch sees CUDA/GPU (if available).
  python .\check_cuda.py
}

Write-Step "Done"
# Friendly reminders for the user.
Write-Host "To activate later:" -ForegroundColor Green
Write-Host "  $EnvName\Scripts\Activate.ps1"
Write-Host "To run the CUDA check:" -ForegroundColor Green
Write-Host "  python .\check_cuda.py"

