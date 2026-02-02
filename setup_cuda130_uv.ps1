<#
.SYNOPSIS
    Sets up a Python virtual environment for GPU testing with PyTorch and CUDA 13.0 support using UV package manager.

.DESCRIPTION
    This PowerShell script automates the setup of a Python virtual environment by:
    - Checking and installing UV package manager if needed
    - Using Python 3.13 from C:\Users\ranga\Python313\python.exe
    - Creating a new virtual environment named 'venv_uv'
    - Installing PyTorch, torchvision, torchaudio with CUDA 13.0 support
    - Installing packages from requirements.txt (except torch, torchvision, torchaudio)
    - Verifying PyTorch and CUDA installations

.PREREQUISITES
    - Python 3.13 installed at C:\Users\ranga\Python313\python.exe
    - Python 3.14 installed at C:\Users\ranga\Python314\python.exe (for UV installation)
    - requirements.txt file in the current directory
    - PowerShell execution policy that allows running scripts
    - NVIDIA driver version 600+ (check with: nvidia-smi)

.NOTES
    Author: [Ranga Seshadri]
    Last Updated: [2025-01-18]
    Version: 1.0
    CUDA Version: 13.0
#>

# Set Python paths
$python313_path = "C:\Users\ranga\Python313\python.exe"
$python314_path = "C:\Users\ranga\Python314\python.exe"
$venv_name = "venv_uv"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup CUDA 13.0 Environment with UV" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check if Python 3.13 exists
Write-Host "Checking Python 3.13..." -ForegroundColor Yellow
if (-not (Test-Path $python313_path)) {
    Write-Host "Error: Python 3.13 not found at $python313_path!" -ForegroundColor Red
    exit 1
}
Write-Host "  Python 3.13 found" -ForegroundColor Green
Write-Host ""

# 2. Check if Python 3.14 exists (for UV installation)
Write-Host "Checking Python 3.14 (for UV installation)..." -ForegroundColor Yellow
if (-not (Test-Path $python314_path)) {
    Write-Host "Warning: Python 3.14 not found at $python314_path!" -ForegroundColor Yellow
    Write-Host "  UV installation may fail if not already installed." -ForegroundColor Yellow
}
Write-Host ""

# 3. Check if UV is already installed
Write-Host "Checking UV installation..." -ForegroundColor Yellow
$uvInstalled = Get-Command uv -ErrorAction SilentlyContinue
if ($uvInstalled) {
    Write-Host "  UV is already installed, skipping installation." -ForegroundColor Green
    $uvVersion = & uv --version
    Write-Host "  UV version: $uvVersion" -ForegroundColor Gray
} else {
    Write-Host "  UV not found. Installing UV..." -ForegroundColor Yellow
    if (-not (Test-Path $python314_path)) {
        Write-Host "Error: Cannot install UV - Python 3.14 not found at $python314_path!" -ForegroundColor Red
        exit 1
    }
    & $python314_path -m pip install uv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to install UV!" -ForegroundColor Red
        exit 1
    }
    Write-Host "  UV installed successfully." -ForegroundColor Green
}
Write-Host ""

# 4. Check if requirements.txt exists
Write-Host "Checking requirements.txt..." -ForegroundColor Yellow
if (-not (Test-Path "requirements.txt")) {
    Write-Host "Error: requirements.txt not found in current directory!" -ForegroundColor Red
    exit 1
}
Write-Host "  requirements.txt found" -ForegroundColor Green
Write-Host ""

# 5. Handle existing venv_uv
if (Test-Path $venv_name) {
    Write-Host "Existing virtual environment '$venv_name' found." -ForegroundColor Yellow
    $confirmation = Read-Host "Remove and recreate? (yes/no)"
    if ($confirmation -eq "yes" -or $confirmation -eq "y") {
        Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venv_name
        Write-Host "  Removed successfully." -ForegroundColor Green
    } else {
        Write-Host "Operation cancelled." -ForegroundColor Yellow
        exit 0
    }
    Write-Host ""
}

# 6. Create new virtual environment with UV
Write-Host "Creating new virtual environment: $venv_name ..." -ForegroundColor Yellow
& uv venv $venv_name --python $python313_path
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to create virtual environment!" -ForegroundColor Red
    exit 1
}
Write-Host "  Virtual environment created successfully." -ForegroundColor Green
Write-Host ""

# 7. Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
# Dot-source the activation script to ensure prompt is modified in current scope
. "$venv_name\Scripts\Activate.ps1"

# Verify activation by checking VIRTUAL_ENV environment variable
if ($env:VIRTUAL_ENV) {
    Write-Host "  Virtual environment activated successfully." -ForegroundColor Green
    Write-Host "  VIRTUAL_ENV: $env:VIRTUAL_ENV" -ForegroundColor Gray
    Write-Host "  Note: To see '($venv_name)' in your prompt, activate manually in your terminal:" -ForegroundColor Cyan
    Write-Host "    .\$venv_name\Scripts\Activate.ps1" -ForegroundColor White
} else {
    Write-Host "  Warning: Virtual environment activation may not have worked correctly." -ForegroundColor Yellow
}
Write-Host ""

# 8. Install PyTorch, torchvision, torchaudio with CUDA 13.0 support
Write-Host "Installing PyTorch with CUDA 13.0 support..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor Gray
& uv pip install torch>=2.9.0 torchvision>=0.24.0 torchaudio>=2.9.0 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install PyTorch with CUDA 13.0 support!" -ForegroundColor Red
    Write-Host "  Note: If GPU is not detected, update NVIDIA driver to version 600+" -ForegroundColor Yellow
    exit 1
}
Write-Host "  PyTorch installed successfully." -ForegroundColor Green
Write-Host ""

# 9. Verify CUDA installation
Write-Host "Verifying CUDA installation..." -ForegroundColor Yellow
if (Test-Path "check_cuda.py") {
    Write-Host "  Running check_cuda.py..." -ForegroundColor Gray
    & "$venv_name\Scripts\python.exe" check_cuda.py
} else {
    Write-Host "  Running inline CUDA verification..." -ForegroundColor Gray
    & "$venv_name\Scripts\python.exe" -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
}
Write-Host ""

# 10. Install all requirements except torch/torchvision/torchaudio
Write-Host "Installing packages from requirements.txt (excluding torch, torchvision, torchaudio)..." -ForegroundColor Yellow
$reqs = Get-Content requirements.txt | Where-Object { 
    $_ -notmatch '^torch' -and 
    $_ -notmatch '^torchvision' -and 
    $_ -notmatch '^torchaudio' -and 
    $_ -notmatch '^#' -and 
    $_.Trim() -ne '' 
}

if ($reqs.Count -gt 0) {
    $reqs | Set-Content temp_requirements.txt
    & uv pip install -r temp_requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Some packages from requirements.txt may have failed to install." -ForegroundColor Yellow
    } else {
        Write-Host "  All requirements installed successfully." -ForegroundColor Green
    }
    Remove-Item temp_requirements.txt
} else {
    Write-Host "  No additional requirements to install (all were torch modules)." -ForegroundColor Gray
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment and see '($venv_name)' in your prompt:" -ForegroundColor Cyan
Write-Host "  .\$venv_name\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "  After activation, your prompt will show: ($venv_name) PS C:\path>" -ForegroundColor Gray
Write-Host ""
Write-Host "To verify CUDA again:" -ForegroundColor Cyan
if (Test-Path "check_cuda.py") {
    Write-Host "  python check_cuda.py" -ForegroundColor White
} else {
    Write-Host "  python -c `"import torch; print('CUDA available:', torch.cuda.is_available())`"" -ForegroundColor White
}
Write-Host ""
