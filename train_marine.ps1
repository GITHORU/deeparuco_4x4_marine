# Script pour lancer les entraînements DeepArUco marine
# Lancer chaque commande dans un terminal séparé (ou une après l'autre)

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("detector", "corners", "decoder")]
    [string]$Model
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Activer le venv
& .\venv\Scripts\Activate.ps1

switch ($Model) {
    "detector" {
        Write-Host "=== Train 1/3 : Détecteur YOLO ===" -ForegroundColor Cyan
        python train_detector.py marine_detection marine_detector
    }
    "corners" {
        Write-Host "=== Train 2/3 : Corners (heatmaps) ===" -ForegroundColor Cyan
        python train_corners.py marine_regression marine_corners -m
    }
    "decoder" {
        Write-Host "=== Train 3/3 : Decoder ===" -ForegroundColor Cyan
        python train_decoder.py marine_regression marine_decoder
    }
}
