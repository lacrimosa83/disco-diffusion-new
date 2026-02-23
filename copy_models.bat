@echo off
echo ========================================
echo Disco Diffusion - Model Setup Script
echo ========================================
echo.

echo Checking for model files...
echo.

set "SOURCE_DIR=C:\D disk\Disco Difussion\content"
set "DEST_DIR=%~dp0content"

if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

REM Copy main diffusion model (2.2 GB)
if exist "%SOURCE_DIR%\512x512_diffusion_uncond_finetune_008100.pt" (
    echo Copying 512x512_diffusion_uncond_finetune_008100.pt ...
    copy /Y "%SOURCE_DIR%\512x512_diffusion_uncond_finetune_008100.pt" "%DEST_DIR%\"
) else (
    echo ERROR: Model file not found in source directory!
    echo Please ensure the original Disco Difussion folder exists.
    pause
    exit /b 1
)

REM Copy secondary model (55 MB)
if exist "%SOURCE_DIR%\secondary_model_imagenet_2.pth" (
    echo Copying secondary_model_imagenet_2.pth ...
    copy /Y "%SOURCE_DIR%\secondary_model_imagenet_2.pth" "%DEST_DIR%\"
)

echo.
echo Model files copied successfully!
echo.
pause
