@echo off
REM ============================================================
REM  setup.bat  —  Windows 10/11, Python 3.10/3.11, CUDA 12.x
REM ============================================================
echo.
echo ============================================================
echo   Lenta Price Tag Detector — Установка
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERR] Python не найден!
    echo       Установите Python 3.10 или 3.11 с python.org
    echo       При установке поставьте галочку "Add Python to PATH"
    pause & exit /b 1
)
echo [OK] Python:
python --version
echo.

if exist .venv (
    echo [OK] .venv уже существует
) else (
    echo [1/4] Создание виртуального окружения...
    python -m venv .venv
    if errorlevel 1 ( echo [ERR] Не удалось создать venv & pause & exit /b 1 )
    echo [OK] .venv создан
)
echo.

call .venv\Scripts\activate.bat
echo [OK] venv активирован
echo.

echo [2/4] Обновление pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip обновлён
echo.

echo [3/4] Установка PyTorch с CUDA 12.1 (~2 GB, подождите)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 ( echo [ERR] Ошибка PyTorch & pause & exit /b 1 )
echo [OK] PyTorch установлен
echo.

echo Проверка GPU:
python -c "import torch; c=torch.cuda.is_available(); print(f'  CUDA: {c}'); print(f'  GPU:  {torch.cuda.get_device_name(0) if c else \"N/A\"}')"
echo.

echo [4/4] Установка зависимостей...
pip install -r requirements.txt
if errorlevel 1 ( echo [ERR] Ошибка зависимостей & pause & exit /b 1 )
echo.

echo ============================================================
echo   Установка завершена!
echo.
echo   Далее:
echo     .venv\Scripts\activate
echo     python prepare.py
echo     python train.py
echo     python infer.py
echo ============================================================
pause
