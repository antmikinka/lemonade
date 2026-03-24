@echo off
REM Build script for KittenTTS standalone server (pre-built binary)
REM This script builds the kitten-tts-server binary with ONNX Runtime bundled
REM for distribution via HuggingFace (like Kokoro)

setlocal enabledelayedexpansion

echo ========================================
echo KittenTTS Server Build Script (Windows)
echo ========================================

REM Configuration
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR..
set BUILD_DIR=%PROJECT_ROOT%\build-kitten-tts
set INSTALL_DIR=%BUILD_DIR%\install
set ONNXRUNTIME_VERSION=1.18.0
set ESPEAK_NG_VERSION=1.50

set PLATFORM_NAME=windows
set ARCH_NAME=x86_64

set OUTPUT_NAME=kitten-tts-server-%PLATFORM_NAME%-%ARCH_NAME%
echo Platform: %PLATFORM_NAME%
echo Architecture: %ARCH_NAME%
echo Output: %OUTPUT_NAME%
echo.

REM Create build directory
echo Creating build directory...
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

REM Download and set up ONNX Runtime (static library)
echo.
echo Setting up ONNX Runtime...
set ONNXRUNTIME_DIR=%BUILD_DIR%\onnxruntime
if not exist "%ONNXRUNTIME_DIR%" (
    echo Downloading ONNX Runtime v%ONNXRUNTIME_VERSION%...
    cd /d "%BUILD_DIR%"

    set ONNXRUNTIME_URL=https://github.com/microsoft/onnxruntime/releases/download/v%ONNXRUNTIME_VERSION%/onnxruntime-win-x64-%ONNXRUNTIME_VERSION%.zip

    REM Download using PowerShell
    powershell -Command "Invoke-WebRequest -Uri '%ONNXRUNTIME_URL%' -OutFile 'onnxruntime.zip'"

    REM Extract using PowerShell
    powershell -Command "Expand-Archive -Path 'onnxruntime.zip' -DestinationPath '.' -Force"

    for /d %%i in (onnxruntime-*) do ren "%%i" onnxruntime
    del onnxruntime.zip
) else (
    echo ONNX Runtime already downloaded at %ONNXRUNTIME_DIR%
)

REM Download espeak-ng (for phonemization)
echo.
echo Setting up espeak-ng...
set ESPEAK_DIR=%BUILD_DIR%\espeak-ng
if not exist "%ESPEAK_DIR%" (
    echo Downloading espeak-ng v%ESPEAK_NG_VERSION%...
    cd /d "%BUILD_DIR%"

    set ESPEAK_URL=https://github.com/espeak-ng/espeak-ng/archive/refs/tags/%ESPEAK_NG_VERSION%.zip
    powershell -Command "Invoke-WebRequest -Uri '%ESPEAK_URL%' -OutFile 'espeak-ng.zip'"
    powershell -Command "Expand-Archive -Path 'espeak-ng.zip' -DestinationPath '.' -Force"
    ren espeak-ng-%ESPEAK_NG_VERSION% espeak-ng
    del espeak-ng.zip

    REM Build espeak-ng using CMake
    cd "%ESPEAK_DIR%"
    echo Building espeak-ng...
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="%ESPEAK_DIR%\install" ^
             -DCMAKE_BUILD_TYPE=Release ^
             -DBUILD_SHARED_LIBS=OFF
    cmake --build . --config Release --target install
) else (
    echo espeak-ng already built at %ESPEAK_DIR%
)

REM Build kitten-tts-server using standalone CMakeLists.txt
echo.
echo Building kitten-tts-server...
cd /d "%PROJECT_ROOT%\src\cpp\server\kitten-tts-server"

if not exist "%BUILD_DIR%\kitten-tts-build" mkdir "%BUILD_DIR%\kitten-tts-build"
cd /d "%BUILD_DIR%\kitten-tts-build"

REM Set up CMake with ONNX Runtime and espeak-ng
cmake "%PROJECT_ROOT%\src\cpp\server\kitten-tts-server" ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
    -DONNXRUNTIME_DIR="%ONNXRUNTIME_DIR%" ^
    -DEspeakNG_INCLUDE_DIR="%ESPEAK_DIR%\install\include" ^
    -DEspeakNG_LIBRARY="%ESPEAK_DIR%\install\lib\espeak-ng.lib"

REM Build the executable
cmake --build . --config Release

REM Copy the binary to install directory
echo.
echo Installing kitten-tts-server...
if not exist "%INSTALL_DIR%\bin" mkdir "%INSTALL_DIR%\bin"
copy /Y "%BUILD_DIR%\kitten-tts-build\Release\kitten-tts-server.exe" "%INSTALL_DIR%\bin\"

REM Copy ONNX Runtime DLLs
echo Copying ONNX Runtime DLLs...
if exist "%ONNXRUNTIME_DIR%\lib\onnxruntime.dll" (
    copy /Y "%ONNXRUNTIME_DIR%\lib\onnxruntime.dll" "%INSTALL_DIR%\bin\"
)
if exist "%ONNXRUNTIME_DIR%\lib\onnxruntime.lib" (
    copy /Y "%ONNXRUNTIME_DIR%\lib\onnxruntime.lib" "%INSTALL_DIR%\bin\"
)

REM Copy espeak-ng data
echo Copying espeak-ng data...
if not exist "%INSTALL_DIR%\share\espeak-ng-data" mkdir "%INSTALL_DIR%\share\espeak-ng-data"
xcopy /E /I /Y "%ESPEAK_DIR%\install\share\espeak-ng-data\*" "%INSTALL_DIR%\share\espeak-ng-data\"

REM Create distribution package
echo.
echo Creating distribution package...
cd /d "%INSTALL_DIR%"

REM Use PowerShell to create tar.gz
powershell -Command "$files = Get-ChildItem -Path '.' -Recurse; Add-Type -AssemblyName System.IO.Compression.FileSystem; $compression = [System.IO.Compression.CompressionLevel]::Optimal; $archive = [System.IO.Compression.ZipFile]::Open('%PROJECT_ROOT%\%OUTPUT_NAME%.zip', 'Create'); foreach ($file in $files) { [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $file.FullName, $file.FullName, $compression) | Out-Null }; $archive.Dispose()"

echo.
echo ========================================
echo Build complete!
echo Output: %PROJECT_ROOT%\%OUTPUT_NAME%.zip
echo ========================================

endlocal
