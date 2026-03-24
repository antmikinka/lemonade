#!/bin/bash
# Build script for KittenTTS standalone server (pre-built binary)
# This script builds the kitten-tts-server binary with ONNX Runtime bundled
# for distribution via HuggingFace (like Kokoro)

set -e  # Exit on error

echo "========================================"
echo "KittenTTS Server Build Script"
echo "========================================"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build-kitten-tts"
INSTALL_DIR="$BUILD_DIR/install"
ONNXRUNTIME_VERSION="1.18.0"
ESPEAK_NG_VERSION="1.50"

# Platform detection
PLATFORM=$(uname -s)
ARCH=$(uname -m)

case "$PLATFORM" in
    Linux)
        PLATFORM_NAME="linux"
        ;;
    Darwin)
        PLATFORM_NAME="mac"
        ;;
    *)
        echo "Error: Unsupported platform: $PLATFORM"
        exit 1
        ;;
esac

case "$ARCH" in
    x86_64|amd64)
        ARCH_NAME="x86_64"
        ;;
    aarch64|arm64)
        ARCH_NAME="arm64"
        ;;
    *)
        echo "Error: Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

OUTPUT_NAME="kitten-tts-server-${PLATFORM_NAME}-${ARCH_NAME}"
echo "Platform: $PLATFORM_NAME"
echo "Architecture: $ARCH_NAME"
echo "Output: $OUTPUT_NAME"
echo ""

# Create build directory
echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_DIR"

# Download and build ONNX Runtime (static library)
echo ""
echo "Setting up ONNX Runtime..."
ONNXRUNTIME_DIR="$BUILD_DIR/onnxruntime"
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    echo "Downloading ONNX Runtime v${ONNXRUNTIME_VERSION}..."
    cd "$BUILD_DIR"

    # Download pre-built ONNX Runtime static library
    ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
    if [ "$PLATFORM_NAME" = "mac" ]; then
        ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-macos-arm64-${ONNXRUNTIME_VERSION}.tgz"
        if [ "$ARCH_NAME" = "x86_64" ]; then
            ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-macos-x86_64-${ONNXRUNTIME_VERSION}.tgz"
        fi
    fi

    if command -v curl &> /dev/null; then
        curl -L -o onnxruntime.tar.gz "$ONNXRUNTIME_URL"
    else
        wget -O onnxruntime.tar.gz "$ONNXRUNTIME_URL"
    fi

    tar -xzf onnxruntime.tar.gz
    mv onnxruntime-* onnxruntime
    rm onnxruntime.tar.gz
else
    echo "ONNX Runtime already downloaded at $ONNXRUNTIME_DIR"
fi

# Download espeak-ng (for phonemization)
echo ""
echo "Setting up espeak-ng..."
ESPEAK_DIR="$BUILD_DIR/espeak-ng"
if [ ! -d "$ESPEAK_DIR" ]; then
    echo "Downloading espeak-ng v${ESPEAK_NG_VERSION}..."
    cd "$BUILD_DIR"

    ESPEAK_URL="https://github.com/espeak-ng/espeak-ng/archive/refs/tags/${ESPEAK_NG_VERSION}.tar.gz"
    if command -v curl &> /dev/null; then
        curl -L -o espeak-ng.tar.gz "$ESPEAK_URL"
    else
        wget -O espeak-ng.tar.gz "$ESPEAK_URL"
    fi

    tar -xzf espeak-ng.tar.gz
    mv espeak-ng-${ESPEAK_NG_VERSION} espeak-ng
    rm espeak-ng.tar.gz

    # Build espeak-ng
    cd "$ESPEAK_DIR/src"
    echo "Building espeak-ng..."
    ./autogen.sh
    ./configure --prefix="$ESPEAK_DIR/install" --disable-espeakng-libshared
    make -j$(nproc)
    make install
else
    echo "espeak-ng already built at $ESPEAK_DIR"
fi

# Build kitten-tts-server
echo ""
echo "Building kitten-tts-server..."
cd "$PROJECT_ROOT"

mkdir -p "$BUILD_DIR/kitten-tts-build"
cd "$BUILD_DIR/kitten-tts-build"

# Set up CMake with ONNX Runtime and espeak-ng
cmake "$PROJECT_ROOT/src/cpp/server/kitten-tts-server" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR" \
    -DEspeakNG_INCLUDE_DIR="$ESPEAK_DIR/install/include" \
    -DEspeakNG_LIBRARY="$ESPEAK_DIR/install/lib/libespeak-ng.a"

# Build the executable
cmake --build . --config Release -j$(nproc)

# Copy the binary to install directory
echo ""
echo "Installing kitten-tts-server..."
mkdir -p "$INSTALL_DIR/bin"
cp "$BUILD_DIR/kitten-tts-build/Release/kitten-tts-server" "$INSTALL_DIR/bin/" 2>/dev/null || \
cp "$BUILD_DIR/kitten-tts-build/kitten-tts-server" "$INSTALL_DIR/bin/"

# Copy ONNX Runtime library
echo "Copying ONNX Runtime library..."
cp "$ONNXRUNTIME_DIR"/lib/*.so* "$INSTALL_DIR/bin/" 2>/dev/null || \
cp "$ONNXRUNTIME_DIR"/lib/*.dylib* "$INSTALL_DIR/bin/" 2>/dev/null || true

# Copy espeak-ng data
echo "Copying espeak-ng data..."
mkdir -p "$INSTALL_DIR/share/espeak-ng-data"
cp -r "$ESPEAK_DIR/install/share/espeak-ng-data/"* "$INSTALL_DIR/share/espeak-ng-data/" 2>/dev/null || true

# Create distribution package
echo ""
echo "Creating distribution package..."
cd "$INSTALL_DIR"
tar -czf "$PROJECT_ROOT/${OUTPUT_NAME}.tar.gz" bin share

echo ""
echo "========================================"
echo "Build complete!"
echo "Output: $PROJECT_ROOT/${OUTPUT_NAME}.tar.gz"
echo "========================================"
