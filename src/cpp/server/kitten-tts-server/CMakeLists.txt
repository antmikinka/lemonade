cmake_minimum_required(VERSION 3.10)
project(kitten-tts-server VERSION 1.0.0)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows SDK version
if(WIN32)
    add_compile_definitions(_WIN32_WINNT=0x0A00)
endif()

# Find dependencies
find_package(nlohmann_json REQUIRED)
find_package(httplib REQUIRED)
find_package(onnxruntime REQUIRED)

# espeak-ng - try pkg-config first, then manual find
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ESPEAK_NG QUIET espeak-ng)
endif()
if(NOT ESPEAK_NG_FOUND)
    find_path(ESPEAK_NG_INCLUDE_DIR NAMES espeak-ng.h)
    find_library(ESPEAK_NG_LIBRARY NAMES espeak-ng)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/../../include
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${ESPEAK_NG_INCLUDE_DIRS}
)

# Source files
set(KITTEN_TTS_SOURCES
    main.cpp
    ../backends/kitten_tts_native.cpp
    ../backends/kitten_tts_onnx_model.cpp
    ../backends/kitten_tts_phonemizer.cpp
    ../backends/kitten_tts_voice.cpp
    ../backends/kitten_tts_preprocessor.cpp
    ../utils/npz_reader.cpp
)

# Create executable
add_executable(kitten-tts-server ${KITTEN_TTS_SOURCES})

# Link dependencies
target_link_libraries(kitten-tts-server PRIVATE
    nlohmann_json::nlohmann_json
    httplib::httplib
    onnxruntime
)

if(ESPEAK_NG_FOUND)
    target_link_libraries(kitten-tts-server PRIVATE ${ESPEAK_NG_LIBRARIES})
    target_include_directories(kitten-tts-server PRIVATE ${ESPEAK_NG_INCLUDE_DIRS})
else()
    target_link_libraries(kitten-tts-server PRIVATE espeak-ng)
endif()

# Platform-specific linking
if(WIN32)
    target_link_libraries(kitten-tts-server PRIVATE
        ws2_32
        wsock32
    )
elseif(APPLE)
    target_link_libraries(kitten-tts-server PRIVATE
        "-framework Foundation"
        "-framework CoreServices"
    )
endif()

# Install target
install(TARGETS kitten-tts-server RUNTIME DESTINATION bin)
