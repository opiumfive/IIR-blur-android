cmake_minimum_required(VERSION 3.6.0)

set(CMAKE_CXX_FLAGS "-std=c++14 -DANDROID -g")
set(CMAKE_C_FLAGS "-w -std=c11 -DANDROID -D_LARGEFILE_SOURCE=1 -g")
set(CMAKE_ASM_FLAGS "${CFLAGS} -x assembler-with-cpp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")

if (${ANDROID_ABI} STREQUAL "armeabi-v7a" OR ${ANDROID_ABI} STREQUAL "arm64-v8a")
    enable_language(ASM)
else()
    enable_language(ASM_NASM)
endif()

add_library(
        native-lib
        SHARED
        IIRBlur.cpp
        IIRBlurNeon.cpp
        )

target_compile_options(native-lib PUBLIC
        -ffast-math -Os -funroll-loops -ffast-math -fno-strict-aliasing -fno-math-errno)

target_link_options(native-lib PRIVATE -pthread)

if (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    set_target_properties(native-lib PROPERTIES
            ANDROID_ARM_NEON FALSE
            ANDROID_ARM_MODE arm)
    target_compile_definitions(native-lib PUBLIC
            ANDROID_ARM_NEON=false)
else()
    set_target_properties(native-lib PROPERTIES
            ANDROID_ARM_NEON FALSE
            ANDROID_ARM_MODE arm)
endif()

target_link_libraries(
        native-lib
        log
        android
        jnigraphics
        cpufeatures
)

include(AndroidNdkModules)
android_ndk_import_module_cpufeatures()