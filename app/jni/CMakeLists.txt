cmake_minimum_required(VERSION 3.10)
project(iirblur LANGUAGES CXX)

add_library(native-lib SHARED
        IIRBlur.cpp
        IIRBlurNeon.cpp
        IIRBlurFast.cpp
        IIRBlurNeonFp16.cpp
        BoxBlur.cpp)
target_compile_features(native-lib PRIVATE cxx_std_14)
target_compile_options(native-lib PRIVATE
        -O3 -ffp-contract=fast -ffunction-sections -fdata-sections)
# Keep IEEE semantics: fast-math breaks finite-input checks and the stable recurrence.
target_link_libraries(native-lib PRIVATE log android jnigraphics)
