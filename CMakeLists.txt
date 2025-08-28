cmake_minimum_required(VERSION 3.20) # 3.30도 OK, 3.20이면 대부분 환경 호환
project(cpp_dl LANGUAGES CXX)

# 표준과 빌드 타입
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo) # 기본값
endif()

# 소스
file(GLOB SRC CONFIGURE_DEPENDS src/*.cpp)
add_library(dl STATIC ${SRC})
target_include_directories(dl PUBLIC include)
target_compile_features(dl PUBLIC cxx_std_20)

# 경고 플래그
if(MSVC)
    target_compile_options(dl PRIVATE /W4)
else()
    target_compile_options(dl PRIVATE -Wall -Wextra -Wpedantic)
endif()

## 디버그 때 sanitizer
#if(CMAKE_BUILD_TYPE MATCHES "Debug")
#    if(NOT MSVC)
#        target_compile_options(dl PRIVATE -fsanitize=address,undefined)
#        target_link_options(dl PRIVATE -fsanitize=address,undefined)
#    endif()
#endif()

# 예제/테스트
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor dl)

add_executable(test_autograd tests/test_autograd.cpp)
target_link_libraries(test_autograd dl)

add_executable(test_nn_linear tests/test_nn_linear.cpp)
target_link_libraries(test_nn_linear dl)

# ctest 연동
include(CTest)
add_test(NAME tensor COMMAND test_tensor)
add_test(NAME autograd COMMAND test_autograd)
add_test(NAME nn_linear COMMAND test_nn_linear)