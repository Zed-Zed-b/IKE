cmake_minimum_required(VERSION 3.15)
project(IKE VERSION 1.0.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 项目配置选项
option(IKE_BUILD_TESTS "Build IKE tests" ON)

set(COMPILE_OPTIONS
    -O3
    -funroll-loops
    -fomit-frame-pointer
    -march=native
    # -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mavx512vpopcntdq
    -mpopcnt
    -fopenmp
)
    

# 查找依赖包
find_package(OpenMP REQUIRED)

# 设置 FAISS 路径（建议改为可配置的）
set(FAISS_INCLUDE_ROOT "/path/to/faiss/include" CACHE PATH "Path to FAISS include")
set(FAISS_LIB_ROOT "/path/to/faiss/lib" CACHE PATH "Path to FAISS library")

message(STATUS "FAISS include: ${FAISS_INCLUDE_ROOT}")
message(STATUS "FAISS lib directory: ${FAISS_LIB_ROOT}")

# 定义 IKE 库的源文件
set(IKE_SOURCES
    src/IForest_index/gemc.cpp
    src/IForest_index/IForest_Flat.cpp
    src/IForest_index/IVFIForest.cpp
    src/IForest_index/IForest_HNSW.cpp
    src/IForest_index/myHNSW.cpp
)

# 创建 IKE 静态库
add_library(IKE STATIC ${IKE_SOURCES})
target_include_directories(IKE PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${FAISS_INCLUDE_ROOT}
)

target_compile_options(IKE PRIVATE 
    ${COMPILE_OPTIONS}
)

target_link_libraries(IKE PRIVATE 
    OpenMP::OpenMP_CXX
)

target_link_directories(IKE PUBLIC ${FAISS_LIB_ROOT})
target_link_libraries(IKE PUBLIC faiss_avx512)

# 构建测试程序
if(IKE_BUILD_TESTS)
    # 定义测试程序列表
    set(TEST_PROGRAMS
        Exhaustive_Search_with_IKE
        HNSW_with_IKE  
        IVF_with_IKE
    )
    
    # 为每个测试程序创建可执行文件
    foreach(test_program IN LISTS TEST_PROGRAMS)
        add_executable(${test_program} test/${test_program}.cpp test/utils.cpp)
        target_include_directories(${test_program} PRIVATE 
          ${CMAKE_CURRENT_SOURCE_DIR}/test
        )
        target_link_libraries(${test_program} PRIVATE IKE OpenMP::OpenMP_CXX)
        
        target_compile_options(${test_program} PRIVATE 
          ${COMPILE_OPTIONS}
        )
        
    endforeach()
endif()
