cmake_minimum_required(VERSION 3.16)

# Project name
project(CVSegmenter VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt specific settings (Auto-generates necessary Qt code)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Find Qt
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

# Find OpenCV
find_package(OpenCV REQUIRED)

# 1. Define ALL source files first
set(SOURCES
    src/main.cpp
    src/MainWindow.cpp
    include/MainWindow.h
)

# 2. Create the executable using the full list of sources
add_executable(${PROJECT_NAME} ${SOURCES})

# 3. Tell CMake where to look for your custom header files
target_include_directories(${PROJECT_NAME} PRIVATE include)

# 4. Include OpenCV headers
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

# 5. Link Qt and OpenCV libraries to the executable
target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets ${OpenCV_LIBS})