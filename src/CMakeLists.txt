cmake_minimum_required(VERSION 3.8)

project(seat C CXX)

find_package(OpenCV REQUIRED)
find_package(X11 REQUIRED)  # Tambahkan ini untuk menemukan library X11

set(CMAKE_CXX_STANDARD 17)

# Add support for threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR})

add_executable(test yolov8Normal.cpp inference.cpp)
target_link_libraries(test
    ${OpenCV_LIBS}
    Threads::Threads  # Link dengan Threads library
    X11  # Link dengan X11 library
)

add_executable(both both.cpp inference.cpp)
target_link_libraries(both
    ${OpenCV_LIBS}
    Threads::Threads  # Link dengan Threads library
    X11  # Link dengan X11 library
)

add_executable(HR hr.cpp inference.cpp)
target_link_libraries(HR
    ${OpenCV_LIBS}
    Threads::Threads  # Link dengan Threads library
    X11  # Link dengan X11 library
)

add_executable(RR rr.cpp inference.cpp)
target_link_libraries(RR
    ${OpenCV_LIBS}
    Threads::Threads  # Link dengan Threads library
    X11  # Link dengan X11 library
)