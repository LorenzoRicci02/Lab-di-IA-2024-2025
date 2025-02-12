cmake_minimum_required(VERSION 3.10) # O una versione successiva
project(FaceScan)

# Trova la libreria OpenCV
find_package(OpenCV REQUIRED)

# Aggiungi tutti i file sorgente, inclusi ml.cpp
add_executable(FaceScan
    src/main.cpp
    src/corners.cpp
    src/ml.cpp 
    src/sift.cpp
    src/orb.cpp
)

# Collega OpenCV
target_link_libraries(FaceScan ${OpenCV_LIBS})

# Imposta la directory di output per l'eseguibile nella directory principale
set_target_properties(FaceScan PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
