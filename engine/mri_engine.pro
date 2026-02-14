# ================================
# Glimpse MRI - Engine (Jetson)
# Builds: gui/release/libmri_engine.so
# CPU-first by default. Enable CUDA with: CONFIG+=with_cuda
# ================================

TEMPLATE = lib
TARGET = mri_engine
CONFIG += c++17 shared release warn_on
CONFIG -= app_bundle

# Output where GUI expects it
DESTDIR = $$PWD/../gui/release
OBJECTS_DIR = $$OUT_PWD/obj
MOC_DIR     = $$OUT_PWD/moc

# ---- include paths (common patterns) ----
INCLUDEPATH += $$PWD
exists($$PWD/include) { INCLUDEPATH += $$PWD/include }
exists($$PWD/src)     { INCLUDEPATH += $$PWD/src }

# ---- pkg-config deps that exist on Jetson ----
CONFIG += link_pkgconfig
PKGCONFIG += opencv4 pugixml fftw3f hdf5

# ---- HDF5 HL/C++: pc files often missing; link manually (you confirmed these exist) ----
HDF5_INCDIR = /usr/include/hdf5/serial
HDF5_LIBDIR = /usr/lib/aarch64-linux-gnu/hdf5/serial
INCLUDEPATH += $$HDF5_INCDIR
LIBS += -L$$HDF5_LIBDIR -lhdf5_hl -lhdf5_cpp -lhdf5_hl_cpp
message([eng][hdf5] manual hl/cpp: inc=$$HDF5_INCDIR lib=$$HDF5_LIBDIR)

# ---- ISMRMRD: installed but no ismrmrd.pc; link manually ----
INCLUDEPATH += /usr/include/ismrmrd
LIBS += -L/usr/lib/aarch64-linux-gnu -lismrmrd
message([eng][ismrmrd] manual link: -lismrmrd)

# ---- common linux libs ----
unix:LIBS += -ldl -lpthread -lz

# ---- optional OpenMP ----
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS   += -fopenmp

# ---- sources (recursive) ----
SOURCES += $$files($$PWD/*.c, true)
SOURCES += $$files($$PWD/*.cpp, true)
HEADERS += $$files($$PWD/*.h, true)
HEADERS += $$files($$PWD/*.hpp, true)

# ---- CUDA (optional) ----
contains(CONFIG, with_cuda) {
    DEFINES += ENGINE_HAS_CUDA=1
    INCLUDEPATH += /usr/local/cuda/include
    message([eng][cuda] C++ include=/usr/local/cuda/include define ENGINE_HAS_CUDA=1)
    CUDA_DIR = /usr/local/cuda
    NVCC = $$CUDA_DIR/bin/nvcc
    CUDA_ARCH = 87   # Jetson Orin = SM 87

    CUDA_SOURCES = $$files($$PWD/*.cu, true)

    cuda.input = CUDA_SOURCES
    cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o

    cuda.commands = $$NVCC -c -O3 --use_fast_math \
        -Xcompiler -fPIC \
        -gencode arch=compute_$${CUDA_ARCH},code=sm_$${CUDA_ARCH} \
        -I$$PWD -I$$PWD/include -I$$PWD/src \
        -I/usr/include/ismrmrd -I/usr/include/hdf5/serial \
        ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

    cuda.variable_out = OBJECTS
    cuda.CONFIG += no_link
    QMAKE_EXTRA_COMPILERS += cuda

    LIBS += -L$$CUDA_DIR/lib64 -lcudart -lcufft -lcublas
    message([eng][cuda] enabled: nvcc=$$NVCC arch=sm_$$CUDA_ARCH cu_files=$$size(CUDA_SOURCES))
} else {
    message([eng][cuda] disabled (CPU-only). Add CONFIG+=with_cuda to enable.)
}

message([eng] DESTDIR=$$DESTDIR)
message([eng] PKGCONFIG=$$PKGCONFIG)
