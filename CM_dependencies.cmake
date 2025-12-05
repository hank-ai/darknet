# Darknet object detection framework


# =================
# == NVIDIA CUDA ==
# =================
CMAKE_DEPENDENT_OPTION (DARKNET_TRY_CUDA "Attempt to find NVIDIA/CUDA GPU support" ON "" ON)
IF (DARKNET_TRY_CUDA)
	CHECK_LANGUAGE (CUDA)
	IF (CMAKE_CUDA_COMPILER)
		MESSAGE (STATUS "CUDA detected. Darknet will use NVIDIA GPUs.  CUDA compiler is ${CMAKE_CUDA_COMPILER}.")
		ENABLE_LANGUAGE (CUDA)
		FIND_PACKAGE(CUDAToolkit REQUIRED)
		INCLUDE_DIRECTORIES (${CUDAToolkit_INCLUDE_DIRS})
		ADD_COMPILE_DEFINITIONS (DARKNET_GPU_CUDA)
		ADD_COMPILE_DEFINITIONS (DARKNET_GPU)
		SET (CMAKE_CUDA_STANDARD 17)
		SET (CMAKE_CUDA_STANDARD_REQUIRED ON)
		#
		# Best to use "native" as the architecture, as this will use whatever GPU is installed.
		# But if desired, the exact major architecture index can also be specified.  For example:
		#
		# (note that some of these are no longer supported in recent versions of CUDA)
		#
		#	20: GeForce 400, 500, 600, GT-630
		#	30: GeForce 700, GT 730, 740, 760, 770
		#	35: Tesla K40
		#	37: Tesla K80
		#	50: Tesla Quadro M
		#	52: Quadro M6000, GeForce 900, 970, 980, Titan X
		#	53: Tegra Jetson TX1, X1, Drive CX, Drive PX, Jetson Nano
		#	60: Quadro GP100, Tesla P100, DGX-1
		#	61: GTX 1080, 1070, 1060, 1050, 1030, 1010, GP108 Titan Xp, Tesla P40, Tesla P4, Drive PX2
		#	62: Drive PX2, Tegra Jetson TX2
		#	70: DGX-1 Volta, Tesla V100, GTX 1180 GV104, Titan V, Quadro VG100
		#	72: Jetson AGX Xavier, AGX Pegasus, Jetson Xavier NX
		#	75: GTX RTX Turing, GTX 1660, RTX 2060, RTX 2070, RTX 2080, Titan RTX, Quadro RTX 4000, 5000, 6000, 8000, T1000, T2000, Tesla T4, XNOR Tensor Cores
		#	80: A100, GA100, DGX-A100, RTX 3080 (?)
		#	86: Tesla GA10x, RTX Ampere, RTX 3050, 3070, 3080, 3090, GA102, GA107, RTX A2000, A3000, A4000, A5000, A6000, A40, GA106, RTX 3060, GA104, A10, A16, A40, A2 Tensor
		#	87: Jetson AGX Orin, Drive AGX Orin
		#	89: RTX 4090, 4080, 6000, Tesla L40
		#	90: H100, GH100
		#
		IF (NOT DEFINED DARKNET_CUDA_ARCHITECTURES)
		#	SET (DARKNET_CUDA_ARCHITECTURES "86")
		#	SET (DARKNET_CUDA_ARCHITECTURES "75;80;86")
			SET (DARKNET_CUDA_ARCHITECTURES "native")
		ENDIF ()
		SET (DARKNET_USE_CUDA ON)
		LIST (APPEND DARKNET_LINK_LIBS CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::curand)
	ELSE ()
		MESSAGE (WARNING "Support for NVIDIA CUDA not found.")
	ENDIF ()
ELSE ()
	MESSAGE (WARNING "Support for NVIDIA CUDA is disabled.")
ENDIF ()


# ===========
# == cuDNN ==
# ===========
IF (DARKNET_USE_CUDA)
	# Look for cudnn, we will look in the same place as other CUDA libraries and also a few other places as well.
	FIND_PATH(cudnn_include cudnn.h
				HINTS ${CUDA_INCLUDE_DIRS} ENV CUDNN_INCLUDE_DIR ENV CUDA_PATH ENV CUDNN_HOME
				PATHS /usr/local /usr/local/cuda ENV CPATH
				PATH_SUFFIXES include)
	GET_FILENAME_COMPONENT(cudnn_hint_path "${CUDA_CUBLAS_LIBRARIES}" PATH)
	FIND_LIBRARY(cudnn cudnn
				HINTS ${cudnn_hint_path} ENV CUDNN_LIBRARY_DIR ENV CUDA_PATH ENV CUDNN_HOME
				PATHS /usr/local /usr/local/cuda ENV LD_LIBRARY_PATH
				PATH_SUFFIXES lib64 lib/x64 lib x64)
	IF (cudnn AND cudnn_include)
		MESSAGE (STATUS "Found cuDNN library: " ${cudnn})
		ADD_COMPILE_DEFINITIONS (CUDNN) # TODO this needs to be renamed
		ADD_COMPILE_DEFINITIONS (CUDNN_HALF)
		LIST (APPEND DARKNET_LINK_LIBS ${cudnn})
		MESSAGE (STATUS "Found cuDNN include: " ${cudnn_include})
		INCLUDE_DIRECTORIES (${cudnn_include})
	ELSE ()
		MESSAGE (WARNING "cuDNN not found.")
	ENDIF ()
ENDIF ()


# ======================
# == AMD GPU aka ROCM ==
# ======================
CMAKE_DEPENDENT_OPTION (DARKNET_TRY_ROCM "Attempt to find AMD/ROCm/HIP GPU support" ON "" ON)
IF (DARKNET_TRY_ROCM)
	CHECK_LANGUAGE (HIP)
	IF (CMAKE_HIP_COMPILER)
		MESSAGE (STATUS "AMD ROCm detected. Darknet will use AMD GPUs. HIP compiler is ${CMAKE_HIP_COMPILER}.")
		IF (NOT DEFINED ROCM_PATH)
			SET (ROCM_PATH "/opt/rocm")
		ENDIF ()
		LIST (APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
		ENABLE_LANGUAGE (HIP)
		FIND_PACKAGE(hip REQUIRED)
		FIND_PACKAGE(hipblas REQUIRED)
		FIND_PACKAGE(hiprand REQUIRED)
		FIND_PACKAGE(amd_smi REQUIRED)

		SET (DARKNET_USE_ROCM ON)

		SET (CMAKE_HIP_STANDARD 17)
		SET (CMAKE_HIP_STANDARD_REQUIRED ON)

		SET (CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
		SET (CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

		ADD_COMPILE_DEFINITIONS (__HIP_PLATFORM_HCC__)
		ADD_COMPILE_DEFINITIONS (__HIP_PLATFORM_AMD__)
		ADD_COMPILE_DEFINITIONS (DARKNET_GPU_ROCM)
		ADD_COMPILE_DEFINITIONS (DARKNET_GPU)

		INCLUDE_DIRECTORIES ("${ROCM_PATH}/include/")

		# Run "rocm-smi --showproductname" or "rocm-smi --showhw" to see which architecture to use.
		# For example, this can be set to "gfx1035;gfx1036;gfx1037" to build code for multiple architectures.
		#
		#	gfx1101: RX 7700 / 7800
		#
		IF (NOT DEFINED CMAKE_HIP_ARCHITECTURES)
			SET (CMAKE_HIP_ARCHITECTURES "gfx1101")
		ENDIF ()

		LIST (APPEND DARKNET_LINK_LIBS hip::host hip::device roc::hipblas roc::rocrand hip::hiprand amd_smi)

#		MESSAGE (STATUS "Enabling hipDNN")
#		ADD_COMPILE_DEFINITIONS (CUDNN) # TODO this needs to be renamed
#		ADD_COMPILE_DEFINITIONS (CUDNN_HALF)

	ELSE ()
		MESSAGE (WARNING "Support for AMD/ROCm/HIP not found.")
	ENDIF ()
ELSE ()
	MESSAGE (WARNING "Support for AMD/ROCm/HIP is disabled.")
ENDIF ()


# ==============
# == CPU-only ==
# ==============
IF (NOT DARKNET_USE_CUDA AND NOT DARKNET_USE_ROCM)
	SET (DARKNET_DETECTED_CPU_ONLY TRUE)
	MESSAGE (WARNING "Neither NVIDIA CUDA nor AMD ROCm detected.  Darknet will be CPU-only.")
ENDIF ()


# ========================
# == Intel/AMD Hardware ==
# ========================
IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86" OR
	CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_32" OR
	CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64" OR
	CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "AMD64")
	SET (HARDWARE_IS_X86 TRUE)
	MESSAGE (STATUS "Hardware is 32-bit or 64-bit, and seems to be Intel or AMD:  ${CMAKE_HOST_SYSTEM_PROCESSOR}")
ELSE ()
	SET (HARDWARE_IS_X86 FALSE)
	MESSAGE (STATUS "Hardware does not appear to be 32-bit or 64-bit, Intel or AMD:  ${CMAKE_HOST_SYSTEM_PROCESSOR}")
ENDIF ()


# ===============
# == GCC/Clang ==
# ===============
IF (CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	SET (COMPILER_IS_GNU_OR_CLANG TRUE)
ELSE ()
	SET (COMPILER_IS_GNU_OR_CLANG FALSE)
ENDIF ()


# ====================
# == GCC/Clang/MSCV ==
# ====================
IF (COMPILER_IS_GNU_OR_CLANG OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
	SET (COMPILER_IS_GNU_OR_CLANG_OR_MSVC TRUE)
ELSE ()
	SET (COMPILER_IS_GNU_OR_CLANG_OR_MSVC FALSE)
ENDIF ()

MESSAGE (STATUS "Compiler:  GNU/Clang=${CMAKE_COMPILER_IS_GNUCC} GNU/Clang/MSVC=${COMPILER_IS_GNU_OR_CLANG_OR_MSVC}: ${CMAKE_CXX_COMPILER_ID}")


# =============
# == Threads ==
# =============
FIND_PACKAGE (Threads REQUIRED)
MESSAGE (STATUS "Found Threads ${Threads_VERSION}")
LIST (APPEND DARKNET_LINK_LIBS Threads::Threads)

# ============================================================
# == OpenBLAS (Basic Linear Algebra Subprograms)			==
# == This is only used when Darknet is built for CPU-only.	==
# ============================================================
IF (DARKNET_DETECTED_CPU_ONLY)

	IF (NOT DEFINED DARKNET_TRY_OPENBLAS)
		SET (DARKNET_TRY_OPENBLAS True)
	ENDIF ()

	IF (DARKNET_TRY_OPENBLAS)
		IF (APPLE)
			# APPLE devices need a hint to find the brew installation.  On top of which, on some distrios (and again APPLE)
			# the package is called OpenBLAS, while on other distros it is called OpenBLAS64.  We need to search for both.
			FIND_PACKAGE (OpenBLAS NAMES OpenBLAS64 OpenBLAS QUIET HINTS "/opt/homebrew/opt/openblas/lib/cmake/openblas")

			IF (OpenBLAS_FOUND)
				LIST (APPEND DARKNET_LINK_LIBS ${OpenBLAS_LIBRARIES})
				INCLUDE_DIRECTORIES (${OpenBLAS_INCLUDE_DIRS})
				ADD_COMPILE_DEFINITIONS (DARKNET_USE_OPENBLAS)
			ELSE ()
				MESSAGE (WARNING "Apple OpenBLAS not found. Building Darknet for CPU-only without support for OpenBLAS.")
			ENDIF()
		ELSE() # Win32, vcpkg and Linux
			SET(BLA_VENDOR OpenBLAS)
			SET(BLA_SIZEOF_INTEGER 8) # force 64 bit
			FIND_PACKAGE (BLAS)

			IF (BLAS_FOUND)
				MESSAGE (STATUS "Found OpenBLAS")
				LIST (APPEND DARKNET_LINK_LIBS BLAS::BLAS)
				ADD_COMPILE_DEFINITIONS (DARKNET_USE_OPENBLAS)
			ELSE ()
				MESSAGE (WARNING "OpenBLAS not found. Building Darknet for CPU-only without support for OpenBLAS.")
			ENDIF()
		ENDIF()
	ELSE ()
		MESSAGE (WARNING "OpenBLAS is disabled. Building Darknet for CPU-only without support for OpenBLAS.")
	ENDIF ()
ELSE ()
	MESSAGE (STATUS "Skipping OpenBLAS since we have a GPU.")
ENDIF ()

# ============
# == OpenCV ==
# ============
FIND_PACKAGE (OpenCV REQUIRED)
MESSAGE (STATUS "Found OpenCV ${OpenCV_VERSION}")
INCLUDE_DIRECTORIES (${OpenCV_INCLUDE_DIRS})
LIST (APPEND DARKNET_LINK_LIBS ${OpenCV_LIBS})


# ============
# == OpenMP ==
# ============
FIND_PACKAGE (OpenMP QUIET) # optional
IF (NOT OPENMP_FOUND)
	MESSAGE (WARNING "OpenMP not found. Building Darknet without support for OpenMP.")
ELSEIF (DARKNET_USE_ROCM)
	# TODO: This needs to be fixed.  What are we missing during the link process to make this work with clang++?
	MESSAGE (WARNING "Skipping OpenMP due to ROCm.")
ELSE ()
	MESSAGE (STATUS "Found OpenMP ${OpenMP_VERSION}")
	ADD_COMPILE_DEFINITIONS (DARKNET_OPENMP)
	LIST (APPEND DARKNET_LINK_LIBS OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
	IF (WIN32)
		ADD_COMPILE_OPTIONS (/openmp:experimental)
	ELSE ()
		ADD_COMPILE_DEFINITIONS (_GLIBCXX_PARALLEL)
		ADD_COMPILE_OPTIONS (-fopenmp)
		ADD_COMPILE_OPTIONS (${OpenMP_C_FLAGS})
		ADD_COMPILE_OPTIONS (${OpenMP_CXX_FLAGS})
	ENDIF()
ENDIF ()


# ===============
# == AVX & SSE ==
# ===============
CMAKE_DEPENDENT_OPTION (ENABLE_SSE_AND_AVX "Enable AVX and SSE optimizations (Intel and AMD only)" ON "COMPILER_IS_GNU_OR_CLANG_OR_MSVC;HARDWARE_IS_X86" OFF)
IF (NOT ENABLE_SSE_AND_AVX)
	MESSAGE (WARNING "AVX and SSE optimizations are disabled.")
ELSE ()
	MESSAGE (STATUS "Enabling AVX and SSE optimizations.")
	IF (COMPILER_IS_GNU_OR_CLANG)
		ADD_COMPILE_OPTIONS(-ffp-contract=fast)
		ADD_COMPILE_OPTIONS(-mavx)
		ADD_COMPILE_OPTIONS(-mavx2)
		ADD_COMPILE_OPTIONS(-msse3)
		ADD_COMPILE_OPTIONS(-msse4.1)
		ADD_COMPILE_OPTIONS(-msse4.2)
		ADD_COMPILE_OPTIONS(-msse4a)
	ELSE ()
		STRING (APPEND CMAKE_CXX_FLAGS " /arch:AVX2")
	ENDIF()
ENDIF ()


# ============
# == Timing ==
# ============
CMAKE_DEPENDENT_OPTION (ENABLE_TIMING_AND_TRACKING "Enable Darknet timing and tracking debugging" OFF "" OFF)
IF (ENABLE_TIMING_AND_TRACKING)
	MESSAGE (WARNING "Darknet timing and tracking debug code is *ENABLED*!")
	ADD_COMPILE_DEFINITIONS(DARKNET_TIMING_AND_TRACKING_ENABLED)
ENDIF ()


# ===================================
# == Protocol Buffer (ONNX export) ==
# ===================================
IF (NOT DEFINED DARKNET_TRY_ONNX)
	SET (DARKNET_TRY_ONNX True)
ENDIF ()
IF (DARKNET_TRY_ONNX)
	FIND_PACKAGE (Protobuf QUIET)
	IF (Protobuf_FOUND)
		MESSAGE (STATUS "Found protocol buffer (needed for ONNX export) ${Protobuf_VERSION}")
		INCLUDE_DIRECTORIES (${Protobuf_INCLUDE_DIRS})
		ADD_COMPILE_DEFINITIONS(DARKNET_HAS_PROTOBUF)
	ELSE ()
		MESSAGE (WARNING "Protocol buffer not found.  Skipping support for ONNX export.")
	ENDIF ()
ELSE ()
	MESSAGE (STATUS "Darknet is skipping ONNX.  Run cmake with '-DDARKNET_TRY_ONNX=True' to add support for the ONNX export tool.")
ENDIF ()
