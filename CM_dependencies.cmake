# Darknet object detection framework


# ==========
# == CUDA ==
# ==========
CHECK_LANGUAGE (CUDA)
IF (CMAKE_CUDA_COMPILER)
	MESSAGE (STATUS "CUDA detected. Darknet will use the GPU.")
	ENABLE_LANGUAGE (CUDA)
	FIND_PACKAGE(CUDAToolkit)
	INCLUDE_DIRECTORIES (${CUDAToolkit_INCLUDE_DIRS})
	ADD_COMPILE_DEFINITIONS (GPU) # TODO rename this to DARKNET_USE_GPU or DARKNET_USE_CUDA?
	SET (CMAKE_CUDA_STANDARD 14)
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
#	SET (DARKNET_CUDA_ARCHITECTURES "86")
#	SET (DARKNET_CUDA_ARCHITECTURES "75;80;86")
	SET (DARKNET_CUDA_ARCHITECTURES "native")
	SET (DARKNET_USE_CUDA ON)
	SET (DARKNET_LINK_LIBS ${DARKNET_LINK_LIBS} CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::curand)
ELSE ()
	MESSAGE (WARNING "CUDA not found. Darknet will be CPU-only.")
ENDIF ()


# ===========
# == cuDNN ==
# ===========
IF (DARKNET_USE_CUDA)
	IF (WIN32)
		# If installed according to the NVIDIA instructions, CUDNN should look like this:
		#		C:\Program Files\NVIDIA\CUDNN\v8.x\...
		# The .dll is named:
		#		C:\Program Files\NVIDIA\CUDNN\v8.x\bin\cudnn64_8.dll
		# And the header should look like:
		#		C:\Program Files\NVIDIA\CUDNN\v8.x\include\cudnn.h
		#
		SET (CUDNN_DIR "C:/Program Files/NVIDIA/CUDNN/v8.x")
		SET (CUDNN_DLL "${CUDNN_DIR}/bin/cudnn64_8.dll")
		SET (CUDNN_LIB "${CUDNN_DIR}/lib/x64/cudnn.lib")
		SET (CUDNN_HEADER "${CUDNN_DIR}/include/cudnn.h")
		IF (EXISTS ${CUDNN_DLL} AND EXISTS ${CUDNN_LIB} AND EXISTS ${CUDNN_HEADER})
			MESSAGE (STATUS "cuDNN found at ${CUDNN_DIR}")
			INCLUDE_DIRECTORIES (${CUDNN_DIR}/include/)
			ADD_COMPILE_DEFINITIONS (CUDNN) # TODO this needs to be renamed
			ADD_COMPILE_DEFINITIONS (CUDNN_HALF)
			SET (DARKNET_LINK_LIBS ${DARKNET_LINK_LIBS} ${CUDNN_LIB})
		ELSE ()
			MESSAGE (WARNING "Did not find cuDNN at ${CUDNN_DIR}")
		ENDIF ()
	ELSE ()
		# Should be slightly easier to deal with on Linux if it was installed correctly.
		FIND_LIBRARY (CUDNN cudnn OPTIONAL QUIET)
		IF (NOT CUDNN)
			MESSAGE (STATUS "Skipping cuDNN")
		ELSE ()
			MESSAGE (STATUS "Enabling cuDNN")
			ADD_COMPILE_DEFINITIONS (CUDNN) # TODO this needs to be renamed
			ADD_COMPILE_DEFINITIONS (CUDNN_HALF)
			SET (DARKNET_LINK_LIBS ${DARKNET_LINK_LIBS} ${CUDNN})
		ENDIF ()
	ENDIF ()
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


# =============
# == Threads ==
# =============
FIND_PACKAGE (Threads REQUIRED)
MESSAGE (STATUS "Found Threads ${Threads_VERSION}")
SET (DARKNET_LINK_LIBS ${DARKNET_LINK_LIBS} Threads::Threads)
IF (WIN32)
	FIND_PACKAGE (PThreads4W REQUIRED)
	SET (DARKNET_LINK_LIBS ${DARKNET_LINK_LIBS} PThreads4W::PThreads4W)
ENDIF ()


# ============
# == OpenCV ==
# ============
FIND_PACKAGE (OpenCV CONFIG REQUIRED)
MESSAGE (STATUS "Found OpenCV ${OpenCV_VERSION}")
INCLUDE_DIRECTORIES (${OpenCV_INCLUDE_DIRS})
SET (DARKNET_LINK_LIBS ${DARKNET_LINK_LIBS} ${OpenCV_LIBS})


# ============
# == OpenMP ==
# ============
FIND_PACKAGE (OpenMP QUIET) # optional
IF (NOT OPENMP_FOUND)
	MESSAGE (WARNING "OpenMP not found. Building Darknet without support for OpenMP.")
ELSE ()
	MESSAGE (STATUS "Found OpenMP ${OpenMP_VERSION}")
	ADD_COMPILE_DEFINITIONS (OPENMP)
	SET (DARKNET_LINK_LIBS ${DARKNET_LINK_LIBS} OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
	IF (COMPILER_IS_GNU_OR_CLANG)
		ADD_COMPILE_OPTIONS(-fopenmp)
	ENDIF ()
ENDIF ()


# ===============
# == AVX & SSE ==
# ===============
CMAKE_DEPENDENT_OPTION (ENABLE_SSE_AND_AVX "Enable AVX and SSE optimizations (Intel and AMD only)" ON "COMPILER_IS_GNU_OR_CLANG;HARDWARE_IS_X86" OFF)
IF (NOT ENABLE_SSE_AND_AVX)
	MESSAGE (WARNING "AVX and SSE optimizations are disabled.")
ELSE ()
	MESSAGE (STATUS "Enabling AVX and SSE optimizations.")
	ADD_COMPILE_OPTIONS(-ffp-contract=fast)
	ADD_COMPILE_OPTIONS(-mavx)
	ADD_COMPILE_OPTIONS(-mavx2)
	ADD_COMPILE_OPTIONS(-msse3)
	ADD_COMPILE_OPTIONS(-msse4.1)
	ADD_COMPILE_OPTIONS(-msse4.2)
	ADD_COMPILE_OPTIONS(-msse4a)
ENDIF ()

#MESSAGE (STATUS "Link: ${DARKNET_LINK_LIBS}")
