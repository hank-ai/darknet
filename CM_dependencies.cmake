# Darknet object detection framework


FIND_PACKAGE (Threads REQUIRED)
MESSAGE (STATUS "Found Threads ${Threads_VERSION}")


FIND_PACKAGE (OpenCV CONFIG REQUIRED)
MESSAGE (STATUS "Found OpenCV ${OpenCV_VERSION}")
INCLUDE_DIRECTORIES (${OpenCV_INCLUDE_DIRS})
ADD_COMPILE_DEFINITIONS (OPENCV) # TODO remove this once OpenCV is no longer optional


FIND_PACKAGE (OpenMP QUIET) # optional
IF (NOT OPENMP_FOUND)
	MESSAGE (WARNING "OpenMP not found. Building Darknet without support for OpenMP.")
ELSE ()
	MESSAGE (STATUS "Found OpenMP ${OpenMP_VERSION}")
	ADD_COMPILE_DEFINITIONS (OPENMP)
	ADD_COMPILE_OPTIONS(-fopenmp)
#	TODO LDFLAGS+= -lgomp
ENDIF ()
