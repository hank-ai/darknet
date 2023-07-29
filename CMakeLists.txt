# Darknet object detection framework

CMAKE_MINIMUM_REQUIRED( VERSION 3.20)

SET (CMAKE_CXX_STANDARD 17)
SET (CMAKE_CXX_STANDARD_REQUIRED ON)

IF (NOT CMAKE_BUILD_TYPE)
	SET (CMAKE_BUILD_TYPE Release)
ENDIF ()

PROJECT (Darknet C CXX CUDA)

# Do not build in the project's root directory!  The build steps are explained in the git repo.
# See here for details:  https://github.com/hank-ai/darknet/tree/stephane-dev#building
IF (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
	MESSAGE (FATAL_ERROR  "Please create a 'build' directory.  See the build instructions for details.")
ENDIF ()


# Create a version string from the git tag and commit hash (see src/darknet_version.h.in).
# Should look similar to this:
#
#		v1.99-63-gc5c3569
#
EXECUTE_PROCESS (COMMAND git describe --tags --dirty OUTPUT_VARIABLE DARKNET_VERSION_STRING OUTPUT_STRIP_TRAILING_WHITESPACE)
MESSAGE (STATUS "Darknet ${DARKNET_VERSION_STRING}")


# Rename several old pre-v2.x files and directories if they exist so they don't get used by mistake.
FOREACH (filename darknet libdarknet.so obj uselib)
	IF (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
		FILE (RENAME "${CMAKE_CURRENT_SOURCE_DIR}/${filename}" "${CMAKE_CURRENT_SOURCE_DIR}/old_${filename}")
	ENDIF ()
ENDFOREACH ()


FIND_PACKAGE (Threads REQUIRED)
MESSAGE (STATUS "Found Threads ${Threads_VERSION}")


FIND_PACKAGE (OpenCV CONFIG REQUIRED)
MESSAGE (STATUS "Found OpenCV ${OpenCV_VERSION}")
INCLUDE_DIRECTORIES (${OpenCV_INCLUDE_DIRS})
ADD_COMPILE_DEFINITIONS (DOPENCV) # TODO remove this


FIND_PACKAGE (OpenMP QUIET) # optional!
IF (NOT OPENMP_FOUND)
	MESSAGE (WARNING "OpenMP not found. Building Darknet without support for OpenMP.")
ELSE ()
	MESSAGE (STATUS "Found OpenMP ${OpenMP_VERSION}")
	ADD_COMPILE_DEFINITIONS (DOPENMP)
	ADD_COMPILE_OPTIONS(-fopenmp)
#	TODO LDFLAGS+= -lgomp
ENDIF ()


IF (WIN32)
	ADD_COMPILE_OPTIONS (/W4)				# warning level (high)
#	ADD_COMPILE_OPTIONS (/WX)				# treat warnings as errors
	ADD_COMPILE_OPTIONS (/permissive-)		# stick to C++ standards (turn off Microsoft-specific extensions)
	ADD_COMPILE_OPTIONS (/wd4100)			# disable "unreferenced formal parameter"
	ADD_COMPILE_OPTIONS (/wd4127)			# disable "conditional expression is constant"
	ADD_COMPILE_DEFINITIONS (_CRT_SECURE_NO_WARNINGS )	# don't complain about localtime()
#	SET (CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" )
ELSE ()
	ADD_COMPILE_OPTIONS (-Wall -Wextra -Wno-unused-parameter)
	ADD_COMPILE_OPTIONS (-Wno-write-strings -Wno-unused-result -Wno-missing-field-initializers -Wno-ignored-qualifiers -Wno-sign-compare) # TODO clean up code so we can remove this
ENDIF ()


INCLUDE_DIRECTORIES (3rdparty/stb/include/) # TODO remove
INCLUDE_DIRECTORIES (include) # TODO remove

ADD_SUBDIRECTORY (src)
