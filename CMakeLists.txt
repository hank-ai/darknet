# Darknet object detection framework


# add_compile_definitions() requires 3.12
# CUDA detection requires 3.18
# runtime_dependencies requires 3.21 (windows only)
# cuda_architectures auto detection requires 3.24
CMAKE_MINIMUM_REQUIRED (VERSION 3.24)


IF (CMAKE_HOST_APPLE)
	OPTION (DARKNET_TRY_MPS "Attempt to find Apple Metal/MPS support" ON)
	IF (NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND
		NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
		IF (DARKNET_TRY_MPS)
			# This must be set before PROJECT() initializes the compiler toolchain.
			SET (CMAKE_OSX_DEPLOYMENT_TARGET "26.0")
		ELSE ()
			SET (CMAKE_OSX_DEPLOYMENT_TARGET "")
		ENDIF ()
		# Keep the project-selected value out of the cache so toggling MPS in
		# the same build tree cannot retain an automatic deployment target.
	ENDIF ()
ENDIF ()


SET (CMAKE_CXX_STANDARD 17)
SET (CMAKE_CXX_STANDARD_REQUIRED ON)


IF (NOT CMAKE_BUILD_TYPE)
	# create either Release or Debug builds
	SET (CMAKE_BUILD_TYPE Release)
ENDIF ()

# to get verbose build output, set this to ON
SET (CMAKE_VERBOSE_MAKEFILE OFF)

# to get lots of debug output while running cmake, set this to ON
SET (CMAKE_FIND_DEBUG_MODE OFF)

INCLUDE (CM_version.cmake)

PROJECT (Darknet LANGUAGES C CXX VERSION ${DARKNET_VERSION_SHORT})

INCLUDE (CheckLanguage)
INCLUDE (CMakeDependentOption)

IF (APPLE)
	IF (DARKNET_TRY_MPS)
		IF (CMAKE_OSX_ARCHITECTURES)
			IF (NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
				MESSAGE (FATAL_ERROR "Apple Metal/MPS builds require CMAKE_OSX_ARCHITECTURES=arm64.")
			ENDIF ()
		ELSEIF (NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
			MESSAGE (FATAL_ERROR "Apple Metal/MPS builds require an arm64 host. Configure with -DDARKNET_TRY_MPS=OFF for a CPU-only macOS build.")
		ENDIF ()

		IF (CMAKE_OSX_SYSROOT)
			SET (DARKNET_APPLE_SDK "${CMAKE_OSX_SYSROOT}")
		ELSE ()
			SET (DARKNET_APPLE_SDK "macosx")
		ENDIF ()
		EXECUTE_PROCESS (
			COMMAND xcrun --sdk "${DARKNET_APPLE_SDK}" --show-sdk-version
			RESULT_VARIABLE DARKNET_APPLE_SDK_RESULT
			OUTPUT_VARIABLE DARKNET_APPLE_SDK_VERSION
			OUTPUT_STRIP_TRAILING_WHITESPACE
			ERROR_QUIET)
		IF (NOT DARKNET_APPLE_SDK_RESULT EQUAL 0 OR
			DARKNET_APPLE_SDK_VERSION VERSION_LESS 26.0)
			MESSAGE (FATAL_ERROR "Apple Metal/MPS builds require the macOS 26.0 SDK or newer (found '${DARKNET_APPLE_SDK_VERSION}').")
		ENDIF ()
		UNSET (DARKNET_APPLE_SDK)
		UNSET (DARKNET_APPLE_SDK_RESULT)
		UNSET (DARKNET_APPLE_SDK_VERSION)

		IF (NOT CMAKE_OSX_DEPLOYMENT_TARGET OR
			CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 26.0)
			MESSAGE (FATAL_ERROR "Apple Metal/MPS builds require macOS deployment target 26.0 or newer.")
		ENDIF ()
	ENDIF ()
ENDIF ()

INCLUDE (CM_misc.cmake)
INCLUDE (CM_dependencies.cmake)
INCLUDE (CM_source.cmake)
INCLUDE (CM_package.cmake)

# If you have Google Test installed, then you can enable the unit tests.
# On Ubuntu for example, you'd run the following commands:
#
#		sudo apt-get install googletest
#		cd /usr/src/googletest/
#		sudo mkdir build
#		cd build
#		sudo cmake ..
#		sudo make
#		sudo cp lib/*.a /usr/lib
#
# Once you successfully built Darknet, run "make test", "ctest", or "darknet_tests".
#
FIND_LIBRARY (GTEST gtest)
IF (GTEST)
	ENABLE_TESTING ()
	ADD_SUBDIRECTORY (src-test)
ENDIF ()
