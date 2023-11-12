# Darknet object detection framework


# add_compile_definitions() requires 3.12
# CUDA detection requires 3.18
# runtime_dependencies requires 3.21 (windows only)
# cuda_architectures auto detection requires 3.24
CMAKE_MINIMUM_REQUIRED (VERSION 3.24)


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

if (UNIX)
  set(_INSTALL_CFG_PREFIX "/opt/darknet/")
else()
  set(_INSTALL_CFG_PREFIX "")
endif()
set(INSTALL_CFG_PREFIX	 ${_INSTALL_CFG_PREFIX} CACHE PATH "Path where cfg files will be installed")


INCLUDE (CM_misc.cmake)
INCLUDE (CM_dependencies.cmake)
INCLUDE (CM_source.cmake)
INCLUDE (CM_package.cmake)
