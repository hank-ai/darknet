# Darknet object detection framework


# add_compile_definitions() requires 3.12
# runtime_dependencies requires 3.21 (windows only)
CMAKE_MINIMUM_REQUIRED (VERSION 3.12)


SET (CMAKE_CXX_STANDARD 17)
SET (CMAKE_CXX_STANDARD_REQUIRED ON)


IF (NOT CMAKE_BUILD_TYPE)
	SET (CMAKE_BUILD_TYPE Release)
ENDIF ()


INCLUDE (CM_version.cmake)

PROJECT (Darknet LANGUAGES C CXX VERSION ${DARKNET_VERSION_SHORT})

INCLUDE (CMakeDependentOption)
INCLUDE (CM_misc.cmake)
INCLUDE (CM_dependencies.cmake)
INCLUDE (CM_source.cmake)
INCLUDE (CM_package.cmake)
