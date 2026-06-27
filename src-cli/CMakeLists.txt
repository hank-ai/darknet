# Darknet object detection framework


# ==
# Darknet CLI is actually quite simple in terms of source files.
# ==
MESSAGE(STATUS "Setting up DARKNET CLI")

FILE (GLOB CLISRC *.cpp)
IF (WIN32)
	INCLUDE_DIRECTORIES (windows)
	FILE (GLOB WINRC windows/*.rc)
	LIST (APPEND CLISRC ${WINRC})
ENDIF ()

LIST (SORT CLISRC)

SET (DARKNETCLI_OBJECTS $<TARGET_OBJECTS:darknetobjlib>)
IF (TARGET darknetgpuobjlib)
	LIST (APPEND DARKNETCLI_OBJECTS $<TARGET_OBJECTS:darknetgpuobjlib>)
ENDIF ()

ADD_EXECUTABLE (darknetcli ${CLISRC} ${DARKNETCLI_OBJECTS})
SET_TARGET_PROPERTIES (darknetcli PROPERTIES OUTPUT_NAME "darknet")
IF (DARKNET_USE_CUDA OR DARKNET_USE_ROCM)
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()
TARGET_LINK_LIBRARIES (darknetcli PRIVATE ${DARKNET_LINK_LIBS})
IF (DARKNET_USE_OPENMP)
	TARGET_LINK_LIBRARIES (darknetcli PRIVATE OpenMP::OpenMP_CXX)
ENDIF ()

IF (UNIX)
	# non-Windows installation is very simple
	INSTALL (TARGETS darknetcli DESTINATION bin)
ELSE ()
	# more complicated install for Windows so we also get the .DLL files copied over from vcpkg
	# https://stackoverflow.com/a/72088102/13022
	INSTALL (TARGETS darknetcli
		DESTINATION bin
		RUNTIME_DEPENDENCIES
			PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-" "wpaxholder" "HvsiFileTrust" "PdmUtilities"
			POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
			DIRECTORIES ${CMAKE_BINARY_DIR}/bin
		RUNTIME DESTINATION bin
		)
ENDIF ()
