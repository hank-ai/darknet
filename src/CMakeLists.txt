# Darknet object detection framework


CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/darknet_version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/darknet_version.h)

# all the source files that make up the main application
FILE (GLOB APPSRC darknet.cpp)
LIST (SORT APPSRC)

# all the source files that make up the darknet library
FILE (GLOB LIBSRC *.cpp)
LIST (REMOVE_ITEM LIBSRC ${APPSRC})
LIST (SORT LIBSRC)

IF (WIN32)
	INCLUDE_DIRECTORIES (windows)
	FILE (GLOB WINSRC windows/*.c)
	LIST (APPEND LIBSRC ${WINSRC})
ENDIF ()

IF (DARKNET_USE_CUDA)
	FILE (GLOB CUDASRC *.cu)
	LIST (APPEND LIBSRC ${CUDASRC})
	MESSAGE (STATUS "CU=${CUDASRC}")
ENDIF ()

# compile everything once
ADD_LIBRARY(objlib OBJECT ${LIBSRC})
IF (DARKNET_USE_CUDA)
	SET_TARGET_PROPERTIES (objlib PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
ENDIF ()

# create 2 outputs:  the darknet library, and the ususual darknet CLI
ADD_LIBRARY (darknetlib SHARED $<TARGET_OBJECTS:objlib>)
SET_TARGET_PROPERTIES (darknetlib PROPERTIES OUTPUT_NAME "darknet")
SET_TARGET_PROPERTIES (darknetlib PROPERTIES PUBLIC_HEADER "darknet.h;darknet_version.h")
IF (DARKNET_USE_CUDA)
	SET_TARGET_PROPERTIES (darknetlib PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
ENDIF ()
TARGET_LINK_LIBRARIES (darknetlib PRIVATE ${DARKNET_LINK_LIBS})

# would like to have the CLI link against the darknet library, but this fails on Windows:
#		fatal error LNK1149: output filename matches input filename 'C:\src\darknet\build\src\Release\darknet.lib'
# so for now we'll simply re-link all the objects into the CLI
ADD_EXECUTABLE (darknetcli ${APPSRC} $<TARGET_OBJECTS:objlib>)
SET_TARGET_PROPERTIES (darknetcli PROPERTIES OUTPUT_NAME "darknet")
IF (DARKNET_USE_CUDA)
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
ENDIF ()
TARGET_LINK_LIBRARIES (darknetcli PRIVATE ${DARKNET_LINK_LIBS})

INSTALL (TARGETS darknetlib LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)

IF (WIN32)
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
ELSE ()
	# non-Windows simple installation
	INSTALL (TARGETS darknetcli DESTINATION bin)
ENDIF ()
