# Darknet object detection framework


CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/darknet_version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/darknet_version.h)

# ==
# For now we only have 1 source file in the main application.  Keep things simple I guess.
# ==
FILE (GLOB APPSRC darknet.cpp)
IF (WIN32)
	INCLUDE_DIRECTORIES (windows)
	FILE (GLOB WINRC windows/*.rc)
	LIST (APPEND APPSRC ${WINRC})
ENDIF ()

LIST (SORT APPSRC)


# ==
# The library is more complicated.  It has lots of source files, more optional files depending
# on whether or not CUDA is enabled, and yet more files if we're building on Windows.
# ==
FILE (GLOB LIBSRC *.cpp)
LIST (REMOVE_ITEM LIBSRC ${APPSRC})

IF (DARKNET_USE_CUDA)
	FILE (GLOB CUDASRC *.cu)
	LIST (APPEND LIBSRC ${CUDASRC})
ENDIF ()

IF (WIN32)
	INCLUDE_DIRECTORIES (windows)
	FILE (GLOB WINSRC windows/*.c)
	LIST (APPEND LIBSRC ${WINSRC})
ENDIF ()

LIST (SORT LIBSRC)


# ==
# First thing we do is build an object library so things don't have to be compiled twice.
# ==
ADD_LIBRARY(darknetobjlib OBJECT ${LIBSRC})
IF (DARKNET_USE_CUDA)
	SET_TARGET_PROPERTIES (darknetobjlib PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (darknetobjlib PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (darknetobjlib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()


# ==
# Next step is to create the actual darknet library.
# ==
ADD_LIBRARY (darknetlib SHARED $<TARGET_OBJECTS:darknetobjlib>)
SET_TARGET_PROPERTIES (darknetlib PROPERTIES OUTPUT_NAME "darknet")
SET_TARGET_PROPERTIES (darknetlib PROPERTIES PUBLIC_HEADER "darknet.h;darknet_version.h")
IF (DARKNET_USE_CUDA)
	SET_TARGET_PROPERTIES (darknetlib PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (darknetlib PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (darknetlib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()
TARGET_LINK_LIBRARIES (darknetlib PRIVATE ${DARKNET_LINK_LIBS})


# ==
# And lastly is the traditional darknet CLI.  I would have liked to have just 1 source file in the CLI and link against
# the darknet library, but this was causing some link errors in Windows:
#
#		fatal error LNK1149: output filename matches input filename 'C:\src\darknet\build\src\Release\darknet.lib'
#
# So for now we'll simply re-link all the objects into the CLI.  Not perfect, but it gets us past the problem.
# ==
ADD_EXECUTABLE (darknetcli ${APPSRC} $<TARGET_OBJECTS:darknetobjlib>)
SET_TARGET_PROPERTIES (darknetcli PROPERTIES OUTPUT_NAME "darknet")
IF (DARKNET_USE_CUDA)
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()
TARGET_LINK_LIBRARIES (darknetcli PRIVATE ${DARKNET_LINK_LIBS})


INSTALL (TARGETS darknetlib LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)


IF (UNIX)
	# non-Windows simple installation
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
