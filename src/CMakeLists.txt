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
LIST (REMOVE_ITEM LIBSRC yolo_v2_class.cpp)

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
# First step is to create the actual darknet library.
# ==
ADD_LIBRARY (darknetlib SHARED ${LIBSRC})
SET_TARGET_PROPERTIES (darknetlib PROPERTIES OUTPUT_NAME "darknet")
SET_TARGET_PROPERTIES (darknetlib PROPERTIES PUBLIC_HEADER "darknet.h;darknet_version.h")
IF (DARKNET_USE_CUDA)
	SET_TARGET_PROPERTIES (darknetlib PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (darknetlib PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (darknetlib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()
TARGET_LINK_LIBRARIES (darknetlib PRIVATE ${DARKNET_LINK_LIBS})
INSTALL (TARGETS darknetlib LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)


# ==
# The old C++ YOLOv2 library is still used by some projects.  E.g., the original C# wrapper.
# But as far as I can tell, this library was only ever used on Windows.  Doesn't look like
# the old Linux Makefile in the AlexeyAB repo ever tried to build this on Linux.
# ==
IF (WIN32)
	ADD_LIBRARY (yolo_v2_class SHARED yolo_v2_class.cpp)
	SET_TARGET_PROPERTIES (yolo_v2_class PROPERTIES PUBLIC_HEADER "yolo_v2_class.hpp")
	TARGET_LINK_LIBRARIES (yolo_v2_class PRIVATE ${DARKNET_LINK_LIBS} darknetlib)
	INSTALL (TARGETS yolo_v2_class LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)
ENDIF ()


# ==
# And lastly is the traditional darknet CLI.
# ==
ADD_EXECUTABLE (darknetcli ${APPSRC})
#SET_TARGET_PROPERTIES (darknetcli PROPERTIES OUTPUT_NAME "darknet") -- this conflicts with "darknet" the library, so instead we'll rename it when it is installed
IF (DARKNET_USE_CUDA)
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()
TARGET_LINK_LIBRARIES (darknetcli PRIVATE ${DARKNET_LINK_LIBS} darknetlib)

IF (UNIX)
	# non-Windows simple installation
	INSTALL (TARGETS darknetcli DESTINATION bin RENAME darknet)
ELSE ()
	# more complicated install for Windows so we also get the .DLL files copied over from vcpkg
	# https://stackoverflow.com/a/72088102/13022
	INSTALL (TARGETS darknetcli
		DESTINATION bin
		RENAME darknet.exe
		RUNTIME_DEPENDENCIES
			PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-" "wpaxholder" "HvsiFileTrust" "PdmUtilities"
			POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
			DIRECTORIES ${CMAKE_BINARY_DIR}/bin
		RUNTIME DESTINATION bin
		)
ENDIF ()


# We use to to have the CLI called "darknet" instead of "darknetcli".  But this conflicted with "darknet" the library
# target.  It caused strange problems on Windows where the DLL would depend on the .exe file.  To solve this we ended
# up renaming the CLI from "darknet" to "darknetcli".  However, people with a build environment from before this change
# was made (2023-12-12) might still have a copy of the _old_ CLI in their build output directory.  If this is the case,
# to prevent anyone from accidentally using this old CLI, we're going to rename it.
GET_TARGET_PROPERTY(CLI_OUTPUT_DIR darknetcli BINARY_DIR)
IF (WIN32)
	SET (OLD_CLI "${CLI_OUTPUT_DIR}/darknet.exe")
ELSE ()
	SET (OLD_CLI "${CLI_OUTPUT_DIR}/darknet")
ENDIF ()
IF (EXISTS ${OLD_CLI})
	MESSAGE(WARNING "An old copy of the CLI was found and will be renamed: ${OLD_CLI}")
	FILE(RENAME "${OLD_CLI}" "${OLD_CLI}_OLD")
ENDIF ()
