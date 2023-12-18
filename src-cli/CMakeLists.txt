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

ADD_EXECUTABLE (darknetcli ${CLISRC} $<TARGET_OBJECTS:darknetobjlib>)
SET_TARGET_PROPERTIES (darknetcli PROPERTIES OUTPUT_NAME "darknet")
IF (DARKNET_USE_CUDA)
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_ARCHITECTURES "${DARKNET_CUDA_ARCHITECTURES}")
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
	SET_TARGET_PROPERTIES (darknetcli PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
ENDIF ()
TARGET_LINK_LIBRARIES (darknetcli PRIVATE ${DARKNET_LINK_LIBS})

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


# The CLI used to be called "darknet" instead of "darknetcli".  But this conflicted with "darknet" the library target.
# It caused strange problems on Windows where the DLL would depend on the .exe file.  To solve this we ended up
# renaming the CLI from "darknet" to "darknetcli".  However, people with a build environment from before this change
# was made (2023-12-17) might still have a copy of the _old_ CLI in their build output directory.  If this is the case,
# to prevent anyone from accidentally using this old CLI, we're going to rename it.
#GET_TARGET_PROPERTY(CLI_OUTPUT_DIR darknetcli BINARY_DIR)
#IF (WIN32)
#	SET (OLD_CLI "${CLI_OUTPUT_DIR}/${CMAKE_BUILD_TYPE}/darknet.exe")
#ELSE ()
#	SET (OLD_CLI "${CLI_OUTPUT_DIR}/darknet")
#ENDIF ()
#IF (EXISTS "${OLD_CLI}")
#	MESSAGE (WARNING "An old copy of the CLI was found and will be renamed: ${OLD_CLI}")
#	FILE (RENAME "${OLD_CLI}" "${OLD_CLI}_OLD")
#ENDIF ()
