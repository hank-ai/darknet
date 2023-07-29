# Darknet object detection framework


# Do not build in the project's root directory!  The build steps are explained in the git repo.
# See here for details:  https://github.com/hank-ai/darknet/tree/stephane-dev#building
IF (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
	MESSAGE (FATAL_ERROR  "Please create a 'build' directory.  See the build instructions for details.")
ENDIF ()


# Rename several old pre-v2.x files and directories if they exist so they don't get used by mistake.
FOREACH (filename darknet libdarknet.so obj uselib)
	IF (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
		FILE (RENAME "${CMAKE_CURRENT_SOURCE_DIR}/${filename}" "${CMAKE_CURRENT_SOURCE_DIR}/old_${filename}")
	ENDIF ()
ENDFOREACH ()
