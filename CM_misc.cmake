# Darknet object detection framework


# Do not build in the project's root directory!  The build steps are explained in the git repo.
# See here for details:  https://github.com/hank-ai/darknet#building
IF (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
	MESSAGE (FATAL_ERROR  "Please create a 'build' directory.  See the build instructions for details.")
ENDIF ()


# Rename several old pre-v2.x files and directories if they exist so they don't get used by mistake.
FOREACH (filename darknet libdarknet.so obj uselib build.log build_release build_debug vcpkg_installed CMakeFiles)
	IF (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
		FILE (RENAME "${CMAKE_CURRENT_SOURCE_DIR}/${filename}" "${CMAKE_CURRENT_SOURCE_DIR}/old_${filename}")
	ENDIF ()
ENDFOREACH ()


# The old tutorials for previous versions of darknet would tell users to copy files to /usr/local/...  If those files
# still exist, we'll end up picking up the old header and .so file which will cause all sorts of strange problems.
IF (UNIX)
	IF (EXISTS /usr/local/lib/libdarknet.so)
		MESSAGE (FATAL_ERROR "Looks like an older version of libdarknet.so is installed in /usr/local/lib/. Please delete, move, or rename this file.")
	ENDIF ()
	IF (EXISTS /usr/local/include/darknet.h)
		MESSAGE (FATAL_ERROR "Looks like an older version of darknet.h is installed in /usr/local/include/. Please delete, move, or rename this file.")
	ENDIF ()
ENDIF ()
