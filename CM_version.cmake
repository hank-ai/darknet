# Darknet object detection framework


# Create a version string from the git tag and commit hash (see src/darknet_version.h.in).
# Should look similar to this:
#
#		v1.99-63-gc5c3569
#
EXECUTE_PROCESS (COMMAND git describe --tags --dirty --long OUTPUT_VARIABLE DARKNET_VERSION_STRING OUTPUT_STRIP_TRAILING_WHITESPACE)
MESSAGE (STATUS "Darknet ${DARKNET_VERSION_STRING}")

STRING (REGEX MATCH "v([0-9]+)\.([0-9]+)\.([0-9]+)-([0-9]+)-g([0-9a-fA-F]+)" _ ${DARKNET_VERSION_STRING})
# note that MATCH_5 is not numeric

IF (CMAKE_MATCH_1 AND CMAKE_MATCH_2 AND CMAKE_MATCH_3)
	SET (DARKNET_VERSION_SHORT ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3})
ELSE()
	SET (DARKNET_VERSION_SHORT "1.0.0")
ENDIF()

EXECUTE_PROCESS (COMMAND git branch --show-current OUTPUT_VARIABLE DARKNET_BRANCH_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
MESSAGE (STATUS "Darknet branch name: ${DARKNET_BRANCH_NAME}")
