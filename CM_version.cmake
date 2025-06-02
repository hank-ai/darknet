# Darknet object detection framework


# Create a version string from the git tag and commit hash (see src/darknet_version.h.in).
# Should look similar to this:
#
#		v1.99-63-gc5c3569
#
# Set default version in case git information isn't available
SET (DARKNET_VERSION_STRING "v1.0.0")
SET (DARKNET_VERSION_SHORT "1.0.0")
SET (DARKNET_BRANCH_NAME "master")

# Try to get version from git if available
EXECUTE_PROCESS (
    COMMAND git describe --tags --dirty
    OUTPUT_VARIABLE GIT_VERSION_STRING
    RESULT_VARIABLE GIT_RESULT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

IF (GIT_RESULT EQUAL 0 AND NOT "${GIT_VERSION_STRING}" STREQUAL "")
    SET (DARKNET_VERSION_STRING ${GIT_VERSION_STRING})

    # Only try to parse the git version if we got a valid result
    STRING (REGEX MATCH "v([0-9]+)\.([0-9]+)-([0-9]+)-g([0-9a-fA-F]+)" VALID_VERSION ${DARKNET_VERSION_STRING})

    IF (VALID_VERSION)
        SET (DARKNET_VERSION_SHORT ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3})
    ENDIF()

    # Try to get branch name
    EXECUTE_PROCESS (
        COMMAND git branch --show-current
        OUTPUT_VARIABLE GIT_BRANCH_NAME
        RESULT_VARIABLE GIT_BRANCH_RESULT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    IF (GIT_BRANCH_RESULT EQUAL 0 AND NOT "${GIT_BRANCH_NAME}" STREQUAL "")
        SET (DARKNET_BRANCH_NAME ${GIT_BRANCH_NAME})
    ENDIF()
ENDIF()

MESSAGE (STATUS "Darknet ${DARKNET_VERSION_STRING}")
MESSAGE (STATUS "Darknet branch name: ${DARKNET_BRANCH_NAME}")
