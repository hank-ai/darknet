# Darknet object detection framework


MESSAGE(STATUS "Setting up DARKNET unit tests")

INCLUDE_DIRECTORIES (/usr/src/googletest/googletest/include/)

FILE (GLOB TEST_SOURCE *.cpp)
LIST (SORT TEST_SOURCE)

ADD_EXECUTABLE (darknet_tests ${TEST_SOURCE})
TARGET_LINK_LIBRARIES ( darknet_tests PRIVATE darknet ${DARKNET_LINK_LIBS} ${GTEST})

ADD_TEST (NAME darknet_tests COMMAND darknet_tests)
