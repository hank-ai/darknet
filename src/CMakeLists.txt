# Darknet object detection framework


CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/darknet_version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/darknet_version.h)

FILE (GLOB SRC *.cpp)
LIST (SORT SRC)

ADD_EXECUTABLE (darknet ${SRC})
SET_TARGET_PROPERTIES (darknet PROPERTIES OUTPUT_NAME "darknet" )
TARGET_LINK_LIBRARIES (darknet PRIVATE Threads::Threads ${OpenCV_LIBS} OpenMP::OpenMP_CXX OpenMP::OpenMP_C)

INSTALL (TARGETS darknet DESTINATION bin)
INSTALL (FILES darknet.h darknet_version.h DESTINATION include)
