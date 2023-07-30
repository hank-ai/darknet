# Darknet object detection framework


CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/darknet_version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/darknet_version.h)

# all the source files that make up the main application
FILE (GLOB APPSRC darknet.cpp)
LIST (SORT APPSRC)

# all the source files that make up the darknet library
FILE (GLOB LIBSRC *.cpp)
LIST (REMOVE_ITEM LIBSRC APPSRC)
LIST (SORT LIBSRC)

ADD_LIBRARY (dn ${LIBSRC})
SET_TARGET_PROPERTIES (dn PROPERTIES OUTPUT_NAME "darknet")
TARGET_LINK_LIBRARIES (dn PRIVATE Threads::Threads ${OpenCV_LIBS} OpenMP::OpenMP_CXX OpenMP::OpenMP_C)

ADD_EXECUTABLE (cli ${APPSRC})
SET_TARGET_PROPERTIES (cli PROPERTIES OUTPUT_NAME "darknet")
TARGET_LINK_LIBRARIES (cli PRIVATE dn)

INSTALL (TARGETS dn DESTINATION lib)
INSTALL (TARGETS cli DESTINATION bin)
INSTALL (FILES darknet.h darknet_version.h DESTINATION include)
