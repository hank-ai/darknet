# Darknet object detection framework


CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/darknet_version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/darknet_version.h)

# all the source files that make up the main application
FILE (GLOB APPSRC darknet.cpp)
LIST (SORT APPSRC)

# all the source files that make up the darknet library
FILE (GLOB LIBSRC *.cpp)
LIST (REMOVE_ITEM LIBSRC ${APPSRC})
LIST (SORT LIBSRC)

IF (WIN32)
	INCLUDE_DIRECTORIES (windows)
	FILE (GLOB WINSRC windows/*.c)
	LIST (APPEND LIBSRC ${WINSRC})
ENDIF ()

# compile everything once
ADD_LIBRARY(objlib OBJECT ${LIBSRC})

# create 2 outputs:  the darknet library, and the ususual darknet CLI
ADD_LIBRARY (darknetlib SHARED $<TARGET_OBJECTS:objlib>)
SET_TARGET_PROPERTIES (darknetlib PROPERTIES OUTPUT_NAME "darknet")
SET_TARGET_PROPERTIES (darknetlib PROPERTIES PUBLIC_HEADER "darknet.h;darknet_version.h")
TARGET_LINK_LIBRARIES (darknetlib PRIVATE ${DARKNET_LINK_LIBS})

ADD_EXECUTABLE (darknetcli ${APPSRC})
SET_TARGET_PROPERTIES (darknetcli PROPERTIES OUTPUT_NAME "darknet")
TARGET_LINK_LIBRARIES (darknetcli PRIVATE darknetlib ${DARKNET_LINK_LIBS})

INSTALL (TARGETS darknetlib LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)
INSTALL (TARGETS darknetcli DESTINATION bin)
