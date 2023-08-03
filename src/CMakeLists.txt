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

# create 2 outputs:  the darknet library, and the ususual CLI
ADD_LIBRARY (dnlib SHARED $<TARGET_OBJECTS:objlib>)
SET_TARGET_PROPERTIES (dnlib PROPERTIES OUTPUT_NAME "darknet")
SET_TARGET_PROPERTIES (dnlib PROPERTIES PUBLIC_HEADER "darknet.h")
TARGET_LINK_LIBRARIES (dnlib PRIVATE Threads::Threads ${OpenCV_LIBS} ${OpenCV_LINKED_COMPONENTS} OpenMP::OpenMP_CXX OpenMP::OpenMP_C PThreads4W::PThreads4W)

ADD_EXECUTABLE (dncli ${APPSRC} $<TARGET_OBJECTS:objlib>)
SET_TARGET_PROPERTIES (dncli PROPERTIES OUTPUT_NAME "darknet")
TARGET_LINK_LIBRARIES (dncli PRIVATE Threads::Threads ${OpenCV_LIBS} ${OpenCV_LINKED_COMPONENTS} OpenMP::OpenMP_CXX OpenMP::OpenMP_C PThreads4W::PThreads4W)

INSTALL (TARGETS dnlib LIBRARY DESTINATION lib)
INSTALL (TARGETS dncli DESTINATION bin)
INSTALL (FILES darknet.h darknet_version.h DESTINATION include)
