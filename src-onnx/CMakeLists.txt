# Darknet object detection framework


MESSAGE(STATUS "Setting up DARKNET ONNX export tool")

IF (NOT Protobuf_FOUND)
	MESSAGE(FATAL_ERROR "The Darknet/YOLO ONNX export tool requires support for protocol buffers.")
ENDIF ()

FILE (GLOB SRC_ONNX *.cpp *.cc)
LIST (SORT SRC_ONNX)

ADD_EXECUTABLE (darknet_onnx_export ${SRC_ONNX})
TARGET_LINK_LIBRARIES (darknet_onnx_export PRIVATE darknet ${DARKNET_LINK_LIBS} ${Protobuf_LIBRARIES})
INSTALL (TARGETS darknet_onnx_export DESTINATION bin)
