# Darknet object detection framework


MESSAGE(STATUS "Setting up DARKNET ONNX export tool")

IF (NOT Protobuf_FOUND)
	MESSAGE(FATAL_ERROR "The Darknet/YOLO ONNX export tool requires support for protocol buffers.")
ENDIF ()

ADD_CUSTOM_COMMAND(
	OUTPUT	${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto3.pb.cc
			${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto3.pb.h
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	COMMAND ${Protobuf_PROTOC_EXECUTABLE} --fatal_warnings --proto_path=${CMAKE_CURRENT_SOURCE_DIR} --cpp_out=${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto3
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto3
	COMMENT "Generating Protocol Buffer output files for ONNX...")
ADD_CUSTOM_TARGET (darknet_generated_proto DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto3.pb.cc ${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto3.pb.h)

FILE (GLOB SRC_ONNX *.cpp) # note the ONNX .cc file doesn't yet exist when this line runs
LIST (SORT SRC_ONNX)

ADD_EXECUTABLE (darknet_onnx_export ${SRC_ONNX} onnx.proto3.pb.cc)
ADD_DEPENDENCIES (darknet_onnx_export darknet_generated_proto)
TARGET_LINK_LIBRARIES (darknet_onnx_export PRIVATE darknet ${DARKNET_LINK_LIBS} protobuf::libprotobuf)
INSTALL (TARGETS darknet_onnx_export DESTINATION bin)
