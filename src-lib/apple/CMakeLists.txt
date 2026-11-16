# Apple backend source ownership stays here. The generic Darknet build adds
# this directory only when MPS is enabled.
SET (DARKNET_APPLE_OBJCXX_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/apple_backend.mm
	${CMAKE_CURRENT_SOURCE_DIR}/apple_training_buffers.mm
	${CMAKE_CURRENT_SOURCE_DIR}/apple_training_facade.mm
	${CMAKE_CURRENT_SOURCE_DIR}/apple_training_graph.mm
	${CMAKE_CURRENT_SOURCE_DIR}/apple_training_optimizer.mm
	${CMAKE_CURRENT_SOURCE_DIR}/apple_training_session.mm
	${CMAKE_CURRENT_SOURCE_DIR}/apple_training_yolo.mm
	${CMAKE_CURRENT_SOURCE_DIR}/apple_mps.mm
	${CMAKE_CURRENT_SOURCE_DIR}/metal_backend.mm)

SET (DARKNET_APPLE_SOURCES
	${DARKNET_APPLE_OBJCXX_SOURCES}
	${CMAKE_CURRENT_SOURCE_DIR}/apple_training_plan.cpp)

TARGET_SOURCES (darknetobjlib PRIVATE ${DARKNET_APPLE_SOURCES})

# These sources compile in the parent target, so set properties in its directory.
SET_SOURCE_FILES_PROPERTIES (${DARKNET_APPLE_SOURCES}
	TARGET_DIRECTORY darknetobjlib PROPERTIES
	COMPILE_OPTIONS "-Werror=deprecated-declarations;-Werror=unguarded-availability")
SET_SOURCE_FILES_PROPERTIES (${DARKNET_APPLE_OBJCXX_SOURCES}
	TARGET_DIRECTORY darknetobjlib PROPERTIES COMPILE_FLAGS "-fobjc-arc")
