# Apple backend source ownership stays here so the generic Darknet build only
# needs a single guarded include.
SET (DARKNET_APPLE_OBJCXX_SOURCES
	apple/apple_backend.mm
	apple/apple_training_buffers.mm
	apple/apple_training_facade.mm
	apple/apple_training_graph.mm
	apple/apple_training_optimizer.mm
	apple/apple_training_session.mm
	apple/apple_training_yolo.mm
	apple/apple_mps.mm
	apple/metal_backend.mm)

SET (DARKNET_APPLE_SOURCES
	${DARKNET_APPLE_OBJCXX_SOURCES}
	apple/apple_training_plan.cpp)

SET_SOURCE_FILES_PROPERTIES (${DARKNET_APPLE_SOURCES} PROPERTIES
	COMPILE_OPTIONS "-Werror=deprecated-declarations;-Werror=unguarded-availability")
SET_SOURCE_FILES_PROPERTIES (${DARKNET_APPLE_OBJCXX_SOURCES} PROPERTIES COMPILE_FLAGS "-fobjc-arc")
