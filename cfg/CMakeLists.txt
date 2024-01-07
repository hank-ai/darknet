# Darknet object detection framework


FILE (GLOB CFG *.cfg *.names)

IF (UNIX)
	INSTALL (FILES ${CFG} DESTINATION /opt/darknet/cfg/)
ELSE ()
	INSTALL (FILES ${CFG} DESTINATION cfg/)
ENDIF ()
