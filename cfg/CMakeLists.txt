# Darknet object detection framework


FILE (GLOB CFG *.cfg)
LIST (SORT CFG)

IF (UNIX)
	INSTALL (FILES ${CFG} DESTINATION /opt/darknet/cfg/)
ELSE ()
	INSTALL (FILES ${CFG} DESTINATION cfg/)
ENDIF ()
