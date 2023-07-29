# Darknet object detection framework


FILE (GLOB CFG *.cfg)
LIST (SORT CFG)

INSTALL (FILES ${CFG} DESTINATION /opt/darknet/cfg/)
