# Darknet object detection framework


FILE (GLOB CFG *.cfg *.names)

if(${INSTALL_CFG_PREFIX} STREQUAL "")
	INSTALL (FILES ${CFG} DESTINATION cfg/)
else()
	INSTALL (FILES ${CFG} DESTINATION ${INSTALL_CFG_PREFIX}/cfg/)
endif()