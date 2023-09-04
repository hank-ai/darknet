# Darknet object detection framework

# sudo apt-get install doxygen graphviz
FIND_PACKAGE (Doxygen QUIET)

IF (NOT Doxygen_FOUND)
	MESSAGE (STATUS "Skipping Doxygen (not found)")
ELSE ()
	CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in Doxyfile)
	ADD_CUSTOM_TARGET (
		doc
		${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generating documentation with Doxygen" VERBATIM )
ENDIF ()
