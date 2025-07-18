# Darknet object detection framework


# Build an installation package.
# For example, set the generator to "DEB" or "RPM" depending on the platform you are using.
#
SET (CPACK_PACKAGE_NAME "darknet")
SET (CPACK_PACKAGE_HOMEPAGE_URL "https://darknetcv.ai/")
SET (CPACK_PACKAGE_DESCRIPTION "Darknet/YOLO Object Detection Framework")
SET (CPACK_PACKAGE_CONTACT "Stephane Charette <stephanecharette@gmail.com>")
SET (CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)

# You need to pick one package file type, and comment out the other.
# On Ubuntu, you'd typically use DEB files.  If you are on using a Linux
# distro that uses RPM files such as Centos or OpenSUSE then you'd likely
# want to use RPM.
#
# And if using Windows, then NSIS is the only option supported by CPack.
#
IF (UNIX)
	SET (CPACK_GENERATOR "DEB")
#	SET (CPACK_GENERATOR "RPM")

	SET (CPACK_DEBIAN_PACKAGE_SHLIBDEPS "ON")
ENDIF ()

IF (APPLE)
	SET (CPACK_GENERATOR "DragNDrop")
	SET (CPACK_DMG_FORMAT "UDZO")
	SET (CPACK_DMG_BACKGROUND_IMAGE "${CMAKE_CURRENT_SOURCE_DIR}/artwork/hankai_darknet.png")
ENDIF ()

IF (WIN32)
	SET (CPACK_PACKAGE_INSTALL_DIRECTORY "Darknet") # C:/Program Files/Darknet/...
	SET (CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/src-cli/windows/darknet_logo_blue.ico")
	SET (CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/src-cli/windows/darknet_logo_blue.ico")
	SET (CPACK_NSIS_DISPLAY_NAME "Darknet/YOLO Object Detection Framework")
	SET (CPACK_NSIS_PACKAGE_NAME "Darknet/YOLO Object Detection Framework")
	SET (CPACK_NSIS_MODIFY_PATH "ON")
	SET (CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL "ON")
	SET (CPACK_NSIS_CONTACT "stephanecharette@gmail.com")
	SET (CPACK_GENERATOR "NSIS")
ENDIF ()

INCLUDE (CPack)
