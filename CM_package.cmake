# Darknet object detection framework


# Build an installation package.
# For example, set the generator to "DEB" or "RPM" depending on the platform you are using.
#
SET (CPACK_PACKAGE_NAME "darknet")
SET (CPACK_PACKAGE_HOMEPAGE_URL "https://darknetcv.ai/")
SET (CPACK_PACKAGE_DESCRIPTION "Darknet Object Detection Framework and YOLO")
SET (CPACK_PACKAGE_CONTACT "Stephane Charette <stephanecharette@gmail.com>")
SET (CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)


# You need to pick one package file type, and comment out the other.
# On Ubuntu, you'd typically use DEB files.  If you are on using a Linux
# distro that uses RPM files such as Centos or OpenSUSE then you'd likely
# want to use RPM.
#
# And if using Windows, then NSIS is the only option support by CPack.
#
IF (UNIX)
	SET (CPACK_GENERATOR "DEB")
	#SET (CPACK_GENERATOR "RPM")
ENDIF ()


IF (WIN32)
	SET (CPACK_GENERATOR "NSIS")
ENDIF ()


INCLUDE (CPack)
