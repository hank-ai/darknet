# Darknet object detection framework


# Build an installation package.
# For example, set the generator to "DEB" or "RPM" depending on the platform you are using.
#
SET (CPACK_GENERATOR "DEB")
#SET (CPACK_GENERATOR "RPM")
#SET (CPACK_GENERATOR "NSIS")
SET (CPACK_PACKAGE_NAME "darknet")
SET (CPACK_PACKAGE_HOMEPAGE_URL "https://darknetcv.ai/")
SET (CPACK_PACKAGE_DESCRIPTION "Darknet Object Detection Framework and YOLO")
SET (CPACK_PACKAGE_CONTACT "Stephane Charette <stephanecharette@gmail.com>")
SET (CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)

INCLUDE (CPack)
