# Darknet object detection framework


IF (WIN32)
	ADD_COMPILE_OPTIONS (/W4)				# warning level (high)
#	ADD_COMPILE_OPTIONS (/WX)				# treat warnings as errors
	ADD_COMPILE_OPTIONS (/permissive-)		# stick to C++ standards (turn off Microsoft-specific extensions)
	ADD_COMPILE_OPTIONS (/wd4100)			# disable "unreferenced formal parameter"
	ADD_COMPILE_OPTIONS (/wd4127)			# disable "conditional expression is constant"
#	ADD_COMPILE_DEFINITIONS (_CRT_SECURE_NO_WARNINGS )	# don't complain about localtime()
#	SET (CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" )
ENDIF ()

IF (UNIX)
	ADD_COMPILE_OPTIONS (-Wall)					# enable "all" warnings
	ADD_COMPILE_OPTIONS (-Wextra)				# enable even more warnings
	ADD_COMPILE_OPTIONS (-Wno-unused-parameter)	# don't report this error

	# TODO remove the following options and clean up the code instead of ignoring the problem
	ADD_COMPILE_OPTIONS (-Wno-write-strings)
	ADD_COMPILE_OPTIONS (-Wno-unused-result)
	ADD_COMPILE_OPTIONS (-Wno-missing-field-initializers)
	ADD_COMPILE_OPTIONS (-Wno-ignored-qualifiers)
	ADD_COMPILE_OPTIONS (-Wno-sign-compare)
ENDIF ()


# With old compilers (or Windows only?) it used to be necessary to define
# this prior to #including cmath.  Not sure if this is still required.
ADD_COMPILE_DEFINITIONS (_USE_MATH_DEFINES)


SET (BUILD_SHARED_LIBS TRUE)
SET (CMAKE_ENABLE_EXPORTS TRUE)				# equivalent to -rdynamic (to get the backtrace when something goes wrong)
SET (CMAKE_POSITION_INDEPENDENT_CODE ON)	# equivalent to -fpic (position independent code)


INCLUDE_DIRECTORIES (3rdparty/stb/include/) # TODO remove


ADD_SUBDIRECTORY (doc)
ADD_SUBDIRECTORY (cfg)
ADD_SUBDIRECTORY (src)
