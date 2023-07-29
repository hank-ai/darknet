# Darknet object detection framework


IF (WIN32)
	ADD_COMPILE_OPTIONS (/W4)				# warning level (high)
#	ADD_COMPILE_OPTIONS (/WX)				# treat warnings as errors
	ADD_COMPILE_OPTIONS (/permissive-)		# stick to C++ standards (turn off Microsoft-specific extensions)
	ADD_COMPILE_OPTIONS (/wd4100)			# disable "unreferenced formal parameter"
	ADD_COMPILE_OPTIONS (/wd4127)			# disable "conditional expression is constant"
	ADD_COMPILE_DEFINITIONS (_CRT_SECURE_NO_WARNINGS )	# don't complain about localtime()
	ADD_COMPILE_DEFINITIONS (_USE_MATH_DEFINES)	# pull in everything we need from cmath
#	SET (CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" )
ELSE ()
	ADD_COMPILE_OPTIONS (-Wall)					# enable "all" warnings
	ADD_COMPILE_OPTIONS (-Wextra)				# enable even more warnings
	ADD_COMPILE_OPTIONS (-Wno-unused-parameter)	# don't report this error
	ADD_COMPILE_OPTIONS (-Wno-write-strings -Wno-unused-result -Wno-missing-field-initializers -Wno-ignored-qualifiers -Wno-sign-compare) # TODO clean up code so we can remove this
ENDIF ()


SET (CMAKE_ENABLE_EXPORTS TRUE)				# equivalent to -rdynamic (to get the backtrace when something goes wrong)
SET (CMAKE_POSITION_INDEPENDENT_CODE ON)	# equivalent to -fpic (position independent code)
#SET (BUILD_SHARED_LIBS TRUE)


INCLUDE_DIRECTORIES (3rdparty/stb/include/) # TODO remove


ADD_SUBDIRECTORY (src)
