# Darknet object detection framework


IF (WIN32)
#	ADD_COMPILE_OPTIONS (/W4)				# warning level (high)
#	ADD_COMPILE_OPTIONS (/WX)				# treat warnings as errors
#	ADD_COMPILE_OPTIONS (/permissive-)		# stick to C++ standards (turn off Microsoft-specific extensions)
#	ADD_COMPILE_OPTIONS (/Zc:__cplusplus)	# force Visual Studio to update __cplusplus (but this seems to break nvcc.exe)
#	ADD_COMPILE_OPTIONS (/wd4013)
#	ADD_COMPILE_OPTIONS (/wd4018)
#	ADD_COMPILE_OPTIONS (/wd4028)
#	ADD_COMPILE_OPTIONS (/wd4047)
#	ADD_COMPILE_OPTIONS (/wd4068)
#	ADD_COMPILE_OPTIONS (/wd4090)
#	ADD_COMPILE_OPTIONS (/wd4100)			# disable "unreferenced formal parameter"
#	ADD_COMPILE_OPTIONS (/wd4101)
#	ADD_COMPILE_OPTIONS (/wd4113)
#	ADD_COMPILE_OPTIONS (/wd4127)			# disable "conditional expression is constant"
#	ADD_COMPILE_OPTIONS (/wd4133)
#	ADD_COMPILE_OPTIONS (/wd4190)
#	ADD_COMPILE_OPTIONS (/wd4244)
#	ADD_COMPILE_OPTIONS (/wd4267)
#	ADD_COMPILE_OPTIONS (/wd4305)
#	ADD_COMPILE_OPTIONS (/wd4477)
#	ADD_COMPILE_OPTIONS (/wd4819)
#	ADD_COMPILE_OPTIONS (/wd4996)
#	ADD_COMPILE_OPTIONS (/Zc:strictStrings-)
	ADD_COMPILE_DEFINITIONS (LIB_EXPORTS)
	ADD_COMPILE_DEFINITIONS (NOMINMAX)
	ADD_COMPILE_DEFINITIONS (_CRT_SECURE_NO_WARNINGS )	# don't complain about localtime()

	# With old compilers (or Windows only?) it used to be necessary to define this prior to #including cmath.
	# Not sure if (or why?) this still seems to be required with Visual Studio.
	ADD_COMPILE_DEFINITIONS (_USE_MATH_DEFINES)
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


# TODO: https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior?view=msvc-170
# TODO: https://stackoverflow.com/questions/36501542/what-is-gcc-clang-equivalent-of-fp-model-fast-1-in-icc
# TODO: -ffast-math and -funsafe-math-optimizations


SET (BUILD_SHARED_LIBS TRUE)				# ADD_LIBRARY() will default to shared libs
SET (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)	# automatically create a module definition (.def) file with all global symbols
SET (CMAKE_ENABLE_EXPORTS TRUE)				# equivalent to -rdynamic (to get the backtrace when something goes wrong)
SET (CMAKE_POSITION_INDEPENDENT_CODE ON)	# equivalent to -fpic (position independent code)


ADD_SUBDIRECTORY (doc)
ADD_SUBDIRECTORY (cfg)
ADD_SUBDIRECTORY (src)
