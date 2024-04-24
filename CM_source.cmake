# Darknet object detection framework


# The old "src" directory was split into "src-lib" and "src-cli" in December 2023.
# Rename that old dirctory, otherwise people will be confused if source files show
# up in multiple subdirectories.
IF (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src")
	FILE (RENAME "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/delete_me_old_src")
ENDIF ()
IF (EXISTS "${CMAKE_CURRENT_BINARY_DIR}/src")
	FILE (RENAME "${CMAKE_CURRENT_BINARY_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/delete_me_old_src")
ENDIF ()


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
	ADD_COMPILE_DEFINITIONS (NOMINMAX)
	ADD_COMPILE_DEFINITIONS (_CRT_SECURE_NO_WARNINGS)	# don't complain about localtime()

	# With old compilers (or Windows only?) it used to be necessary to define this prior to #including cmath.
	# Not sure if (or why?) this still seems to be required with Visual Studio.
	ADD_COMPILE_DEFINITIONS (_USE_MATH_DEFINES)
ENDIF ()


IF (UNIX)
	# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

	ADD_COMPILE_OPTIONS (-Wall)					# enable "all" warnings
	ADD_COMPILE_OPTIONS (-Wextra)				# enable even more warnings
	ADD_COMPILE_OPTIONS (-Wno-unused-parameter)	# don't report this error

	IF (CMAKE_BUILD_TYPE MATCHES DEBUG OR
		CMAKE_BUILD_TYPE MATCHES Debug OR
		CMAKE_BUILD_TYPE MATCHES debug)
		MESSAGE (WARNING "Making a DEBUG build.")
		ADD_COMPILE_OPTIONS (-O0)				# turn off optimizations
		ADD_COMPILE_OPTIONS (-ggdb)				# turn on GDB info
		ADD_COMPILE_DEFINITIONS (DEBUG)
	ELSE ()
		MESSAGE (STATUS "Making an optimized release build.")
		ADD_COMPILE_OPTIONS (-O3)				# turn on optimizations

		# this breaks the windows build, so even though it shouldn't be a
		# linux-only optimization, we only set this for UNIX-type builds
		SET (CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # enable link-time optimization for all targets (e.g., -flto)
	ENDIF ()

	# nvcc incorrectly fails to parse this flag as it expects a number to come after -O
	# instead, see where this is set for specific source files in src-lib/CMakeLists.txt
#	ADD_COMPILE_OPTIONS (-Ofast)				# turn on optimizations for speed

	# this cannot be used since OpenCV has places where doubles are used and they cannot (should not!) be converted to float
#	ADD_COMPILE_OPTIONS (-fsingle-precision-constant)	# treat floating-point constants as single precision instead of implicitly converting them to double-precision constants

	# TODO Investigate these options further.  When I tried them in early April 2024, they did not seem to have a huge
	# impact on training speed.  I enabled them individually, and recorded the time it took to train a specific simple
	# network I had on hand.
	#
	# 2024-04-24:  For now, I've enabled just unsafe-math-optimizations.  I've not found any combination of options which
	# gives better results.  Note that enabling some or all of these options may cause issues with training neural networks.
	#
														# [8:00] no optimization options enabled
														# [7:53] all optimizations enabled
#	ADD_COMPILE_OPTIONS (-ffast-math)					# [7:55] this option can result in incorrect output for programs that depend on an exact implementation of IEEE or ISO rules/specifications for math functions.
#	ADD_COMPILE_OPTIONS (-frename-registers)			# [7:55] attempt to avoid false dependencies in scheduled code by making use of registers left over after register allocation
#	ADD_COMPILE_OPTIONS (-ffinite-math-only)			# [7:53] allow optimizations for floating-point arithmetic that assume that arguments and results are not NaNs or +-Infs
#	ADD_COMPILE_OPTIONS (-funroll-loops)				# [7:53] unroll loops whose number of iterations can be determined at compile time or upon entry to the loop
#	ADD_COMPILE_OPTIONS (-fno-signed-zeros)				# [7:57] allow optimizations for floating-point arithmetic that ignore the signedness of zero
#	ADD_COMPILE_OPTIONS (-freciprocal-math)				# [7:54] allow the reciprocal of a value to be used instead of dividing by the value if this enables optimizations
#	ADD_COMPILE_OPTIONS (-fno-math-errno)				# [7:53] do not set errno after calling math functions that are executed with a single instruction [...] it can result in incorrect output for programs that depend on an exact implementation of IEEE or ISO rules/specifications for math functions
#	ADD_COMPILE_OPTIONS (-fassociative-math)			# [7:52] allow re-association of operands in series of floating-point operations
#	ADD_COMPILE_OPTIONS (-fno-trapping-math)			# [7:52] compile code assuming that floating-point operations cannot generate user-visible traps
	ADD_COMPILE_OPTIONS (-funsafe-math-optimizations)	# [7:49] allow optimizations for floating-point arithmetic that (a) assume that arguments and results are valid and (b) may violate IEEE or ANSI standards.

	# TODO remove the following options and clean up the code instead of ignoring the problem
	ADD_COMPILE_OPTIONS (-Wno-write-strings)
	ADD_COMPILE_OPTIONS (-Wno-unused-result)
	ADD_COMPILE_OPTIONS (-Wno-missing-field-initializers)
	ADD_COMPILE_OPTIONS (-Wno-ignored-qualifiers)
	ADD_COMPILE_OPTIONS (-Wno-sign-compare)
ENDIF ()

# TODO: https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior?view=msvc-170
# TODO: https://stackoverflow.com/questions/36501542/what-is-gcc-clang-equivalent-of-fp-model-fast-1-in-icc

SET (BUILD_SHARED_LIBS TRUE)					# ADD_LIBRARY() will default to shared libs
SET (CMAKE_ENABLE_EXPORTS TRUE)					# equivalent to -rdynamic (to get the backtrace when something goes wrong)
SET (CMAKE_OPTIMIZE_DEPENDENCIES TRUE)			# some dependencies may be removed if they are not necessary to build the library
SET (CMAKE_POSITION_INDEPENDENT_CODE TRUE)		# equivalent to -fpic (position independent code)

INCLUDE_DIRECTORIES (src-cli)
INCLUDE_DIRECTORIES (src-lib)

ADD_SUBDIRECTORY (doc)
ADD_SUBDIRECTORY (cfg)
ADD_SUBDIRECTORY (src-lib)
ADD_SUBDIRECTORY (src-cli)
