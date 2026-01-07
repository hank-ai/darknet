# Clean the install prefix before packaging.
# This is intended for local packaging workflows on macOS.

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
	message(FATAL_ERROR "CMAKE_INSTALL_PREFIX is not defined.")
endif()

set(destdir "$ENV{DESTDIR}")
if(destdir)
	set(target_dir "${destdir}${CMAKE_INSTALL_PREFIX}")
else()
	set(target_dir "${CMAKE_INSTALL_PREFIX}")
endif()

# Safety guard: only allow cleaning paths that end with /opt/lib/darknet
string(REGEX MATCH ".*/opt/lib/darknet$" prefix_ok "${target_dir}")
if(NOT prefix_ok)
	message(FATAL_ERROR "Refusing to clean unexpected install prefix: ${target_dir}")
endif()

foreach(subdir bin include lib)
	set(path "${target_dir}/${subdir}")
	if(EXISTS "${path}")
		message(STATUS "Removing ${path}")
		file(REMOVE_RECURSE "${path}")
	endif()
endforeach()
