find_path(OPTIX_HOME include/optix.h 
    PATHS ENV OPTIX_HOME ENV OPTIX_ROOT
	DOC "Path to Optix installation.")

if(${OPTIX_HOME} STREQUAL "OptiX7_HOME-NOTFOUND")
	if (${OptiX7_FIND_REQUIRED})
        message(FATAL_ERROR "OPTIX_HOME not defined")
	elseif(NOT ${OptiX7_FIND_QUIETLY})
        message(STATUS "OPTIX_HOME not defined")
	endif()
endif()

# Include
find_path(OptiX7_INCLUDE_DIR 
	NAMES optix.h
    PATHS "${OPTIX_HOME}/include"
	NO_DEFAULT_PATH
	)
find_path(OptiX7_INCLUDE_DIR
	NAMES optix.h
	)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX7 DEFAULT_MSG 
	OptiX7_INCLUDE_DIR)

set(OptiX7_INCLUDE_DIRS ${OptiX7_INCLUDE_DIR})
if(WIN32)
	set(OptiX7_DEFINITIONS NOMINMAX)
endif()
mark_as_advanced(OptiX7_INCLUDE_DIRS OptiX7_DEFINITIONS)

add_library(OptiX7 INTERFACE)
target_compile_definitions(OptiX7 INTERFACE ${OptiX7_DEFINITIONS})
target_include_directories(OptiX7 INTERFACE ${OptiX7_INCLUDE_DIRS})
if(NOT WIN32)
    target_link_libraries(OptiX7 INTERFACE dl)
endif()

