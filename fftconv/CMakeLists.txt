cmake_minimum_required(VERSION 3.10)

# Project definition, this sets the variable 
# ${CMAKE_PROJECT_NAME} and ${PROJECT_VERSION}
project(fftconv
		VERSION 1.51)

# Sets up the compile flags  
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)

# Search for dependencies
find_package(PkgConfig)

pkg_check_modules(FFTW REQUIRED fftw3>=3.0)

###################################
# Set up specific C++ flags
##################################
# Cmake 3.10 supports the standard up to c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

##################################
# Project hierarchy
##################################
add_subdirectory(include)
add_subdirectory(examples)

# enable_testing()
# add_subdirectory(tests)

##################################
# Pkg-config generation
##################################

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.pc
	"
Name: ${CMAKE_PROJECT_NAME}
Description: An efficient C++ implementation of convolution using the FFT
Version: ${PROJECT_VERSION}
Requires: fftw3 >= 3.0
Libs: 
Cflags: -I${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME} -std=c++17
	"
	)


install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.pc
	DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)

###################################
# dist target
###################################
SET(DIST_DIR "${CMAKE_PROJECT_NAME}")
ADD_CUSTOM_TARGET(dist 
	COMMAND rm -rf ${DIST_DIR}
	COMMAND mkdir  ${DIST_DIR}
	COMMAND cp -r ${CMAKE_SOURCE_DIR}/* ${DIST_DIR} || true 
	COMMAND rm -rf ${DIST_DIR}/build
	COMMAND mkdir ${DIST_DIR}/build
	COMMAND tar --exclude="*~" --exclude="._*" -zcvf ${DIST_DIR}-${PROJECT_VERSION}.tar.gz ${DIST_DIR}
	COMMAND rm -rf  ${DIST_DIR}
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

###################################
# dist target
###################################
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++ application for demonstrating Kmeans clustering")
SET(CPACK_PACKAGE_VENDOR "Jeremy Fix")
SET(CPACK_PACKAGE_CONTACT "{Jeremy_DOT_fix_AT_centralesupelec.fr")
SET(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
SET(CPACK_PACKAGE_LICENSE "GPL")
SET(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_SOURCE_DIR}/../LICENSE)
SET(CPACK_RESOURCE_FILE_README ${CMAKE_SOURCE_DIR}/../README.md)
SET(CPACK_RESOURCE_FILE_WELCOME ${CMAKE_SOURCE_DIR}/../README.md)

ADD_CUSTOM_TARGET(packages
		  COMMAND rm -rf build
		  COMMAND mkdir build
		  COMMAND cd build && cmake ${CMAKE_SOURCE_DIR} -DCMAKE_INSTALL_PREFIX=/usr && make package -j
		  COMMAND cp build/*.deb . || true
		  COMMAND rm -rf build
          WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

###############################################
# For DEBs
###############################################
find_program( DPKGDEB
	 NAMES dpkg-deb
	 PATHS "/usr/bin"
)
IF(NOT DPKGDEB STREQUAL  "DPKGDEB-NOTFOUND")
  MESSAGE("Set up for building DEB")

  SET(CPACK_DEBIAN_PACKAGE_NAME ${CMAKE_PROJECT_NAME}-dev)
  SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libfftw3-dev (>=  3.0)")

  SET(CPACK_PACKAGE_FILE_NAME ${CPACK_DEBIAN_PACKAGE_NAME}-${PROJECT_VERSION}-${CMAKE_SYSTEM_PROCESSOR})

  SET(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT})
  SET(CPACK_GENERATOR "DEB")
ENDIF(NOT DPKGDEB STREQUAL  "DPKGDEB-NOTFOUND")

###############################################
INCLUDE(CPack)
