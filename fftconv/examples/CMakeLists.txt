file(
	GLOB 
	usage_examples
	example*.cpp
)

# For every example
foreach(f ${usage_examples})
  # We build the name of the target by removing the extension ".cpp" 
  get_filename_component(exampleName ${f} NAME_WE) 

  # Actually define the target
  add_executable (${exampleName} ${f}) 

  # Define the include path to the headers or our package
  target_include_directories(${exampleName} PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(${exampleName} PUBLIC ${FFTW_INCLUDE_DIRS})
  target_link_libraries(${exampleName} ${FFTW_LIBRARIES})


  # And we install it in the "${CMAKE_PREFIX}/bin" directory
  # We add the prefix of the project name to avoid collision
  # of programs from other packages
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${exampleName}
    DESTINATION bin
    RENAME ${CMAKE_PROJECT_NAME}-${exampleName})

endforeach(f)
