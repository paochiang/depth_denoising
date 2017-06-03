cmake_minimum_required(VERSION 2.8)
find_package(CUDA)
CUDA_INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIR} )
LINK_DIRECTORIES( ${CUDA_LIBRARY_DIR} )

set(LIB_SRC cudaTool.cu cudaTool.h)  
cuda_add_library(cudatool STATIC ${LIB_SRC})  
#cuda_add_cufft_to_target(libcudatool)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)  
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set_target_properties(cudatool PROPERTIES OUTPUT_NAME "cudatool")  
install(TARGETS cudatool  
LIBRARY DESTINATION lib  
ARCHIVE DESTINATION lib)  
#install(FILES CudaTool.h DESTINATION include/cudatool)
