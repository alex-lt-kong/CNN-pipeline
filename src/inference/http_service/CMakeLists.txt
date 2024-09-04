include(${PROJECT_SOURCE_DIR}/cmake/proto.cmake)

add_library(http_components
        controller/StaticController.hpp
        controller/SwaggerController.hpp
        dto/StatusDto.hpp
        dto/SettingsDto.hpp
        AppComponent.hpp
        swagger_component.hpp
        error_handler.cpp
        ${PROTO_SNAP_SRCS} ${PROTO_SNAP_HDRS}
        ${PROTO_INFER_SRCS} ${PROTO_INFER_HDRS})

## include directories
include_directories(/usr/local/include/torch/)
include_directories(/usr/local/cuda/include/)

target_include_directories(http_components PUBLIC swagger)


## link libs
find_package(oatpp          1.3.0 REQUIRED)
find_package(oatpp-swagger  1.3.0 REQUIRED)
find_package(OpenCV REQUIRED)

target_link_libraries(http_components
        # Oat++
        PUBLIC oatpp::oatpp
        PUBLIC oatpp::oatpp-swagger
        spdlog   
        ${OpenCV_LIBS}
        torch_cpu c10
)


add_definitions(
        ## define path to swagger-ui static resources folder
        -DOATPP_SWAGGER_RES_PATH="${oatpp-swagger_INCLUDE_DIRS}/../bin/oatpp-swagger/res"
)

if(CMAKE_SYSTEM_NAME MATCHES Linux)
    find_package(Threads REQUIRED)
    target_link_libraries(http_components INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
endif()

add_library(http_service oatpp_entry.cpp)
target_link_libraries(http_service http_components)
