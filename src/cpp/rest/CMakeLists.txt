add_library(rest-lib
        controller/StaticController.hpp
        controller/SwaggerController.hpp
        dto/StatusDto.hpp
        dto/UserDto.hpp
        AppComponent.hpp
        SwaggerComponent.hpp
        ErrorHandler.cpp
        ErrorHandler.hpp)

## include directories

target_include_directories(rest-lib PUBLIC swagger)


## link libs
find_package(oatpp          1.3.0 REQUIRED)
find_package(oatpp-swagger  1.3.0 REQUIRED)

target_link_libraries(rest-lib
        # Oat++
        PUBLIC oatpp::oatpp
        PUBLIC oatpp::oatpp-swagger
        spdlog
)

# If CMake can't find SQLite3:
#
# 1. Make sure that you've built oatpp-sqlite with -DOATPP_SQLITE_AMALGAMATION=ON flag
# 2. If you are not willing to use SQLite amalgamation then uncomment the following lines:
#
#find_package(SQLite3 REQUIRED)
#
#target_link_libraries(rest-lib
#        PUBLIC SQLite::SQLite3
#)

add_definitions(
        ## define path to swagger-ui static resources folder
        -DOATPP_SWAGGER_RES_PATH="${oatpp-swagger_INCLUDE_DIRS}/../bin/oatpp-swagger/res"
)

if(CMAKE_SYSTEM_NAME MATCHES Linux)
    find_package(Threads REQUIRED)
    target_link_libraries(rest-lib INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
endif()

add_library(rest oatpp_entry.cpp)
target_link_libraries(rest rest-lib)
