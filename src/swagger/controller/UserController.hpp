
#ifndef UserController_hpp
#define UserController_hpp

#include "../dto/UserDto.hpp"
//#include "db/UserDb.hpp"
#include "../dto/PageDto.hpp"
#include "../dto/StatusDto.hpp"

#include "oatpp/core/macro/component.hpp"
#include "oatpp/web/protocol/http/Http.hpp"
//#include "service/UserService.hpp"

#include "oatpp/core/macro/codegen.hpp"
#include "oatpp/parser/json/mapping/ObjectMapper.hpp"
#include "oatpp/web/server/api/ApiController.hpp"

#include OATPP_CODEGEN_BEGIN(ApiController) //<- Begin Codegen

/**
 * User REST controller.
 */
class UserController : public oatpp::web::server::api::ApiController {
public:
  UserController(OATPP_COMPONENT(std::shared_ptr<ObjectMapper>, objectMapper))
      : oatpp::web::server::api::ApiController(objectMapper) {}
  // private:
  //   UserService m_userService; // Create user service.
public:
  static std::shared_ptr<UserController> createShared(OATPP_COMPONENT(
      std::shared_ptr<ObjectMapper>,
      objectMapper) // Inject objectMapper component here as default parameter
  ) {
    return std::make_shared<UserController>(objectMapper);
  }

  ENDPOINT_INFO(getUserById) {
    info->summary = "Get one User by userId";

    info->addResponse<Object<UserDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["userId"].description = "User Identifier";
  }
  ENDPOINT("GET", "users/{userId}", getUserById, PATH(Int32, userId)) {
    if (userId != 1234) {
      auto userDto = UserDto::createShared();
      userDto->id = userId;
      userDto->userName = "DummyUserName";
      userDto->email = "DummyEmail";
      // oatpp::Object<UserDto> t = userDto.createShared();

      return createDtoResponse(Status::CODE_200, userDto);
    } else {
      auto err = StatusDto::createShared();
      err->status = "error";
      err->code = 500;
      err->message = "1234 not allowed";
      // oatpp::Object<UserDto> t = userDto.createShared();

      return createDtoResponse(Status::CODE_404, err);
    }
    // return createDtoResponse(Status::CODE_200,
    // m_userService.getUserById(userId));
  }
};

#include OATPP_CODEGEN_END(ApiController) //<- End Codegen

#endif /* UserController_hpp */