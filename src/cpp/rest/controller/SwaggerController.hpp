
#ifndef UserController_hpp
#define UserController_hpp

//#include "../dto/PageDto.hpp"
#include "../dto/RespDto.hpp"
#include "../dto/StatusDto.hpp"
#include "../dto/UserDto.hpp"

#include "oatpp/core/macro/component.hpp"
#include "oatpp/core/utils/ConversionUtils.hpp"
#include "oatpp/web/protocol/http/Http.hpp"
//#include "service/UserService.hpp"

#include "oatpp/core/macro/codegen.hpp"
#include "oatpp/parser/json/mapping/ObjectMapper.hpp"
#include "oatpp/web/server/api/ApiController.hpp"

#include OATPP_CODEGEN_BEGIN(ApiController) //<- Begin Codegen

class SwaggerController : public oatpp::web::server::api::ApiController {
public:
  SwaggerController(OATPP_COMPONENT(std::shared_ptr<ObjectMapper>,
                                    objectMapper))
      : oatpp::web::server::api::ApiController(objectMapper) {}

public:
  static std::shared_ptr<SwaggerController> createShared(OATPP_COMPONENT(
      std::shared_ptr<ObjectMapper>,
      objectMapper) // Inject objectMapper component here as default parameter
  ) {
    return std::make_shared<SwaggerController>(objectMapper);
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

  ENDPOINT_INFO(setPredictionInterval) {
    info->summary = "Set prediction interval";

    info->addResponse<Object<RespDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<RespDto>>(Status::CODE_400, "application/json");

    info->pathParams["predictionInterval"].description =
        "Interval in ms for the prediction loop to sleep after each iteration";
  }
  ENDPOINT("POST", "model/{predictionInterval}", setPredictionInterval,
           PATH(UInt32, predictionInterval)) {
    auto resp = RespDto::createShared();
    if (predictionInterval <= 0) {
      resp->success = false;
      resp->responseText = "predictionInterval must be positive";
      return createDtoResponse(Status::CODE_400, resp);
    } else {
      resp->success = true;
      resp->responseText =
          "predictionInterval set to " +
          oatpp::utils::conversion::uint32ToStr(predictionInterval);
      return createDtoResponse(Status::CODE_200, resp);
    }
  }
};

#include OATPP_CODEGEN_END(ApiController) //<- End Codegen

#endif /* UserController_hpp */
