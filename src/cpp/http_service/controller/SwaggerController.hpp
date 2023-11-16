#ifndef SwaggerController_hpp
#define SwaggerController_hpp

#define FMT_HEADER_ONLY

#include "../../global_vars.h"
#include "../../model_utils.h"
#include "../dto/InternalStateDto.hpp"
#include "../dto/ModelInfoDto.hpp"
#include "../dto/RespDto.hpp"
#include "../dto/SettingsDto.hpp"
#include "../dto/StatusDto.hpp"

#include <oatpp/core/macro/codegen.hpp>
#include <oatpp/core/macro/component.hpp>
#include <oatpp/core/utils/ConversionUtils.hpp>
#include <oatpp/parser/json/mapping/ObjectMapper.hpp>
#include <oatpp/web/protocol/http/Http.hpp>
#include <oatpp/web/server/api/ApiController.hpp>
#include <spdlog/spdlog.h>

#include <math.h>
#include <mutex>
#include <regex>

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

  ENDPOINT_INFO(reloadModels) {
    info->summary = "reload models from given model IDs";

    info->addConsumes<Object<ModelInfoDto>>("application/json");
    info->addResponse<Object<RespDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<RespDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<RespDto>>(Status::CODE_500, "application/json");
  }
  ENDPOINT("POST", "reloadModels", reloadModels,
           BODY_DTO(Object<ModelInfoDto>, modelInfo)) {
    std::vector<std::string> _model_ids;
    auto resp = RespDto::createShared();
    Status code;
    if ((*modelInfo->modelIds).size() == 0) {
      resp->success = false;
      resp->responseText = "modelIds canNOT be empty!!";
      code = Status::CODE_400;
    }
    for (const auto &ele : *modelInfo->modelIds) {
      _model_ids.push_back(ele);
    }
    try {
      auto _models = load_models(_model_ids, modelInfo->cudaDevice);
      {
        std::scoped_lock lck{models_mtx, model_ids_mtx};
        models = std::move(_models);
        model_ids = std::move(_model_ids);
      }
      resp->success = true;
      resp->responseText = "Models loaded";
      code = Status::CODE_200;
    } catch (const c10::Error &e) {
      spdlog::error("Error loading the model: {}", e.what());
      resp->success = false;
      resp->responseText = std::string("Model not reloaded: ") + e.what();
      code = Status::CODE_400;
    }
    return createDtoResponse(code, resp);
  }

  ENDPOINT_INFO(setPredictionInterval) {
    info->summary = "Set prediction interval";

    info->addResponse<Object<RespDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<RespDto>>(Status::CODE_400, "application/json");

    info->pathParams["predictionInterval"].description =
        "Interval in ms for the prediction loop to sleep after each iteration";
  }
  ENDPOINT("POST", "setPredictionInterval/{predictionIntervalMs}",
           setPredictionInterval, PATH(UInt32, predictionIntervalMs)) {
    auto resp = RespDto::createShared();
    if (predictionIntervalMs <= 0) {
      resp->success = false;
      resp->responseText = "predictionInterval must be positive";
      return createDtoResponse(Status::CODE_400, resp);
    } else {
      std::string msg =
          "predictionInterval changed from " +
          to_string(inference_interval_ms) + " to " +
          oatpp::utils::conversion::uint32ToStr(predictionIntervalMs);
      inference_interval_ms = predictionIntervalMs;
      spdlog::info(msg);
      resp->success = true;
      resp->responseText = msg;
      return createDtoResponse(Status::CODE_200, resp);
    }
  }

  ENDPOINT_INFO(getSettings) {
    info->summary =
        "get the content of config.json of this inference daemon instance";
  }
  ENDPOINT("GET", "getSettings/", getSettings) {
    auto jsonObjectMapper =
        oatpp::parser::json::mapping::ObjectMapper::createShared();
    auto settingsOatppJson =
        jsonObjectMapper->readFromString<oatpp::Any>(settings.dump());
    return createDtoResponse(Status::CODE_200, settingsOatppJson);
  }

  ENDPOINT_INFO(getInternalState) {
    info->summary = "get the internal state of the prediction daemon";

    info->addResponse<Object<InternalStateDto>>(Status::CODE_200,
                                                "application/json");
  }
  ENDPOINT("GET", "getInternalState/", getInternalState) {
    auto dto = InternalStateDto::createShared();
    dto->predictionIntervalMs = inference_interval_ms;
    std::stringstream ss;
    {
      std::lock_guard<std::mutex> lock(model_ids_mtx);
      for (int i = 0; i < model_ids.size(); ++i) {
        ss << model_ids[i];
        if (i < model_ids.size() - 1) {
          ss << ", ";
        }
      }
    }
    dto->modelIds = ss.str();
    {
      std::lock_guard<std::mutex> lock(image_queue_mtx);
      dto->image_queue_size = image_queue.size();
    }
    auto percentiles = std::vector<double>{10, 50, 66, 90, 95, 99, 99.99};
    dto->inference_duration_stats = oatpp::data::mapping::type::List<
        oatpp::data::mapping::type::PairList<String, Int32>>::createShared();
    {
      std::lock_guard<std::mutex> lock(swagger_mtx);
      for (const auto &pair : pt_dict) {
        auto interval = pair.first;
        auto pt = pair.second;
        pt.refreshStats();
        dto->inference_duration_stats->push_back(
            oatpp::data::mapping::type::PairList<String,
                                                 Int32>::createShared());
        auto last_pt = dto->inference_duration_stats->back();
        last_pt->push_back(
            std::pair<String, Int32>("inferenceCount", pt.sampleCount()));
        last_pt->push_back(
            std::pair<String, Int32>("inferenceIntervalMs", interval));

        for (auto const &ele : percentiles) {
          ss.str("");
          ss << std::fixed << std::setprecision(2) << ele << "th";
          last_pt->push_back(
              std::pair<String, Int32>(ss.str(), round(pt.getPercentile(ele))));
        }
      }
      return createDtoResponse(Status::CODE_200, dto);
    }
  }
};

#include OATPP_CODEGEN_END(ApiController) //<- End Codegen

#endif /* SwaggerController_hpp */
