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

#include <c10/cuda/CUDACachingAllocator.h>
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

namespace GV = CnnPipeline::GlobalVariables;

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
      auto _models = CnnPipeline::ModelUtils::load_models(
          _model_ids, GV::ts_model_path, GV::cuda_device_string);
      {
        std::scoped_lock lck{GV::models_mtx, GV::model_ids_mtx};
        GV::models = std::move(_models);
        GV::model_ids = std::move(_model_ids);
      }
      c10::cuda::CUDACachingAllocator::emptyCache();
      resp->success = true;
      resp->responseText = "Models loaded without exception";
      code = Status::CODE_200;
    } catch (const c10::Error &e) {
      spdlog::error("Error loading the model: {}", e.what());
      resp->success = false;
      resp->responseText = std::string("Model not reloaded: ") + e.what() +
                           " (existing models will remain in use)";
      code = Status::CODE_400;
    }
    return createDtoResponse(code, resp);
  }

  ENDPOINT_INFO(getSettings) {
    info->summary =
        "get the content of config.json of this inference daemon instance";
  }
  ENDPOINT("GET", "getSettings/", getSettings) {
    auto jsonObjectMapper =
        oatpp::parser::json::mapping::ObjectMapper::createShared();
    auto settingsOatppJson =
        jsonObjectMapper->readFromString<oatpp::Any>(GV::settings.dump());
    return createDtoResponse(Status::CODE_200, settingsOatppJson);
  }

  ENDPOINT_INFO(getInternalState) {
    info->summary = "get the internal state of the prediction daemon";

    info->addResponse<Object<InternalStateDto>>(Status::CODE_200,
                                                "application/json");
  }
  ENDPOINT("GET", "getInternalState/", getInternalState) {
    auto dto = InternalStateDto::createShared();
    std::stringstream ss;
    {
      std::lock_guard<std::mutex> lock(GV::model_ids_mtx);
      for (size_t i = 0; i < GV::model_ids.size(); ++i) {
        ss << GV::model_ids[i];
        if (i < GV::model_ids.size() - 1) {
          ss << ", ";
        }
      }
    }
    dto->modelIds = ss.str();
    dto->image_queue_size = GV::snapshot_pc_queue.size_approx();
    dto->lastInferenceAt = GV::last_inference_at;
    auto percentiles = std::vector<double>{10, 50, 66, 90, 95, 99, 99.99};
    dto->inference_duration_stats =
        oatpp::data::mapping::type::PairList<String, Int32>::createShared();
    {
      std::lock_guard<std::mutex> lock(GV::swagger_mtx);

      GV::pt.refreshStats();
      dto->inference_duration_stats->push_back(
          std::pair<String, Int32>("inferenceCount", GV::pt.sampleCount()));

      for (auto const &ele : percentiles) {
        ss.str("");
        ss << std::fixed << std::setprecision(2) << ele << "th";
        dto->inference_duration_stats->push_back(std::pair<String, Int32>(
            ss.str(), round(GV::pt.getPercentile(ele))));
      }

      return createDtoResponse(Status::CODE_200, dto);
    }
  }
};

#include OATPP_CODEGEN_END(ApiController) //<- End Codegen

#endif /* SwaggerController_hpp */
