#ifndef InternalState_hpp
#define InternalState_hpp

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include OATPP_CODEGEN_BEGIN(DTO)

class InternalStateDto : public oatpp::DTO {

  DTO_INIT(InternalStateDto, DTO)

  DTO_FIELD(Int32, predictionIntervalMs, "predictionIntervalMs");
  DTO_FIELD(List<String>, modelIds, "modelIds");
  DTO_FIELD(UInt32, image_queue_size, "imageQueueSize");
};

#include OATPP_CODEGEN_END(DTO)

#endif /* InternalState_hpp */
