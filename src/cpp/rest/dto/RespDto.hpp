#ifndef ResponseDto_hpp
#define ResponseDto_hpp

#include <nlohmann/json.hpp>

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include OATPP_CODEGEN_BEGIN(DTO)

class RespDto : public oatpp::DTO {

  DTO_INIT(RespDto, DTO)

  DTO_FIELD(Boolean, success, "success");
  DTO_FIELD(String, responseText, "responseText");
};

#include OATPP_CODEGEN_END(DTO)

#endif /* ResponseDto_hpp */
