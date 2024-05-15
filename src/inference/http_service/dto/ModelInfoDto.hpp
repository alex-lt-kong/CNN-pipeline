#ifndef ModelInfoDto_hpp
#define ModelInfoDto_hpp

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include OATPP_CODEGEN_BEGIN(DTO)

class ModelInfoDto : public oatpp::DTO {

  DTO_INIT(ModelInfoDto, DTO)
  DTO_FIELD(List<String>, modelIds, "modelIds") = {"0", "1", "2"};
};

#include OATPP_CODEGEN_END(DTO)

#endif /* ModelInfoDto_hpp */
