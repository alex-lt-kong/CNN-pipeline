#ifndef SettingsDto_hpp
#define SettingsDto_hpp

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include OATPP_CODEGEN_BEGIN(DTO)

class SettingsDto : public oatpp::DTO {

  DTO_INIT(SettingsDto, DTO)
  DTO_FIELD(oatpp::Any, settings, "settings");
};

#include OATPP_CODEGEN_END(DTO)

#endif /* SettingsDto_hpp */
