
#ifndef VBCP_SwaggerComponent_hpp
#define VBCP_SwaggerComponent_hpp

#include <oatpp-swagger/Model.hpp>
#include <oatpp-swagger/Resources.hpp>
#include <oatpp/core/macro/component.hpp>

/**
 *  Swagger ui is served at
 *  http://host:port/swagger/ui
 */
class SwaggerComponent {
private:
  oatpp::String advertisedHost;

public:
  SwaggerComponent(oatpp::String advertisedHost)
      : advertisedHost(advertisedHost) {}
  /**
   *  General API docs info
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::swagger::DocumentInfo>,
                         swaggerDocumentInfo)
  ([this] {
    oatpp::swagger::DocumentInfo::Builder builder;

    builder.setTitle("Swagger of " PROJECT_NAME)
        .setDescription("Configuring predict daemon of " PROJECT_NAME
                        " on the fly")
        .setVersion("1.0")
        .setContactName("Alex Kong")
        .setContactUrl("https://github.com/alex-lt-kong/")

        .setLicenseName("Apache License, Version 2.0")
        .setLicenseUrl("http://www.apache.org/licenses/LICENSE-2.0")

        .addServer(advertisedHost, std::string("server on ") + advertisedHost);

    return builder.build();
  }());

  /**
   *  Swagger-Ui Resources (<oatpp-examples>/lib/oatpp-swagger/res)
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::swagger::Resources>,
                         swaggerResources)
  ([] {
    // Make sure to specify correct full path to oatpp-swagger/res folder !!!
    return oatpp::swagger::Resources::loadResources(OATPP_SWAGGER_RES_PATH);
  }());
};

#endif // VBCP_SwaggerComponent_hpp
