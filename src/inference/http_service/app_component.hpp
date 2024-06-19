#ifndef CP_APP_COMPONENT_HPP
#define CP_APP_COMPONENT_HPP

#include "error_handler.hpp"
#include "swagger_component.hpp"

#include <oatpp/core/macro/component.hpp>
#include <oatpp/network/tcp/server/ConnectionProvider.hpp>
#include <oatpp/parser/json/mapping/ObjectMapper.hpp>
#include <oatpp/web/server/HttpConnectionHandler.hpp>
#include <oatpp/web/server/HttpRouter.hpp>

#include <iostream>

/**
 *  Class which creates and holds Application components and registers
 * components in oatpp::base::Environment Order of components initialization is
 * from top to bottom
 */
class AppComponent {
private:
  oatpp::String host;
  v_uint16 port;

public:
  AppComponent(oatpp::String host, v_uint16 port, oatpp::String advertisedHost)
      : host(host), port(port), swaggerComponent(advertisedHost) {}
  /**
   *  Swagger component
   */
  SwaggerComponent swaggerComponent;
  /**
   * Create ObjectMapper component to serialize/deserialize DTOs in Controller's
   * API
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::data::mapping::ObjectMapper>,
                         apiObjectMapper)
  ([] {
    auto objectMapper =
        oatpp::parser::json::mapping::ObjectMapper::createShared();
    objectMapper->getDeserializer()->getConfig()->allowUnknownFields = false;
    return objectMapper;
  }());

  /**
   *  Create ConnectionProvider component which listens on the port
   */
  OATPP_CREATE_COMPONENT(
      std::shared_ptr<oatpp::network::ServerConnectionProvider>,
      serverConnectionProvider)
  ([this] {
    return oatpp::network::tcp::server::ConnectionProvider::createShared(
        {host, port, oatpp::network::Address::IP_4});
  }());

  /**
   *  Create Router component
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>,
                         httpRouter)
  ([] { return oatpp::web::server::HttpRouter::createShared(); }());

  /**
   *  Create ConnectionHandler component which uses Router component to route
   * requests
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::network::ConnectionHandler>,
                         serverConnectionHandler)
  ([] {
    OATPP_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>,
                    router); // get Router component
    OATPP_COMPONENT(std::shared_ptr<oatpp::data::mapping::ObjectMapper>,
                    objectMapper); // get ObjectMapper component

    auto connectionHandler =
        oatpp::web::server::HttpConnectionHandler::createShared(router);
    connectionHandler->setErrorHandler(
        std::make_shared<ErrorHandler>(objectMapper));
    return connectionHandler;
  }());
};

#endif // CP_APP_COMPONENT_HPP
