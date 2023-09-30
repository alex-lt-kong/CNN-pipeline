#include <iostream>

#define FMT_HEADER_ONLY
#include <spdlog/spdlog.h>

#include "AppComponent.hpp"

#include "controller/StaticController.hpp"
#include "controller/SwaggerController.hpp"

#include "oatpp-swagger/Controller.hpp"

#include "oatpp/network/Server.hpp"

bool server_running = false;
std::mutex server_op_mutex;
std::atomic_bool server_should_continue;
std::thread oatppThread;

/**
 * You can't run two of those threads in one application concurrently in this
 * setup. Especially with the AppComponents inside the Threads scope. If you
 * want to run multiple threads with multiple servers you either have to manage
 * the components by yourself and do not rely on the OATPP_COMPONENT mechanism
 * or have one process-global AppComponent. Further you have to make sure you
 * don't have multiple ServerConnectionProvider listening to the same port.
 */
void StartOatppServer() {
  std::lock_guard<std::mutex> lock(server_op_mutex);

  /* Check if server is already running, if so, do nothing. */
  if (server_running) {
    return;
  }
  /* Signal that the server is running */
  server_running = true;
  /* Tell the server it should run */
  server_should_continue.store(true);

  oatppThread = std::thread([] {
    AppComponent components(
        "0.0.0.0", 8000,
        "http://gpu-prod.sz.lan:8000"); // Create scope Environment components

    /* Get router component */
    OATPP_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>, router);

    oatpp::web::server::api::Endpoints docEndpoints;

    docEndpoints.append(router->addController(SwaggerController::createShared())
                            ->getEndpoints());

    router->addController(
        oatpp::swagger::Controller::createShared(docEndpoints));
    router->addController(StaticController::createShared());

    /* Get connection handler component */
    OATPP_COMPONENT(std::shared_ptr<oatpp::network::ConnectionHandler>,
                    connectionHandler);

    /* Get connection provider component */
    OATPP_COMPONENT(std::shared_ptr<oatpp::network::ServerConnectionProvider>,
                    connectionProvider);

    /* create server */
    oatpp::network::Server server(connectionProvider, connectionHandler);
    spdlog::info("Oat++ listening on {}:{}",
                 connectionProvider->getProperty("host").toString()->c_str(),
                 connectionProvider->getProperty("port").toString()->c_str());

    /* Run server, let it check a lambda-function if it should continue to run
     * Return true to keep the server up, return false to stop it.
     * Treat this function like a ISR: Don't do anything heavy in it! Just check
     * some flags or at max some very lightweight logic. The performance of your
     * REST-API depends on this function returning as fast as possible! */
    std::function<bool()> condition = []() {
      return server_should_continue.load();
    };
    /*
    for (auto i = connectionProvider->getProperties().begin();
         i != connectionProvider->getProperties().end(); ++i)
      std::cout << i->first.toString()->c_str() << " \t\t\t"
                << i->second.toString()->c_str() << std::endl;
    */

    server.run(condition);

    /* Server has shut down, so we dont want to connect any new connections */
    connectionProvider->stop();

    /* Now stop the connection handler and wait until all running connections
     * are served */
    connectionHandler->stop();
  });
}

void initialize_rest_api() {
  oatpp::base::Environment::init();
  StartOatppServer();
}
void finalize_rest_api() {
  spdlog::info("Rest service exiting");
  std::lock_guard<std::mutex> lock(server_op_mutex);

  /* Tell server to stop */
  server_should_continue.store(false);

  /* Wait for the server to stop */
  if (oatppThread.joinable()) {
    oatppThread.join();
  }

  /* Print how many objects were created during app
   * running, and what have left-probably leaked */
  /* Disable object counting for release builds using '-D
   * OATPP_DISABLE_ENV_OBJECT_COUNTERS' flag for better performance */
  std::cout << "\nEnvironment:\n";
  std::cout << "objectsCount = " << oatpp::base::Environment::getObjectsCount()
            << "\n";
  std::cout << "objectsCreated = "
            << oatpp::base::Environment::getObjectsCreated() << "\n\n";

  oatpp::base::Environment::destroy();
}
