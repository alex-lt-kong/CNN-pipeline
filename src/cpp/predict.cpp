#include <iostream>

#define FMT_HEADER_ONLY
#include <spdlog/spdlog.h>

#include "rest/App.h"

void zeromq_thread() {}

int main(void) {
  spdlog::set_pattern("%Y-%m-%d %T.%e | %7l | %v");
  spdlog::info("Predict daemon started");
  initialize_rest_api();
  std::cin.ignore();
  finalize_rest_api();
  spdlog::info("Predict daemon exiting");
  return 0;
}