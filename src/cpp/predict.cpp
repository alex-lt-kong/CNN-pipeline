
#define FMT_HEADER_ONLY
#include <spdlog/spdlog.h>

#include "rest/App.h"

int main(void) {
  spdlog::set_pattern("%Y-%m-%d %T.%e | %7l | %v");
  spdlog::info("Predict daemon started");
  init_swagger();
  return 0;
}