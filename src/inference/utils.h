#ifndef VBCP_UTILS_H
#define VBCP_UTILS_H

#include <nlohmann/json.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/script.h> // One-stop header.
#pragma GCC diagnostic pop

#include <deque>
#include <iostream>
#include <mutex>
#include <signal.h>

void install_signal_handler();

std::string get_current_datetime_string();

void interruptible_sleep(const size_t sleep_ms);

void update_last_inference_at();

std::string unix_ts_to_iso_datetime(int64_t unix_ts_ms);

template <typename T> std::string vector_to_string(std::vector<T> vec) {
  std::ostringstream vec_oss;
  vec_oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    vec_oss << vec[i];
    if (i != vec.size() - 1) {
      vec_oss << ", ";
    }
  }
  vec_oss << "]";
  return vec_oss.str();
}

#endif // VBCP_UTILS_H
