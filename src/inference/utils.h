#pragma once

#include <nlohmann/json.hpp>

#include <iostream>
#include <mutex>
#include <csignal>

void install_signal_handler(volatile sig_atomic_t *ev_flag);

std::string get_current_datetime_string(const char *fmt = "%Y%m%d-%H%M%S");

void interruptible_sleep(const size_t sleep_ms, volatile int *ev_flag);

std::string unix_ts_to_iso_datetime(int64_t unix_ts_ms,
                                    const char *fmt = "%FT%T",
                                    bool append_ms_part = true);

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
