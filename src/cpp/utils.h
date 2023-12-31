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

void interruptible_sleep(
    const size_t &sleep_ms, const size_t interval_ms = 1000,
    std::function<void()> func = []() {});

void update_last_inference_at();

std::string unix_ts_to_iso_datetime(int64_t unix_ts_ms);

#endif // VBCP_UTILS_H
