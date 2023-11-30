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

std::string getCurrentDateTimeString();

void interruptible_sleep(const size_t sleep_ms);

void update_last_inference_at();

#endif // VBCP_UTILS_H
