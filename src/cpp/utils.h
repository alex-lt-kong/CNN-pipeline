#ifndef UTILS_H
#define UTILS_H

#include <nlohmann/json.hpp>
#include <signal.h>

extern volatile sig_atomic_t ev_flag;
extern nlohmann::json settings;
extern std::atomic<uint32_t> prediction_interval_ms;

static void signal_handler(int signum);

void install_signal_handler();

#endif /* UTILS_H */
