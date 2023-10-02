#ifndef UTILS_H
#define UTILS_H

#include <deque>
#include <mutex>
#include <signal.h>

#include <nlohmann/json.hpp>

extern volatile sig_atomic_t ev_flag;
extern nlohmann::json settings;
extern std::atomic<uint32_t> prediction_interval_ms;
extern const std::vector<std::string> modelIds;
extern std::mutex image_queue_mtx, ext_program_mtx;
extern std::deque<std::vector<char>> image_queue;

static void signal_handler(int signum);

void install_signal_handler();

#endif /* UTILS_H */
