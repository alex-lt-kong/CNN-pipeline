#ifndef UTILS_H
#define UTILS_H

#include <deque>
#include <mutex>
#include <signal.h>

#include <nlohmann/json.hpp>
#include <torch/script.h> // One-stop header.

template <class T> class PercentileTracker {
public:
  PercentileTracker(size_t sample_size) : sample_size(sample_size) {}

  void addNumber(T num) {
    data.push_back(num);
    if (data.size() > sample_size) {
      data.pop_front();
    }
  }

  void refreshStats() { std::sort(data.begin(), data.end()); }

  double getPercentile(double percent) {
    if (data.empty())
      return -1;

    int index = percent / 100.0 * data.size() - 1;
    return data[index];
  }

  auto sampleCount() { return data.size(); }

private:
  std::deque<T> data;
  size_t sample_size;
};

extern volatile sig_atomic_t ev_flag;
extern nlohmann::json settings;
extern std::atomic<uint32_t> inference_interval_ms;
extern std::vector<std::string> model_ids;
extern std::mutex image_queue_mtx, ext_program_mtx, swagger_mtx, models_mtx,
    model_ids_mtx;
extern std::deque<std::vector<char>> image_queue;
extern std::unordered_map<uint32_t, PercentileTracker<float>> pt_dict;
extern std::vector<torch::jit::script::Module> models;

static void signal_handler(int signum);

void install_signal_handler();

std::string getCurrentDateTimeString();

#endif /* UTILS_H */
