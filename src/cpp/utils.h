#ifndef VBCP_UTILS_H
#define VBCP_UTILS_H

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

static void signal_handler(int signum);

void install_signal_handler();

std::string getCurrentDateTimeString();

void interruptible_sleep(const size_t sleep_ms);

void update_last_inference_at();

#endif // VBCP_UTILS_H
