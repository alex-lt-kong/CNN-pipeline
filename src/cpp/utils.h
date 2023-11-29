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
template <class T> class PercentileTracker {
public:
  PercentileTracker(size_t sample_size) : sample_size(sample_size) {
    refreshStatsCalled = false;
  }

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
    if (!refreshStatsCalled) {
      std::cerr << "WARNING: refreshStats() not called before calling "
                   "getPercentile(), the result will be nonsense!\n";
    }

    int index = percent / 100.0 * data.size() - 1;
    return data[index];
  }

  auto sampleCount() { return data.size(); }

private:
  std::deque<T> data;
  size_t sample_size;
  bool refreshStatsCalled;
};

static void signal_handler(int signum);

void install_signal_handler();

std::string getCurrentDateTimeString();

void interruptible_sleep(const size_t sleep_ms);

void update_last_inference_at();

#endif // VBCP_UTILS_H
