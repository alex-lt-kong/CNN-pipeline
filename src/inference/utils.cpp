//#define FMT_HEADER_ONLY

#include "utils.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <regex>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

using namespace std;

static volatile sig_atomic_t *_ev_flag = 0;

void signal_handler(int signum) {
  char msg[] = "Signal [  ] caught\n";
  msg[8] = '0' + (char)(signum / 10);
  msg[9] = '0' + (char)(signum % 10);
  (void)write(STDIN_FILENO, msg, strlen(msg));
  *_ev_flag = 1;
}

void install_signal_handler(volatile sig_atomic_t *ev_flag) {
  _ev_flag = ev_flag;
  // This design canNOT handle more than 99 signal types
  if (_NSIG > 99) {
    fprintf(stderr, "signal_handler() can't handle more than 99 signals\n");
    abort();
  }
  struct sigaction act;
  // Initialize the signal set to empty, similar to memset(0)
  if (sigemptyset(&act.sa_mask) == -1) {
    perror("sigemptyset()");
    abort();
  }
  act.sa_handler = signal_handler;
  /* SA_RESETHAND means we want our signal_handler() to intercept the signal
  once. If a signal is sent twice, the default signal handler will be used
  again. `man sigaction` describes more possible sa_flags. */
  act.sa_flags = SA_RESETHAND;
  // act.sa_flags = 0;
  if (sigaction(SIGINT, &act, 0) + sigaction(SIGABRT, &act, 0) +
          sigaction(SIGQUIT, &act, 0) + sigaction(SIGTERM, &act, 0) +
          sigaction(SIGPIPE, &act, 0) + sigaction(SIGTRAP, &act, 0) <
      0) {
    perror("sigaction()");
    abort();
  }
}

string get_current_datetime_string() {
  auto now = chrono::system_clock::now();
  auto in_time_t = chrono::system_clock::to_time_t(now);

  stringstream ss;
  ss << put_time(localtime(&in_time_t), "%Y%m%d-%H%M%S%");
  // Get the milliseconds
  auto milliseconds =
      chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch())
          .count() %
      1000;

  // Append the milliseconds to the datetime string
  ss << '.' << setw(3) << setfill('0') << milliseconds;
  return ss.str();
}

void interruptible_sleep(const size_t sleep_ms, volatile int *ev_flag) {
  const size_t interval_ms = 1000;
  if (sleep_ms <= interval_ms) {
    this_thread::sleep_for(chrono::milliseconds(sleep_ms));
  } else {
    size_t slept_ms = 0;
    while (slept_ms < sleep_ms && !*ev_flag) {
      slept_ms += interval_ms;
      this_thread::sleep_for(chrono::milliseconds(interval_ms));
    }
  }
}

std::string unix_ts_to_iso_datetime(int64_t unix_ts_ms) {
  std::chrono::milliseconds ms(unix_ts_ms);
  std::chrono::time_point<std::chrono::system_clock> time_point(ms);
  std::time_t time_t = std::chrono::system_clock::to_time_t(time_point);
  std::tm *p_tm = std::localtime(&time_t);
  std::stringstream ss;
  ss << std::put_time(p_tm, "%FT%T") << '.' << std::setfill('0') << std::setw(3)
     << ms.count() % 1000;
  return ss.str();
}
