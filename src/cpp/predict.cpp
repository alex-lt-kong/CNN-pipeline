#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <stdexcept>
#include <string>

#include <getopt.h>
#define FMT_HEADER_ONLY
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <zmq.hpp>

#include "rest/App.h"

using namespace std;
using json = nlohmann::json;

volatile sig_atomic_t e_flag = 0;

static void signal_handler(int signum) {
  char msg[] = "Signal [  ] caught\n";
  msg[8] = '0' + (char)(signum / 10);
  msg[9] = '0' + (char)(signum % 10);
  write(STDIN_FILENO, msg, strlen(msg));
  e_flag = 1;
}

void install_signal_handler() {
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
          sigaction(SIGPIPE, &act, 0) + sigaction(SIGCHLD, &act, 0) +
          sigaction(SIGTRAP, &act, 0) <
      0) {
    perror("sigaction()");
    abort();
  }
}

void print_usage(string binary_name) {

  cerr << "Usage: " << binary_name << " [OPTION]\n\n";

  cerr << "Options:\n"
       << "  --help,        -h        Display this help and exit\n"
       << "  --config-path, -c        JSON configuration file path" << endl;
}

void parse_arguments(int argc, char **argv, string &config_path) {
  static struct option long_options[] = {{"config", required_argument, 0, 'c'},
                                         {"help", optional_argument, 0, 'h'},
                                         {0, 0, 0, 0}};

  int opt, option_index = 0;

  while ((opt = getopt_long(argc, argv, "c:h", long_options, &option_index)) !=
         -1) {
    switch (opt) {
    case 'c':
      if (optarg != NULL) {
        config_path = string(optarg);
      }
      break;
    default:
      print_usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  if (config_path.empty()) {
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }
}

void zeromq_thread(string uri) { // Create a ZMQ context
  zmq::context_t context(1);

  // Create a ZMQ socket for subscribing
  zmq::socket_t subscriber(context, ZMQ_SUB);

  // Connect to the publisher
  spdlog::info("Connecting to {}", uri);
  subscriber.connect(uri);
  spdlog::info("Connected to {}", uri);
  subscriber.setsockopt(ZMQ_SUBSCRIBE, "", 0);
  spdlog::info("Subscribed to topic ''");
  while (!e_flag) {
    zmq::message_t message;
    subscriber.recv(&message);
    spdlog::info("Received message, size: {}", message.size());
    std::ofstream outputFile("/tmp/received.data",
                             std::ios::out | std::ios::binary);
    if (!outputFile) {
      throw runtime_error("Failed to open output file.");
    }
    outputFile.write(static_cast<const char *>(message.data()), message.size());
  }
  subscriber.close();
  spdlog::info("ZeroMQ connection closed gracefully");
}

int main(int argc, char **argv) {
  install_signal_handler();
  string config_path;
  parse_arguments(argc, argv, config_path);
  spdlog::set_pattern("%Y-%m-%d %T.%e | %7l | %v");
  spdlog::info("Predict daemon started");
  spdlog::info("Loading configurations from {}:", config_path);
  ifstream f(config_path);
  json settings = json::parse(f);
  spdlog::info("{}", settings.dump(2));
  initialize_rest_api(
      settings["prediction"]["swagger"]["host"].get<string>(),
      settings["prediction"]["swagger"]["port"].get<int>(),
      settings["prediction"]["swagger"]["advertised_host"].get<string>());
  std::thread thread_object(zeromq_thread, "tcp://127.0.0.1:4241");
  // zeromq_thread("tcp://127.0.0.1:4241");
  // std::cin.ignore();
  // e_flag = 1;
  thread_object.join();
  finalize_rest_api();
  spdlog::info("Predict daemon exiting");
  return 0;
}