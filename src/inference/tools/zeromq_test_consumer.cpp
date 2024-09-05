#define FMT_HEADER_ONLY
#include "../utils.h"
#include "inference_result.pb.h"

#include <spdlog/spdlog.h>
#include <string>
#include <zmq.hpp>

using namespace std;
static volatile sig_atomic_t ev_flag = 0;

int main() {

  install_signal_handler(&ev_flag);
  zmq::context_t context(1);
  zmq::socket_t subscriber(context, ZMQ_SUB);
  string zmq_address = "tcp://127.0.0.1:14241";
  subscriber.connect(zmq_address);
  spdlog::info("zeromq_consumer started (git commit: {})", GIT_COMMIT_HASH);
  spdlog::info("ZeroMQ client connected to {}", zmq_address);
  constexpr int timeout = 30000;
  // For the use of zmqcpp, refer to this 3rd-party tutorial:
  // https://brettviren.github.io/cppzmq-tour/index.html#org3a79e67
  subscriber.set(zmq::sockopt::rcvtimeo, timeout);
  subscriber.set(zmq::sockopt::subscribe, "");

  size_t msgCount = 0;
  while (!ev_flag) {
    zmq::message_t message;
    InferenceResultMsg msg;
    try {
      auto res = subscriber.recv(message, zmq::recv_flags::none);
      if (!res.has_value()) {
        spdlog::warn("subscriber.recv() timed out, will retry");
        msgCount = 0;
        continue;
      }
      if (res == 0) {
        spdlog::error("subscriber.recv() receives zero bytes, will retry");
        continue;
      }
    } catch (const zmq::error_t &e) {
      spdlog::error("subscriber.recv() throws exception: {}", e.what());
      continue;
    }
    if (message.size() == 0) {
      spdlog::warn("subscriber.recv() returned an empty message (probably due "
                   "to time out)");
      continue;
    }
    if (msg.ParseFromArray(message.data(), message.size())) {
      ++msgCount;

      string labels = "";
      for (auto lbl : msg.labels()) {
        labels += to_string(lbl) + ", ";
      }
      spdlog::info(
          "msg received (msgCount: {}), msg.label(): {}, msg.labels(): {}"
          "msg.payload().length(): {}, msg.snapshotunixepochns(): "
          "{}, latency: {}ms",
          msgCount, msg.label(), labels, msg.payload().length(),
          msg.snapshotunixepochns(),
          (msg.inferenceunixepochns() - msg.snapshotunixepochns()) / 1000 /
              1000);
    } else {
      spdlog::error("Failed to parse ZeroMQ payload as SnapshotMsg");
    }
  }
  subscriber.close();
  spdlog::info("ZeroMQ connection closed");
  spdlog::info("zeromq_consumer_ev_loop() exited gracefully");

  return 0;
}