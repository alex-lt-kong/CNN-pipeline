#define FMT_HEADER_ONLY
#include "inference_result.pb.h"

#include <spdlog/spdlog.h>
#include <string>
#include <zmq.hpp>

using namespace std;

int main() {

  zmq::context_t context(1);
  zmq::socket_t subscriber(context, ZMQ_SUB);
  string zmq_address = "tcp://127.0.0.1:24240";
  subscriber.connect(zmq_address);
  spdlog::info("ZeroMQ client connected to {}", zmq_address);
  constexpr int timeout = 5000;
  // For the use of zmqcpp, refer to this 3rd-party tutorial:
  // https://brettviren.github.io/cppzmq-tour/index.html#org3a79e67
  subscriber.set(zmq::sockopt::rcvtimeo, timeout);
  subscriber.set(zmq::sockopt::subscribe, "");
  bool ev_flag = false;
  while (!ev_flag) {
    zmq::message_t message;
    InferenceResultMsg msg;
    try {
      auto res = subscriber.recv(message, zmq::recv_flags::none);
      if (!res.has_value()) {
        spdlog::warn("subscriber.recv() timed out, will retry");
        continue;
      }
      if (res == 0) {
        spdlog::error("subscriber.recv() receives zero bytes, will retry");
        continue;
      }
    } catch (const zmq::error_t &e) {
      if (e.num() == EAGAIN) {
        spdlog::warn("subscriber.recv() timed out, will retry");
        continue;
      } else {
        spdlog::error("subscriber.recv() throws exception: {}", e.what());
        continue;
      }
    }
    if (message.size() == 0) {
      spdlog::warn("subscriber.recv() returned an empty message (probably due "
                   "to time out)");
      continue;
    }
    if (msg.ParseFromArray(message.data(), message.size())) {
      spdlog::info("msg received, msg.label(): {}, latency: {}ms", msg.label(),
                   (msg.inferenceunixepochns() - msg.snapshotunixepochns()) /
                       1000 / 1000);
    } else {
      spdlog::error("Failed to parse ZeroMQ payload as SnapshotMsg");
    }
  }
  subscriber.close();
  spdlog::info("ZeroMQ connection closed");
  spdlog::info("zeromq_consumer_ev_loop() exited gracefully");

  return 0;
}