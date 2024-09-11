#include "event_loops.h"
#include "global_vars.h"
#include "model_utils.h"
#include "snapshot.pb.h"
#include "utils.h"

#include <chrono>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <tuple>
#include <zmq.hpp>

namespace CnnPipeline::EventLoops {

using namespace std;
namespace GV = CnnPipeline::GlobalVariables;
namespace MU = CnnPipeline::ModelUtils;

static cv::Size zmq_payload_mat_size;
static cv::Size target_img_size;
static size_t num_classes;
static int zmq_payload_mat_type = CV_8UC3;

vector<at::Tensor> infer_images(vector<torch::jit::script::Module> &models,
                                const deque<SnapshotMsg> &snaps) {
  constexpr size_t mil = 1000 * 1000;
  constexpr size_t bil = mil * 1000;
  spdlog::info(
      "Inferring {} images from [{}](idx:0) to [{}](idx:{}) (timespan: {} "
      "sec).",
      GV::inference_batch_size,
      unix_ts_to_iso_datetime(snaps[0].unixepochns() / mil),
      unix_ts_to_iso_datetime(
          snaps[GV::inference_batch_size - 1].unixepochns() / mil),
      GV::inference_batch_size - 1,
      (snaps[GV::inference_batch_size - 1].unixepochns() -
       snaps[0].unixepochns()) /
          bil);

  auto t0 = chrono::steady_clock::now();
  vector<torch::Tensor> images_tensors_vec(GV::inference_batch_size);
  torch::Tensor images_tensor;
  vector<torch::jit::IValue> input(1);
  for (size_t i = 0; i < GV::inference_batch_size; ++i) {
    // TODO: can we remove this std::copy()??
    // Profiling shows this copy takes less than 1 ms
    if ((uint)zmq_payload_mat_size.width * zmq_payload_mat_size.height * 3 !=
        snaps[i].cvmatbytes().size()) {
      throw runtime_error("Unexpected ZeroMQ payload received");
    }
    cv::Mat mat = cv::Mat(zmq_payload_mat_size, zmq_payload_mat_type,
                          (void *)snaps[i].cvmatbytes().data());
    // std::copy(snaps[i].payload().begin(), snaps[i].payload().end(),
    // v.begin());
    images_tensors_vec[i] = MU::cv_mat_to_tensor(mat, target_img_size);
  }
  images_tensor = torch::stack(images_tensors_vec);

  input[0] = images_tensor.to(GV::cuda_device_string);

  vector<at::Tensor> raw_outputs(models.size());
  {
    std::lock_guard<std::mutex> lock(GV::models_mtx);
    for (size_t i = 0; i < models.size(); ++i) {
      auto y = models[i].forward(input).toTensor();
      // Normalize the output, otherwise one model could have (unexpected)
      // outsized impact on the final result
      // Ref:
      // https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
      auto y_min = at::min(y);
      raw_outputs[i] = 2 * ((y - y_min) / (at::max(y) + 0.000001 - y_min)) - 1;
    }
  }
  auto t1 = chrono::steady_clock::now();
  {
    lock_guard<mutex> lock(GV::swagger_mtx);
    GV::pt.addSample(
        (float)chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() /
        GV::inference_batch_size);
  }
  return raw_outputs;
}

void handle_inference_results(vector<at::Tensor> &raw_outputs,
                              const deque<SnapshotMsg> &snap_deque) {

  for (size_t i = 0; i < GV::inference_batch_size; ++i) {
    InferenceResultMsg msg;
    msg.set_inferenceunixepochns(chrono::time_point_cast<chrono::nanoseconds>(
                                     chrono::system_clock::now())
                                     .time_since_epoch()
                                     .count());
    msg.set_snapshotunixepochns(snap_deque[i].unixepochns());
    msg.set_payload(snap_deque[i].cvmatbytes());
    msg.set_cooldown(snap_deque[i].cooldown());
    msg.set_rateofchange(snap_deque[i].rateofchange());
    // msg.set_label(y_pred[i].item<int>());
    for (size_t j = 0; j < raw_outputs.size(); ++j) {
      msg.add_labels(torch::argmax(raw_outputs[j][i], 0).item<int>());
    }
    string labels = "";
    for (size_t j = 0; j < raw_outputs.size(); ++j) {
      labels +=
          to_string(msg.labels(j)) + (j < raw_outputs.size() - 1 ? ", " : "");
    }
    if (!GV::inference_result_pc_queue.try_enqueue(std::move(msg)))
      spdlog::warn("Error inference_result_pc_queue.try_enqueue()");
    spdlog::info(
        "ts: {} ({}), labels: {}", snap_deque[i].unixepochns() / (1000 * 1000),
        unix_ts_to_iso_datetime(snap_deque[i].unixepochns() / (1000 * 1000)),
        labels);
  }
}

void inference_ev_loop() {
  constexpr size_t delay_ms = 10000;
  spdlog::info("inference_ev_loop() started, waiting for {}ms before start",
               delay_ms);
  // Otherwise we won't have any images of detected object in the GIF
  assert(GV::inference_batch_size > 0);
  // Initial wait, just to spread out the stress at the beginning of the run
  interruptible_sleep(delay_ms, &GV::ev_flag);
  try {
    {
      std::scoped_lock lck{GV::models_mtx, GV::model_ids_mtx};
      GV::models = MU::load_models(GV::model_ids, GV::ts_model_path,
                                   GV::cuda_device_string);
    }

  } catch (const c10::Error &e) {
    spdlog::error(
        "Error loading the model (c10::Error): {}\nThe program must exit now",
        e.what());
    GV::ev_flag = 1;
  } catch (const torch::jit::ErrorReport &e) {
    spdlog::error("Error loading the model (torch::jit::ErrorReport): {}\nThe "
                  "program must exit now",
                  e.what());
    GV::ev_flag = 1;
  }

  deque<SnapshotMsg> snap_deque;
  vector<cv::Mat> images_mats(GV::inference_batch_size);

  while (!GV::ev_flag) {
    size_t sample_count = 0;
    while (sample_count < GV::inference_batch_size && !GV::ev_flag) {
      SnapshotMsg t;
      if (GV::snapshot_pc_queue.try_dequeue(t)) {
        snap_deque.emplace_back(t);
        if (snap_deque.size() > GV::inference_batch_size) {
          snap_deque.pop_front();
        }
        ++sample_count;
      } else {
        interruptible_sleep(1000, &GV::ev_flag);
      }
    }
    if (GV::ev_flag) {
      break;
    }
    if (snap_deque.size() < GV::inference_batch_size) {
      spdlog::info(
          "snap_deque.size() too small ({}vs{}), queuing more elements "
          "before inference",
          snap_deque.size(), GV::inference_batch_size);
      continue;
    }

    try {
      auto raw_outputs = infer_images(GV::models, snap_deque);
      handle_inference_results(raw_outputs, snap_deque);
    } catch (const c10::Error &e) {
      spdlog::error("Inference failed, PyTorch-related error: {}", e.what());
    } catch (const runtime_error &e) {
      spdlog::error("Inference failed, runtime_error: {}", e.what());
    }
  }
  spdlog::info("inference_ev_loop() exited gracefully");
}

void zeromq_producer_ev_loop() {
  spdlog::info("zeromq_producer_ev_loop() started");

  zmq::socket_t zmqSocket;
  string zmq_address =
      GV::settings.value("/inference/zeromq_producer/address"_json_pointer,
                         "tcp://127.0.0.1:14240");
  zmq::context_t zmqContext(1);
  zmqSocket = zmq::socket_t(zmqContext, zmq::socket_type::pub);
  try {
    zmqSocket.bind(zmq_address);
  } catch (const zmq::error_t &e) {
    spdlog::error("zmqSocket.bind({}) failed: {}, zeroMQ IPC "
                  "support will be disabled for this device",
                  zmq_address, e.what());
    return;
  }
  spdlog::info("ZeroMQ producer bound to {}", zmq_address);
  while (!GV::ev_flag) {
    InferenceResultMsg msg;
    // spdlog::info("zeromq_producer_ev_loop() iterating...");
    if (!GV::inference_result_pc_queue.try_dequeue(msg)) {
      interruptible_sleep(1000, &GV::ev_flag);
      continue;
    }
    auto serializedMsg = msg.SerializeAsString();
    try {
      if (auto ret =
              zmqSocket.send(
                  zmq::const_buffer(serializedMsg.data(), serializedMsg.size()),
                  zmq::send_flags::none) != serializedMsg.size()) {
        spdlog::error("zmqSocket.send() failed: ZeroMQ socket reports {} bytes "
                      "being sent, but serializedMsg.size() is {} bytes",
                      ret, serializedMsg.size());
      }
    } catch (const zmq::error_t &err) {
      spdlog::error("zmqSocket.send() failed: {}({}). The program will "
                    "continue with this frame being unsent",
                    err.num(), err.what());
    }
  }
  zmqSocket.close();
  spdlog::info("zeromq_producer_ev_loop() exited gracefully");
}

void zeromq_consumer_ev_loop() {
  spdlog::info("zeromq_consumer_ev_loop() started");

  zmq::context_t context(1);
  zmq::socket_t subscriber(context, ZMQ_SUB);
  string zmq_address = GV::settings.value(
      "/inference/zeromq/address"_json_pointer, "tcp://127.0.0.1:4240");
  zmq_payload_mat_size =
      cv::Size(GV::settings.value(
                   "/inference/zeromq/image_size/width"_json_pointer, 1920),
               GV::settings.value(
                   "/inference/zeromq/image_size/height"_json_pointer, 1080));
  target_img_size = cv::Size(
      GV::settings.value("/model/input_image_size/width"_json_pointer, 0),
      GV::settings.value("/model/input_image_size/height"_json_pointer, 0));
  num_classes = GV::settings.value("/model/num_classes"_json_pointer, 2);
  subscriber.connect(zmq_address);
  spdlog::info("ZeroMQ client connected to {}", zmq_address);
  constexpr int timeout = 5000;
  // For the use of zmqcpp, refer to this 3rd-party tutorial:
  // https://brettviren.github.io/cppzmq-tour/index.html#org3a79e67
  subscriber.set(zmq::sockopt::rcvtimeo, timeout);
  subscriber.set(zmq::sockopt::subscribe, "");
  while (!GV::ev_flag) {
    zmq::message_t message;
    SnapshotMsg msg;
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
      spdlog::error("subscriber.recv() throws exception: {}", e.what());
      continue;
    }
    if (message.size() == 0) {
      spdlog::warn("subscriber.recv() returned an empty message (probably due "
                   "to time out)");
      continue;
    }
    if (msg.ParseFromArray(message.data(), message.size())) {
      while (!GV::snapshot_pc_queue.try_enqueue(std::move(msg))) {
        spdlog::warn(
            "try_enqueue() failed: snapshot_queue full (max_capacity(): "
            "{} vs size_approx(): {}), snapshots are not being inferred fast "
            "enough",
            GV::snapshot_pc_queue.max_capacity(),
            GV::snapshot_pc_queue.size_approx());
        SnapshotMsg t;
        if (GV::snapshot_pc_queue.try_dequeue(t)) {
          spdlog::warn(
              "napshot_pc_queue.try_dequeue()'ed to make place for new "
              "snapshot (max_capacity(): {} vs size_approx(): {})",
              GV::snapshot_pc_queue.max_capacity(),
              GV::snapshot_pc_queue.size_approx());
        } else {
          auto errMsg = "Unexpected branch, snapshot_pc_queue is full but "
                        "still can't dequeue";
          spdlog::error(errMsg);
          throw std::runtime_error(errMsg);
        }
      }
    } else {
      spdlog::error("Failed to parse ZeroMQ payload as SnapshotMsg");
    }
  }
  subscriber.close();
  spdlog::info("ZeroMQ connection closed");
  spdlog::info("zeromq_consumer_ev_loop() exited gracefully");
}

} // namespace CnnPipeline::EventLoops