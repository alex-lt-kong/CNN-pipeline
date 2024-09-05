
#define FMT_HEADER_ONLY

#include "../utils.h"
#include "inference_result.pb.h"

#include <Magick++.h>
#include <cxxopts.hpp>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <readerwriterqueue/readerwritercircularbuffer.h>
#include <spdlog/spdlog.h>
#include <zmq.hpp>

#include <string>

using namespace std;
using json = nlohmann::json;

static volatile sig_atomic_t ev_flag = 0;
static string zmq_address;
static string external_program;
static size_t queue_size = 128;
constexpr size_t batch_size = 48;
static int gif_width;
static int gif_height;
static string gif_path;
static int gif_frame_interval;
static cv::Size zmq_payload_mat_size;
static filesystem::path jpg_dir;
static int zmq_payload_mat_type = CV_8UC3;
static auto inference_result_pc_queue =
    moodycamel::BlockingReaderWriterCircularBuffer<InferenceResultMsg>(
        queue_size);

void save_positive_outputs_as_jpeg(const vector<vector<uchar>> &jpegs,
                                   array<int, batch_size> labels) {

  spdlog::info("Saving positive images as JPEG files to [{}]",
               jpg_dir.native());
  for (size_t i = 0; i < batch_size; ++i) {
    if (labels[i] != 2)
      continue;
    filesystem::path jpg_path =
        jpg_dir / (get_current_datetime_string() + ".jpg");
    ofstream out_file(jpg_path, ios::binary);
    if (!out_file) {
      spdlog::error("Failed to open the file [{}]", jpg_path.native());
    } else {
      out_file.write((char *)jpegs[i].data(), jpegs[i].size());
      out_file.close();
    }
  }
}

void execute_external_program() {
  if (external_program.size() == 0) {
    spdlog::warn("external_program unset, not calling...");
    return;
  }
  spdlog::info("Calling external program: {}", external_program);
  (void)!system(external_program.c_str());
}

void prepare_images(array<InferenceResultMsg, batch_size> results,
                    array<int, batch_size> labels) {
  vector<vector<uchar>> jpegs(batch_size);
  // positive_y_preds_idx[0] stores the first non-zero item
  // index in y_pred. Note that y_pred[0] is NOT the prediction
  // of jpegs[0], but jpegs[pre_detection_size]
  // cv::imdecode(v, cv::IMREAD_COLOR);
  vector<Magick::Image> frames;
  for (size_t i = 0; i < batch_size; ++i) {
    cv::imencode(".jpg",
                 cv::Mat(zmq_payload_mat_size, zmq_payload_mat_type,
                         (void *)results[i].payload().data()),
                 jpegs[i]);
    frames.emplace_back(Magick::Blob(jpegs[i].data(), jpegs[i].size()));
    frames.back().animationDelay(gif_frame_interval);
    frames.back().resize(Magick::Geometry(gif_width, gif_height));
  }

  spdlog::info("Saving GIF file (size: {}x{}) to [{}]", gif_width, gif_height,
               gif_path);
  Magick::writeImages(frames.begin(), frames.end(), gif_path);
  save_positive_outputs_as_jpeg(jpegs, labels);
}

void inference_handling_ev_loop() {
  array<InferenceResultMsg, batch_size> results;
  array<int, batch_size> labels;
  spdlog::info("inference_handling_ev_loop() started");
  int neg_idx = -1;
  int pos_idx = -1;
  while (!ev_flag) {
    if (inference_result_pc_queue.size_approx() < batch_size) {
      interruptible_sleep(5000, &ev_flag);
      spdlog::info(
          "inference_result_pc_queue.size_approx() is {}, will wait and retry",
          inference_result_pc_queue.size_approx());
      continue;
    }
    for (size_t i = 0; i < batch_size; ++i) {
      if (!inference_result_pc_queue.try_dequeue(results[i])) {
        spdlog::warn("Unexpected result: "
                     "inference_result_pc_queue.try_dequeue() failed");
      }
    }
    for (size_t i = 0; i < batch_size; ++i) {
      if (results[i].labels_size() != 3) {
        auto errMsg = "results[i].labels_size() != 3";
        spdlog::error(errMsg);
        throw runtime_error(errMsg);
      }
      if (results[i].labels(0) == results[i].labels(1)) {
        labels[i] = results[i].labels(0);
      } else if (results[i].labels(0) == results[i].labels(2)) {
        labels[i] = results[i].labels(0);
      } else if (results[i].labels(1) == results[i].labels(2)) {
        labels[i] = results[i].labels(1);
      } else {
        labels[i] = results[i].labels(1);
      }
    };
    neg_idx = -1;
    pos_idx = -1;
    for (size_t i = 0; i < batch_size; ++i) {
      if (labels[i] == 0) {
        neg_idx = i;
      }
      if (labels[i] == 2) {
        pos_idx = i;
      }
    }
    spdlog::info("Handling batchm, labels: {}", fmt::format("{}", labels));
    if (neg_idx == -1 && pos_idx >= 0) {
      prepare_images(results, labels);
      execute_external_program();
    }
  }
  spdlog::info("inference_handling_ev_loop() exited gracefully");
}

void zeromq_consumer_ev_loop() {
  zmq::context_t context(1);
  zmq::socket_t subscriber(context, ZMQ_SUB);
  subscriber.connect(zmq_address);
  spdlog::info("ZeroMQ consumer connected to {}", zmq_address);
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
      if (!inference_result_pc_queue.try_enqueue(msg)) {
        spdlog::warn("inference_result_pc_queue.try_enqueue(msg) failed, "
                     "consumer not dequeue()ing fast enough?");
      }
    } else {
      spdlog::error("Failed to parse ZeroMQ payload as SnapshotMsg");
    }
  }
  subscriber.close();
  spdlog::info("ZeroMQ connection closed");
  spdlog::info("zeromq_consumer_ev_loop() exited gracefully");
}

int main(int argc, char **argv) {
  install_signal_handler(&ev_flag);
  string config_path;

  cxxopts::Options options(argv[0], "Inference service");
  // clang-format off
  options.add_options()
    ("h,help", "print help message")
    ("c,config-path", "JSON configuration file path", cxxopts::value<string>()->default_value(config_path));
  // clang-format on
  auto result = options.parse(argc, argv);
  if (result.count("help") || !result.count("config-path")) {
    std::cout << options.help() << "\n";
    return 0;
  }
  spdlog::set_pattern("%Y-%m-%d %T.%e | %7l | %5t | %v");
  spdlog::info("three_class_inference_consumer started (git commit: {})",
               GIT_COMMIT_HASH);
  config_path = result["config-path"].as<std::string>();
  ifstream f(config_path);
  json settings = json::parse(f);
  zmq_address =
      settings.value("/inference/zeromq_producer/address"_json_pointer,
                     "tcp://127.0.0.1:14241");
  external_program = settings.value(
      "/inference/on_detected/external_program"_json_pointer, "");
  gif_width =
      settings.value("/inference/on_detected/gif/size/width"_json_pointer, 100);
  gif_height = settings.value(
      "/inference/on_detected/gif/size/height"_json_pointer, 100);
  // The unit is "Time in 1/100ths of a second"
  gif_frame_interval =
      settings.value(
          "/inference/on_detected/gif/frame_interval_ms"_json_pointer, 200) /
      10;
  gif_path = settings.value("/inference/on_detected/gif/path"_json_pointer,
                            "/tmp/detected.gif");
  zmq_payload_mat_size = cv::Size(
      settings.value("/inference/zeromq/image_size/width"_json_pointer, 1920),
      settings.value("/inference/zeromq/image_size/height"_json_pointer, 1080));
  jpg_dir = settings.value(
      "/inference/on_detected/jpegs_directory"_json_pointer, "/");

  thread thread_zeromq_consumer(zeromq_consumer_ev_loop);
  thread thread_inference_handling_ev_loop(inference_handling_ev_loop);
  if (thread_zeromq_consumer.joinable()) {
    thread_zeromq_consumer.join();
  }
  if (thread_inference_handling_ev_loop.joinable()) {
    thread_inference_handling_ev_loop.join();
  }
  spdlog::info("three_class_inference_consumer exited");
  return 0;
}
