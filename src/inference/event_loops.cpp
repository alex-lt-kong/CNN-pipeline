#include "event_loops.h"
#include "global_vars.h"
#include "model_utils.h"
#include "snapshot.pb.h"
#include "utils.h"

#include <Magick++.h>
#include <chrono>
#include <spdlog/spdlog.h>
#include <zmq.hpp>

#include <tuple>

using namespace std;

static string cuda_device_string = "cuda:0";
static cv::Size zmq_payload_mat_size;
static cv::Size target_img_size;
static size_t num_classes;
static int zmq_payload_mat_type = CV_8UC3;

tuple<vector<at::Tensor>, at::Tensor>
infer_images(vector<torch::jit::script::Module> &models,
             const deque<SnapshotMsg> &snaps) {
  const size_t start_idx = pre_detection_size;
  const size_t end_idx = pre_detection_size + inference_batch_size;
  constexpr size_t mil = 1000 * 1000;
  constexpr size_t bil = mil * 1000;
  spdlog::info(
      "Inferring {} images from [{}](idx:{}) to [{}](idx:{}) (timespan: {} "
      "sec). The hypothetical gif contains {}(out of {}) images from "
      "[{}](idx:0) to [{}](idx:{}) (timespan: {} sec)",
      end_idx - start_idx,
      unix_ts_to_iso_datetime(snaps[start_idx].unixepochns() / mil), start_idx,
      unix_ts_to_iso_datetime(snaps[end_idx].unixepochns() / mil), end_idx,
      (snaps[end_idx].unixepochns() - snaps[start_idx].unixepochns()) / bil,
      gif_frame_count, snaps.size(),
      unix_ts_to_iso_datetime(snaps[0].unixepochns() / mil),
      unix_ts_to_iso_datetime(snaps[snaps.size() - 1].unixepochns() / mil),
      snaps.size() - 1,
      (snaps[snaps.size() - 1].unixepochns() - snaps[0].unixepochns()) / bil);

  auto t0 = chrono::steady_clock::now();
  vector<torch::Tensor> images_tensors_vec(inference_batch_size);
  torch::Tensor images_tensor;
  vector<torch::jit::IValue> input(1);
  for (size_t i = start_idx; i < end_idx; ++i) {
    // TODO: can we remove this std::copy()??
    // Profiling shows this copy takes less than 1 ms
    int idx = i - pre_detection_size;
    if ((uint)zmq_payload_mat_size.width * zmq_payload_mat_size.height * 3 !=
        snaps[i].payload().size()) {
      throw runtime_error("Unexpected ZeroMQ payload received");
    }
    cv::Mat mat = cv::Mat(zmq_payload_mat_size, zmq_payload_mat_type,
                          (void *)snaps[i].payload().data());
    // std::copy(snaps[i].payload().begin(), snaps[i].payload().end(),
    // v.begin());
    images_tensors_vec[idx] = cv_mat_to_tensor(mat, target_img_size);
  }
  images_tensor = torch::stack(images_tensors_vec);

  input[0] = images_tensor.to(cuda_device_string);

  // images_tensor.sizes()[0] stores number of images
  at::Tensor avg_output =
      torch::zeros({images_tensor.sizes()[0], static_cast<long>(num_classes)});
  avg_output = avg_output.to(cuda_device_string);
  vector<at::Tensor> raw_outputs(models.size());
  {
    std::lock_guard<std::mutex> lock(models_mtx);
    for (size_t i = 0; i < models.size(); ++i) {
      auto y = models[i].forward(input).toTensor();
      /*
      ostringstream oss;
      if (i == 1) {
        oss << y;
        spdlog::info("{}", y.toString());
        spdlog::info("{}", oss.str());
      }
      */
      // Normalize the output, otherwise one model could have (unexpected)
      // outsized impact on the final result
      // Ref:
      // https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
      auto y_min = at::min(y);
      raw_outputs[i] = 2 * ((y - y_min) / (at::max(y) + 0.000001 - y_min)) - 1;
      avg_output += raw_outputs[i];
    }
  }
  auto t1 = chrono::steady_clock::now();
  {
    lock_guard<mutex> lock(swagger_mtx);
    pt.addSample(
        (float)chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() /
        inference_batch_size);
  }
  return {raw_outputs, avg_output};
}

void execute_external_program_async() {
  auto f = [](string cmd) {
    if (ext_program_mtx.try_lock()) {
      spdlog::info("Calling external program: {}", cmd);
      (void)!system(cmd.c_str());
      ext_program_mtx.unlock();
    } else {
      spdlog::warn("ext_program_mtx is locked already, meaning that another {} "
                   "instance is already running",
                   cmd);
    }
  };
  thread th_exec(
      f, settings.value("/inference/on_detected/external_program"_json_pointer,
                        ""));
  th_exec.detach();
}

void save_positive_outputs_as_jpeg(const vector<int> &positive_y_preds_idx,
                                   const vector<vector<uchar>> &jpegs) {
  filesystem::path jpg_dir = settings.value(
      "/inference/on_detected/jpegs_directory"_json_pointer, "/");
  spdlog::info("Saving positive images as JPEG files to [{}]",
               jpg_dir.native());
  for (auto idx : positive_y_preds_idx) {
    // One needs to think for a while to understand the
    // offset between jpegs_idx and y_pred's index--their gap
    // is exactly inference_batch_size, which is implied in
    // the line: cv::imdecode(jpegs[pre_detection_size + i],
    // cv::IMREAD_COLOR));
    auto jpegs_idx = idx + pre_detection_size;
    filesystem::path jpg_path =
        jpg_dir / (get_current_datetime_string() + ".jpg");
    ofstream out_file(jpg_path, ios::binary);
    if (!out_file) {
      spdlog::error("Failed to open the file [{}]", jpg_path.native());
    } else {
      out_file.write((char *)jpegs[jpegs_idx].data(), jpegs[jpegs_idx].size());
      out_file.close();
    }
  }
}

bool handle_inference_results(vector<at::Tensor> &raw_outputs,
                              at::Tensor &avg_output,
                              const deque<SnapshotMsg> &snaps) {
  ostringstream oss_raw_result, oss_avg_result;
  for (size_t i = 0; i < raw_outputs.size(); ++i) {
    oss_raw_result << "raw_outputs[" << i << "]:\n";
    oss_raw_result << raw_outputs[i];
    oss_raw_result << "\n";
  }
  oss_avg_result << avg_output;
  at::Tensor y_pred = torch::argmax(avg_output, 1);
  vector<int> include_outputs = settings.value(
      "/inference/on_detected/triggers/include_outputs"_json_pointer,
      vector<int>{1});
  vector<int> exclude_outputs = settings.value(
      "/inference/on_detected/triggers/exclude_outputs"_json_pointer,
      vector<int>{});
  vector<int> positive_y_preds_idx, negative_y_preds_idx;
  for (int64_t i = 0; i < y_pred.size(0); i++) {
    for (size_t j = 0; j < include_outputs.size(); ++j) {
      if (y_pred[i].item<int>() == include_outputs[j]) {
        positive_y_preds_idx.push_back(i);
      }
    }
    for (size_t j = 0; j < exclude_outputs.size(); ++j) {
      if (y_pred[i].item<int>() == exclude_outputs[j]) {
        negative_y_preds_idx.push_back(i);
      }
    }
  }

  spdlog::info("y_pred: {}",
               tensor_to_string_like_pytorch(y_pred, 0, y_pred.sizes()[0]));
  if (positive_y_preds_idx.size() == 0) {
    spdlog::info("y_pred does not contain any include_outputs, returning");
    return false;
  }
  if (negative_y_preds_idx.size() > 0) {
    spdlog::info("y_pred contains exclude_outputs, returning");
    return false;
  }

  spdlog::warn("Target detected at {}-th frame in a batch of {} frames",
               positive_y_preds_idx[0], inference_batch_size, 0);
  spdlog::info("positive_y_preds_idx: {}",
               vector_to_string(positive_y_preds_idx));
  spdlog::info("Raw results are:\n{}", oss_raw_result.str());
  spdlog::info("and arithmetic average of raw results is:\n{}",
               oss_avg_result.str());
  spdlog::info("Timestamp of samples:");
  for (size_t i = 0; i < snaps.size(); ++i) {
    spdlog::info("snaps[{}]: {} ({})", i, snaps[i].unixepochns(),
                 unix_ts_to_iso_datetime(snaps[i].unixepochns() / 1000 / 1000));
  }
  spdlog::info("jpegs[{}: {}] of {} images will be used to build the GIF "
               "animation",
               positive_y_preds_idx[0],
               positive_y_preds_idx[0] + gif_frame_count, snaps.size());
  auto gif_width =
      settings.value("/inference/on_detected/gif/size/width"_json_pointer,
                     target_img_size.width);
  auto gif_height =
      settings.value("/inference/on_detected/gif/size/height"_json_pointer,
                     target_img_size.height);

  // The unit is "Time in 1/100ths of a second"
  unsigned int gif_frame_interval =
      settings.value(
          "/inference/on_detected/gif/frame_interval_ms"_json_pointer, 200) /
      10;
  vector<vector<uchar>> jpegs(gif_frame_count);
  // positive_y_preds_idx[0] stores the first non-zero item
  // index in y_pred. Note that y_pred[0] is NOT the prediction
  // of jpegs[0], but jpegs[pre_detection_size]
  // cv::imdecode(v, cv::IMREAD_COLOR);
  vector<Magick::Image> frames;
  for (size_t i = 0; i < gif_frame_count; ++i) {
    auto jpegs_idx = positive_y_preds_idx[0] + i;
    assert(jpegs_idx < snaps.size());
    cv::imencode(".jpg",
                 cv::Mat(zmq_payload_mat_size, zmq_payload_mat_type,
                         (void *)snaps[jpegs_idx].payload().data()),
                 jpegs[i]);
    frames.emplace_back(Magick::Blob(jpegs[i].data(), jpegs[i].size()));
    frames.back().animationDelay(gif_frame_interval);
    frames.back().resize(Magick::Geometry(gif_width, gif_height));
  }
  assert(frames.size() == gif_frame_count);
  string gif_path = settings.value(
      "/inference/on_detected/gif/path"_json_pointer, "/tmp/detected.gif");
  spdlog::info("Saving GIF file (size: {}x{}) to [{}]", gif_width, gif_height,
               gif_path);
  Magick::writeImages(frames.begin(), frames.end(), gif_path);
  execute_external_program_async();
  save_positive_outputs_as_jpeg(positive_y_preds_idx, jpegs);
  return true;
}

void inference_ev_loop() {
  constexpr size_t delay_ms = 10000;
  const size_t cooldown_ms =
      settings.value("/inference/on_detected/cooldown_sec"_json_pointer, 120) *
      1000;
  spdlog::info("inference_ev_loop() started, waiting for {}ms before start",
               delay_ms);
  // Otherwise we won't have any images of detected object in the GIF
  assert(inference_batch_size > 0);
  // Initial wait, just to spread out the stress at the beginning of the run
  interruptible_sleep(delay_ms);
  try {
    {
      std::scoped_lock lck{models_mtx, model_ids_mtx};
      models = load_models(model_ids);
    }

  } catch (const c10::Error &e) {
    spdlog::error("Error loading the model: {}", e.what());
    ev_flag = 1;
    spdlog::critical("The program must exit now");
  }

  Magick::InitializeMagick(nullptr);
  deque<SnapshotMsg> snap_deque;
  // snap_deque_size is larger than gif_frame_count by (inference_batch_size +
  // post_detection_size - 1) because the final gif is generated from a "sliding
  // window" of snap_deque_size
  const size_t snap_deque_size =
      gif_frame_count + inference_batch_size + post_detection_size - 1;
  vector<cv::Mat> images_mats(inference_batch_size);

  while (!ev_flag) {
    size_t sample_count = 0;
    while (sample_count < inference_batch_size && !ev_flag) {
      SnapshotMsg t;
      if (snapshot_pc_queue.try_dequeue(t)) {
        snap_deque.emplace_back(t);
        if (snap_deque.size() > snap_deque_size) {
          snap_deque.pop_front();
        }
        ++sample_count;
      } else {
        this_thread::sleep_for(999ms);
      }
    }
    if (ev_flag) {
      break;
    }
    if (snap_deque.size() < snap_deque_size) {
      spdlog::info(
          "snap_deque.size() too small ({}vs{}), queuing more elements "
          "before inference",
          snap_deque.size(), snap_deque_size);
      continue;
    }

    try {
      auto [raw_outputs, avg_output] = infer_images(models, snap_deque);
      update_last_inference_at();
      if (handle_inference_results(raw_outputs, avg_output, snap_deque)) {
        spdlog::info("inference_ev_loop() will sleep for {} ms after detection",
                     cooldown_ms);
        SnapshotMsg t;
        interruptible_sleep(cooldown_ms);
        for (size_t i = 0; i < image_queue_size; ++i) {
          if (!snapshot_pc_queue.try_dequeue(t))
            break;
        }
        snap_deque.clear();
      }
    } catch (const runtime_error &e) {
      spdlog::error("Inference failed: {}", e.what());
    }
  }
  spdlog::info("inference_ev_loop() exited gracefully");
}

void zeromq_ev_loop() {
  spdlog::info("zeromq_ev_loop() started");

  zmq::context_t context(1);
  zmq::socket_t subscriber(context, ZMQ_SUB);
  string zmq_address = settings.value("/inference/zeromq/address"_json_pointer,
                                      "tcp://127.0.0.1:4240");
  zmq_payload_mat_size = cv::Size(
      settings.value("/inference/zeromq/image_size/width"_json_pointer, 1920),
      settings.value("/inference/zeromq/image_size/height"_json_pointer, 1080));
  target_img_size = cv::Size(
      settings.value("/model/input_image_size/width"_json_pointer, 0),
      settings.value("/model/input_image_size/height"_json_pointer, 0));
  num_classes = settings.value("/model/num_classes"_json_pointer, 2);
  subscriber.connect(zmq_address);
  spdlog::info("ZeroMQ client connected to {}", zmq_address);
  constexpr int timeout = 5000;
  // subscriber.setsockopt(ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
  // subscriber.setsockopt(ZMQ_SUBSCRIBE, "", 0);
  subscriber.set(zmq::sockopt::rcvtimeo, timeout);
  subscriber.set(zmq::sockopt::subscribe, "");
  while (!ev_flag) {
    zmq::message_t message;
    SnapshotMsg msg;
    try {
      if (subscriber.recv(message, zmq::recv_flags::none) == false) {
        spdlog::error("subscriber.recv() returns false");
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
      spdlog::warn("ZeroMQ received empty message");
      continue;
    }
    if (msg.ParseFromArray(message.data(), message.size())) {
      if (!snapshot_pc_queue.try_enqueue(msg)) {
        spdlog::warn("try_enqueue() failed: snapshot_queue full, snapshots are "
                     "not being inferred fast enough");
      }
    } else {
      spdlog::error("Failed to parse ZeroMQ payload as SnapshotMsg");
    }
  }
  subscriber.close();
  spdlog::info("ZeroMQ connection closed");
  spdlog::info("zeromq_ev_loop() exited gracefully");
}
