#define FMT_HEADER_ONLY

#include "../global_vars.h"
#include "../model_utils.h"
#include "../utils.h"

#include <cstdlib>
#include <cxxopts.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <thread>

using namespace cv;
using namespace std;
using json = nlohmann::json;
namespace GV = CnnPipeline::GlobalVariables;
namespace MU = CnnPipeline::ModelUtils;

static volatile sig_atomic_t e_flag = 0;

torch::Tensor get_tensor_from_mat_vector(vector<Mat> &images,
                                         Size target_img_size) {
  std::vector<torch::Tensor> tensor_vec;

  for (const auto &image : images) {
    tensor_vec.push_back(MU::cv_mat_to_tensor(image, target_img_size));
  }
  return torch::stack(tensor_vec);
}

void overlay_result_to_frame(int frame_idx, vector<Mat> &frames, int batch_idx,
                             vector<at::Tensor> &raw_outputs,
                             at::Tensor &avg_output, int y_pred) {

  Scalar color = Scalar(255, 255, 255);

  // The avg_output should be one-dimensional
  assert(avg_output[batch_idx].sizes().size() == 1);

  string oss_str;

  Point cords = Point(8, 48);
  float fontScale = 0.6;
  putText(frames[batch_idx], "frame " + to_string(frame_idx), cords,
          FONT_HERSHEY_DUPLEX, fontScale, Scalar(0, 0, 0), 4 * fontScale,
          LINE_8, false);
  putText(frames[batch_idx], "frame " + to_string(frame_idx), cords,
          FONT_HERSHEY_DUPLEX, fontScale, color, fontScale, LINE_8, false);

  for (size_t i = 0; i < raw_outputs.size(); ++i) {
    cords.y += fontScale * 26;
    ostringstream oss;
    oss << "raw_outputs[" << i << "]: " << raw_outputs[i][batch_idx];
    oss_str = std::regex_replace(oss.str(), std::regex("\n"), " ");
    putText(frames[batch_idx], oss_str, cords, FONT_HERSHEY_DUPLEX, fontScale,
            Scalar(0, 0, 0), 4 * fontScale, LINE_8, false);
    putText(frames[batch_idx], oss_str, cords, FONT_HERSHEY_DUPLEX, fontScale,
            color, fontScale, LINE_8, false);
  }

  ostringstream oss;
  oss << "avg_output:     " << avg_output[batch_idx];
  oss_str = std::regex_replace(oss.str(), std::regex("\n"), " ");
  oss.clear();
  cords.y += fontScale * 26;
  putText(frames[batch_idx], oss_str, cords, FONT_HERSHEY_DUPLEX, fontScale,
          Scalar(0, 0, 0), 4 * fontScale, LINE_8, false);
  putText(frames[batch_idx], oss_str, cords, FONT_HERSHEY_DUPLEX, fontScale,
          color, fontScale, LINE_8, false);

  if (y_pred == 0)
    color = Scalar(0, 0, 255);
  else if (y_pred == 2)
    color = Scalar(0, 255, 0);
  fontScale = 3.5;
  cords.y += fontScale * 26;
  putText(frames[batch_idx], to_string(y_pred), cords, FONT_HERSHEY_DUPLEX,
          fontScale, Scalar(0, 0, 0), 4 * fontScale, LINE_8, false);
  putText(frames[batch_idx], to_string(y_pred), cords, FONT_HERSHEY_DUPLEX,
          fontScale, color, fontScale, LINE_8, false);
}

int main(int argc, const char *argv[]) {
  // install_signal_handler();
  install_signal_handler(&e_flag);
  cxxopts::Options options(
      argv[0],
      fmt::format("Video classifier (git commit: {})", GIT_COMMIT_HASH));
  string configPath, srcVideoPath;
  int output_sample_interval;
  filesystem::path dst_video_dir = filesystem::temp_directory_path();
  string dst_video_base_name = "video";
  size_t batch_size;

  // clang-format off
  options.add_options()("h,help", "Print help message")
    ("s,src-video-path", "Source video path", cxxopts::value<string>()->default_value(srcVideoPath))
    ("d,dst-video-dir", "Directory video path", cxxopts::value<string>()->default_value(dst_video_dir.string()))
    ("n,dst-video-base-name", "Basename of the output video excluding '.mp4', classification will be appended to the base name", cxxopts::value<string>()->default_value(dst_video_base_name))
    ("i,output-sample-interval", "A sample image will be saved every this number of frames", cxxopts::value<int>()->default_value("10"))
    ("b,batch-size", "Batch size of frame inference.", cxxopts::value<int>()->default_value("10"))
    ("c,config-path", "JSON configuration file path", cxxopts::value<string>()->default_value(configPath));
  // clang-format on
  auto result = options.parse(argc, argv);
  if (result.count("help") || !result.count("config-path") ||
      !result.count("src-video-path")) {
    cout << "getBuildInformation():\n" << getBuildInformation() << endl;
    cout << options.help() << "\n";
    return 0;
  }
  configPath = result["config-path"].as<string>();
  srcVideoPath = result["src-video-path"].as<string>();
  dst_video_dir = result["dst-video-dir"].as<string>();
  output_sample_interval = result["output-sample-interval"].as<int>();
  batch_size = result["batch-size"].as<int>();
  dst_video_base_name = result["dst-video-base-name"].as<string>();
  if (batch_size <= 0 || output_sample_interval < 1) {
    cerr << options.help() << "\n";
    cerr << "Invalid batch-size or output-sample-interval" << endl;
    return EXIT_FAILURE;
  }
  auto frames_dir = dst_video_dir / dst_video_base_name;

  if (!filesystem::exists(frames_dir)) {
    try {
      filesystem::create_directory(frames_dir);
    } catch (const filesystem::filesystem_error &ex) {
      spdlog::error(
          "Error creating directory: {}, program will exit prematurely",
          ex.what());
      return EXIT_FAILURE;
    }
  }

  ifstream f(configPath);

  json settings = json::parse(f);

  vector<string> model_ids = settings.value(
      "/inference/initial_model_ids"_json_pointer, vector<string>{"0"});
  GV::ts_model_path =
      settings.value("/model/ts_model_path"_json_pointer, string(""));
  static Size target_img_size =
      Size(settings.value("/model/input_image_size/width"_json_pointer, 0),
           settings.value("/model/input_image_size/height"_json_pointer, 0));
  static int numClasses = settings.value("/model/num_classes"_json_pointer, 1);
  vector<size_t> frameCountByOutput(numClasses, 0);
  string cuda_device_string =
      settings.value("/inference/cuda_device"_json_pointer, "cuda:0");
  auto models =
      MU::load_models(model_ids, GV::ts_model_path, cuda_device_string);

  cuda::GpuMat dFrame;
  vector<Mat> hFrames;

  Ptr<cv::cudacodec::VideoReader> dReader;

  try {
    dReader = cudacodec::createVideoReader(srcVideoPath);
  } catch (const cv::Exception &e) {
    spdlog::error("cudacodec::createVideoReader({}) failed: {}", srcVideoPath,
                  e.what());
    return EXIT_FAILURE;
  }

  if (!dReader->nextFrame(dFrame)) {
    spdlog::error("Failed to read frame from source {}", srcVideoPath);
    return EXIT_FAILURE;
  }

  auto temp_video_path = filesystem::temp_directory_path() /
                         ("temp_" + to_string(time(nullptr)) + ".mp4");

  Ptr<cudacodec::VideoWriter> dWriter = cudacodec::createVideoWriter(
      temp_video_path.string(), dFrame.size(), cudacodec::Codec::H264, 45.0,
      cudacodec::ColorFormat::BGR);
  dReader.release();
  dReader = cudacodec::createVideoReader(srcVideoPath);
  dReader->set(cv::cudacodec::ColorFormat::BGR);
  size_t frame_count = 0;
  while (!e_flag) {
    if (!dReader->nextFrame(dFrame)) {
      spdlog::info("dReader->nextFrame(dFrame) is False");
      break;
    } else if (frame_count % output_sample_interval == 0) {
      spdlog::info("frame_count: {:>5}, size(): {}x{}, channels(): {}",
                   frame_count, dFrame.size().width, dFrame.size().height,
                   dFrame.channels());
    }
    Mat hFrame;
    dFrame.download(hFrame);
    ++frame_count;
    hFrames.push_back(hFrame);

    if (hFrames.size() < batch_size)
      continue;
    auto imgs_tensor = get_tensor_from_mat_vector(hFrames, target_img_size);
    vector<torch::jit::IValue> input(1);
    input[0] = imgs_tensor.to(cuda_device_string);
    // imgs_tensor.sizes()[0] stores number of images
    at::Tensor avg_output = torch::zeros({imgs_tensor.sizes()[0], numClasses});
    avg_output = avg_output.to(cuda_device_string);
    vector<at::Tensor> raw_outputs(models.size());
    for (size_t i = 0; i < models.size(); ++i) {
      auto y = models[i].forward(input).toTensor();
      // Normalize the output, otherwise one model could have (unexpected)
      // outsized impact on the final result
      // Ref:
      // https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
      auto y_min = at::min(y);
      raw_outputs[i] = 2 * ((y - y_min) / (at::max(y) + 0.000001 - y_min)) - 1;
      avg_output += raw_outputs[i];
    }
    at::Tensor y_preds = torch::argmax(avg_output, 1);
    // cout << tensor_to_string_like_pytorch(y_pred, 0, batchSize) << endl;
    for (size_t i = 0; i < hFrames.size(); ++i) {
      int y_pred = y_preds[i].item<int>();
      // frames are handled in batches, so frame_count here is always a multiple
      // of batch_size.
      auto derived_real_frame_count = frame_count - batch_size + i;
      if (derived_real_frame_count % output_sample_interval == 0) {
        string padded_frame_count =
            string(5 - to_string(derived_real_frame_count).length(), '0') +
            to_string(derived_real_frame_count);
        filesystem::path jpg_path =
            frames_dir / (dst_video_base_name + "_" + padded_frame_count + "_" +
                          to_string(y_pred) + ".jpg");
        ofstream out_file(jpg_path, ios::binary);
        vector<uchar> jpeg_data;
        imencode(".jpg", hFrames[i], jpeg_data);
        if (!out_file) {
          spdlog::error("Failed to open the file: {}", jpg_path.native());
        } else {
          out_file.write((char *)jpeg_data.data(), jpeg_data.size());
          out_file.close();
        }
      }
      overlay_result_to_frame(frame_count - batch_size + i, hFrames, i,
                              raw_outputs, avg_output, y_pred);
      dFrame.upload(hFrames[i]);
      dWriter->write(dFrame);
      ++frameCountByOutput[y_pred];
    }
    hFrames.clear();
  }
  dWriter->release();
  spdlog::info("dWriter->release()ed");
  dReader.release();
  spdlog::info("dReader->release()ed");
  dst_video_base_name += "_";
  for (int i = 0; i < numClasses; ++i) {
    if (frameCountByOutput[i] > 0)
      dst_video_base_name += to_string(i) + "-";
  }
  dst_video_base_name =
      dst_video_base_name.substr(0, dst_video_base_name.size() - 1);
  auto dst_video_path = dst_video_dir / (dst_video_base_name + ".mp4");
  try {
    filesystem::copy_file(temp_video_path, dst_video_path,
                          filesystem::copy_options::overwrite_existing);
    filesystem::remove(temp_video_path);
    spdlog::info("dstVideo moved to: [{}]", dst_video_path.string());
  } catch (filesystem::filesystem_error &e) {
    spdlog::error("Unable to copy/remove dstVide", e.what());
  }
  try {
    filesystem::rename(frames_dir, dst_video_dir / dst_video_base_name);
  } catch (filesystem::filesystem_error &e) {
    spdlog::error("Unableto rename frame_dir from {} to {}: {}",
                  frames_dir.string(),
                  (dst_video_dir / dst_video_base_name).string(), e.what());
  }
  return 0;
}
