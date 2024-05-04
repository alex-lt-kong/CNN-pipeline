#define FMT_HEADER_ONLY

#include "model_utils.h"

#include <getopt.h>
#include <nlohmann/json.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/script.h> // One-stop header.

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using json = nlohmann::json;

string torch_script_serialization;
static cv::Size target_img_size;

void print_usage(string binary_name) {

  cerr << "Usage: " << binary_name << " [OPTION]\n\n";

  cerr << "Options:\n"
       << "  --help,        -h        Display this help and exit\n"
       << "  --config-path, -c        JSON configuration file path\n"
       << "  --image-dir,   -d        Directory that is full of images!\n"
       << "  --model-ids,   -i        A comma-separate list of model IDs"
       << endl;
}

void parse_arguments(int argc, char **argv, string &image_path,
                     string &config_path, vector<string> &model_ids) {
  static struct option long_options[] = {
      {"config", required_argument, 0, 'c'},
      {"image-dir", required_argument, 0, 'd'},
      {"model-ids", required_argument, 0, 'i'},
      {"help", optional_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt, option_index = 0;

  while ((opt = getopt_long(argc, argv, "c:d:i:h", long_options,
                            &option_index)) != -1) {
    switch (opt) {
    case 'd':
      if (optarg != NULL) {
        image_path = string(optarg);
      }
      break;
    case 'c':
      if (optarg != NULL) {
        config_path = string(optarg);
      }
      break;
    case 'i':
      if (optarg != NULL) {
        // string t = string(optarg);
        // std::vector<std::string> substrings;
        std::stringstream ss(optarg);
        std::string substring;
        model_ids.clear();
        while (std::getline(ss, substring, ',')) {
          model_ids.push_back(substring);
        }
      }
      break;
    default:
      print_usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  if (image_path.empty() || config_path.empty()) {
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }
}

torch::Tensor get_tensor_from_img_dir(const string &image_dir) {
  std::vector<cv::String> imagePaths;
  cv::glob(image_dir, imagePaths);

  std::vector<cv::Mat> images;
  std::vector<torch::Tensor> tensor_vec;

  for (const auto &imagePath : imagePaths) {
    cv::Mat image = cv::imread(imagePath);
    tensor_vec.push_back(cv_mat_to_tensor(image, target_img_size));
  }
  return torch::stack(tensor_vec);
}

int main(int argc, char **argv) {
  spdlog::set_pattern("%Y-%m-%d %T.%e | %7l | %v");
  filesystem::path binaryPath = filesystem::canonical(argv[0]);
  filesystem::path parentPath = binaryPath.parent_path().parent_path();
  string config_path, image_path;
  vector<string> model_ids;

  parse_arguments(argc, argv, image_path, config_path, model_ids);

  // config_path = (parentPath / "config.json").string();
  ifstream f(config_path);

  json settings = json::parse(f);
  torch_script_serialization = settings.value(
      "/model/torch_script_serialization"_json_pointer, string(""));
  target_img_size = cv::Size(
      settings.value("/model/input_image_size/width"_json_pointer, 0),
      settings.value("/model/input_image_size/height"_json_pointer, 0));
  vector<torch::jit::script::Module> v16mms = load_models(model_ids);

  const size_t preview_ele_num = 5;
  size_t layer_count = 0;
  spdlog::info("Sample values from some layers from the first model:");
  for (const auto &pair : v16mms[0].named_parameters()) {
    ++layer_count;
    if (layer_count % 5 != 0) {
      continue;
    }
    const string &name = pair.name;
    const torch::Tensor &tensor = pair.value;
    ostringstream oss;
    oss << tensor.sizes();
    string tensorStr;
    if (tensor.sizes().size() <= 1) {
      tensorStr = tensor_to_string_like_pytorch(tensor, 0, preview_ele_num);
    } else if (tensor.sizes().size() <= 2) {
      tensorStr = tensor_to_string_like_pytorch(tensor[0], 0, preview_ele_num);
    } else if (tensor.sizes().size() <= 3) {
      tensorStr =
          tensor_to_string_like_pytorch(tensor[0][0], 0, preview_ele_num);
    } else if (tensor.sizes().size() <= 4) {
      tensorStr =
          tensor_to_string_like_pytorch(tensor[0][0][0], 0, preview_ele_num);
    } else {
      tensorStr =
          tensor_to_string_like_pytorch(tensor[0][0][0][0], 0, preview_ele_num);
    }
    spdlog::info("{}({}): {}", name, oss.str(), tensorStr);
  }

  spdlog::info("Loading and transforming image from {}", image_path);
  torch::Tensor images_tensor = get_tensor_from_img_dir(image_path);
  ostringstream oss;
  oss << images_tensor.sizes();

  time_t rounded_unix_time = (time(nullptr) - time_t(0)) / 100000;
  long h = rounded_unix_time % target_img_size.height;
  long w = rounded_unix_time % (target_img_size.width - preview_ele_num);
  spdlog::info("Image tensor ready, tensor shape: {}, sample values starting "
               "from (w{}, h{}):",
               oss.str(), w, h);

  for (int i = 0; i < images_tensor.sizes()[1]; ++i) {
    spdlog::info("{}", tensor_to_string_like_pytorch(images_tensor[0][i][h], w,
                                                     preview_ele_num));
  }
  vector<torch::jit::IValue> input(1);
  input[0] = images_tensor.to(torch::kCUDA);
  // images_tensor.sizes()[0] stores number of images
  // 2 is the num_classes member variable as defined in VGG16MinusMinus@model.py
  at::Tensor output =
      torch::zeros({images_tensor.sizes()[0],
                    settings.value("/model/num_classes"_json_pointer, 2)});
  output = output.to(torch::kCUDA);
  vector<at::Tensor> outputs(v16mms.size());
  spdlog::info("Running inference");
  for (size_t i = 0; i < v16mms.size(); ++i) {
    auto y = v16mms[i].forward(input).toTensor();
    // Normalize the output, otherwise one model could have (unexpected)
    // outsized impact on the final result
    // Ref:
    // https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    auto y_min = at::min(y);
    outputs[i] = 2 * ((y - y_min) / (at::max(y) + 0.000001 - y_min)) - 1;
    output += outputs[i];
  }
  spdlog::info("Done");
  oss.str("");
  oss << output;

  spdlog::info("Raw results from {} models are:", v16mms.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    oss.str("");
    oss << outputs[i];
    spdlog::info("\n{}", oss.str());
  }
  oss.str("");
  oss << output;
  spdlog::info("and arithmetic average of raw results is:\n{}", oss.str());
  at::Tensor y_pred = torch::argmax(output, 1);
  oss.str("");
  oss << y_pred;
  spdlog::info("y_pred: {}",
               tensor_to_string_like_pytorch(y_pred, 0, y_pred.sizes()[0]));
}
