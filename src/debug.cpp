#include "opencv2/core/cvstd.hpp"
#include "opencv2/imgproc.hpp"
#include <sstream>
#define FMT_HEADER_ONLY
#include "torch/csrc/api/include/torch/types.h"
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <torch/script.h> // One-stop header.

using namespace std;
using json = nlohmann::json;
const cv::Size target_img_size = cv::Size(426, 224); // size is in (w, h)
const float target_img_means[] = {0.485, 0.456, 0.406};
const float target_img_stds[] = {0.229, 0.224, 0.225};

void print_usage(string binary_name) {

  cerr << "Usage: " << binary_name << " [OPTION]\n\n";

  cerr << "Options:\n"
       << "  --help, -h             Display this help and exit\n"
       << "  --image-dir, -d        Directory that is full of images!" << endl;
}

void parse_arguments(int argc, char **argv, string &image_path) {
  static struct option long_options[] = {
      {"config", required_argument, 0, 'c'},
      {"image-dir", required_argument, 0, 'p'},
      {"help", optional_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt, option_index = 0;

  while ((opt = getopt_long(argc, argv, "d:h", long_options, &option_index)) !=
         -1) {
    switch (opt) {
    case 'd':
      if (optarg != NULL) {
        image_path = string(optarg);
      }
      break;
    default:
      print_usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  if (image_path.empty()) {
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }
}

string tensor_to_string_like_pytorch(const torch::Tensor &t, const long index,
                                     const long ele_count) {
  ostringstream oss;
  oss << "tensor([";
  bool small_non_zero_numbers = false;
  for (long i = index; i < index + ele_count && i < t.sizes()[0]; ++i) {
    if (t[i].abs().item<float>() < 0.0001 && t[i].abs().item<float>() != 0) {
      small_non_zero_numbers = true;
      break;
    }
  }
  if (small_non_zero_numbers) {
    oss << std::scientific << std::setprecision(4);
  } else {
    oss << std::fixed << std::setprecision(4);
  }

  for (long i = index; i < index + ele_count && i < t.sizes()[0]; ++i) {
    oss << t[i].item();
    if (i < index + ele_count - 1 && i < t.sizes()[0] - 1) {
      oss << ", ";
    }
  }
  oss << "])";
  /*
    Depending on if there are some very small numbers, there are two possible
    formats:
    tensor([ 1.2899,  1.2043,  1.0673, -0.2513, -0.9705])
    tensor([-1.1809e-04, -1.7340e-03, -1.0064e-01, -2.2699e-09,  1.3654e-04])
  */
  return oss.str();
}

torch::Tensor get_tensor_from_img_path(const string &image_path) {

  // image = Image.open(image_path).convert("RGB")
  cv::Mat image = cv::imread(image_path);
  // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  // cv::INTER_LINEAR is the OpenCV equivalent of
  // torchvision.transforms.InterpolationMode.BILINEAR
  // However, it is important to note that the resize() results from OpenCV
  // and PIL are not identical
  cv::resize(image, image, target_img_size, 0, 0, cv::INTER_LINEAR_EXACT);

  // Convert the OpenCV image to a Torch tensor
  torch::Tensor image_tensor = torch::from_blob(
      image.data, {1, image.rows, image.cols, 3}, torch::kByte);

  // permute() is used to "rearrange" dimensions. Before permute(),
  // tensor_image.sizes() is [1, 240, 432, 3], but we want
  // tensor_image.sizes() to be [1, 3, 240, 432]
  // image_tensor = image_tensor.permute({0, 3, 1, 2});

  // image_tensor = image_tensor.to(torch::kFloat32);
  // Mimic torchvision.transforms.ToTensor() which shrinks the range from
  // [0, 255] to [0, 1]
  // image_tensor /= 255.0;

  // Mimic
  // torchvision.transforms.Normalize(mean=target_img_means,
  // std=target_img_stds)
  for (size_t i = 0; i < 3; ++i) {
    image_tensor[0][i] =
        image_tensor[0][i].sub(target_img_means[i]).div(target_img_stds[i]);
  }
  return image_tensor;
}

torch::Tensor get_tensor_from_img_dir(const string &image_dir) {
  std::vector<cv::String> imagePaths;
  cv::glob(image_dir, imagePaths);

  std::vector<cv::Mat> images;
  std::vector<torch::Tensor> tensor_vec;

  for (const auto &imagePath : imagePaths) {
    cv::Mat image = cv::imread(imagePath);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // cv::INTER_LINEAR is the OpenCV equivalent of
    // torchvision.transforms.InterpolationMode.BILINEAR
    // However, it is important to note that the resize() results from OpenCV
    // and PIL are not identical
    cv::resize(image, image, target_img_size, 0, 0, cv::INTER_LINEAR);

    // Convert the OpenCV image to a Torch tensor
    torch::Tensor image_tensor =
        torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte);

    // permute() is used to "rearrange" dimensions. Before permute(),
    // tensor_image.sizes() is [240, 432, 3], but we want
    // tensor_image.sizes() to be [3, 240, 432]
    image_tensor = image_tensor.permute({2, 0, 1});

    image_tensor = image_tensor.to(torch::kFloat32);
    // Mimic torchvision.transforms.ToTensor() which shrinks the range from
    // [0, 255] to [0, 1]
    image_tensor /= 255.0;

    // Mimic
    // torchvision.transforms.Normalize(mean=target_img_means,
    // std=target_img_stds)
    for (size_t i = 0; i < 3; ++i) {
      image_tensor[i] =
          image_tensor[i].sub(target_img_means[i]).div(target_img_stds[i]);
    }
    tensor_vec.push_back(image_tensor);
  }
  return torch::stack(tensor_vec);
}

int main(int argc, char **argv) {
  spdlog::set_pattern("%Y-%m-%d %T.%e | %7l | %v");
  filesystem::path binaryPath = filesystem::canonical(argv[0]);
  filesystem::path parentPath = binaryPath.parent_path().parent_path();
  string config_path, image_path;

  parse_arguments(argc, argv, image_path);
  config_path = (parentPath / "config.json").string();
  ifstream f(config_path);
  json settings = json::parse(f);

  torch::jit::script::Module v16mm;
  try {
    spdlog::info("Desearilizing model from {}",
                 settings["model"]["torch_script_serialization"]);
    v16mm = torch::jit::load(settings["model"]["torch_script_serialization"]);
    spdlog::info("Model deserialized");
  } catch (const c10::Error &e) {
    cerr << "error loading the model\n";
    cerr << e.what() << endl;
    return -1;
  }

  const size_t preview_ele_num = 5;
  size_t layer_count = 0;
  spdlog::info("Sample values from some layers:");
  for (const auto &pair : v16mm.named_parameters()) {
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

  v16mm.to(torch::kCUDA);
  v16mm.eval();

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
  spdlog::info("Running inference");
  vector<torch::jit::IValue> input;
  input.emplace_back(images_tensor.to(torch::kCUDA));
  at::Tensor output = v16mm.forward(input).toTensor();
  oss.str("");
  oss << output;
  /*for (int i = 0; i < output.sizes()[0]; ++i) {
    spdlog::info("Done, raw output: {}",
                 tensor_to_string_like_pytorch(output[i], 0, 2));
  }*/
  spdlog::info("Done, raw output:\n{}", oss.str());
  at::Tensor y_pred = torch::argmax(output, 1);
  oss.str("");
  oss << y_pred;
  spdlog::info("y_pred: {}",
               tensor_to_string_like_pytorch(y_pred, 0, y_pred.sizes()[0]));
}
