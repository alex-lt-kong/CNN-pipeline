#include "torch/csrc/api/include/torch/types.h"
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h> // One-stop header.

using namespace std;
using json = nlohmann::json;

void print_usage(string binary_name) {

  cerr << "Usage: " << binary_name << " [OPTION]\n\n";

  cerr << "Options:\n"
       << "  --help, -h             display this help and exit\n"
       << "  --config, -c           Path of the JSON config file\n"
       << "  --image-path, -p       Path of image to be inferenced" << endl;
}

void parse_arguments(int argc, char **argv, string &config_path,
                     string &image_path) {
  static struct option long_options[] = {
      {"config", required_argument, 0, 'c'},
      {"image-path", required_argument, 0, 'p'},
      {"help", optional_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt, option_index = 0;

  while ((opt = getopt_long(argc, argv, "c:p:h", long_options,
                            &option_index)) != -1) {
    switch (opt) {
    case 'c':
      if (optarg != NULL) {
        config_path = string(optarg);
      }
      break;
    case 'p':
      if (optarg != NULL) {
        image_path = string(optarg);
      }
      break;
    default:
      print_usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  if (config_path.empty() || image_path.empty()) {
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }
}

string tensor_to_string(const torch::Tensor &t, const long ele_count) {
  // Convert tensor elements to strings with scientific notation
  std::ostringstream oss;
  oss << std::scientific;
  oss << "tensor([";
  oss << std::setprecision(6);
  for (long i = 0; i < ele_count && i < t.sizes()[0]; ++i) {
    oss << t[i].item<float>();
    if (i < ele_count - 1 && i < t.sizes()[0] - 1) {
      oss << ", ";
    }
  }
  oss << "])";
  // tensor([-1.1809e-04, -1.7340e-03, -1.0064e-01, -2.2699e-09,  1.3654e-04])
  return oss.str();
}

int main(int argc, char **argv) {
  string config_path, image_path;
  parse_arguments(argc, argv, config_path, image_path);

  ifstream f(config_path);
  json settings = json::parse(f);

  torch::jit::script::Module v16mm;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    v16mm = torch::jit::load(settings["model"]["torch_script_serialization"]);
  } catch (const c10::Error &e) {
    cerr << "error loading the model\n";
    cerr << e.what() << endl;
    return -1;
  }

  // Iterate over the model parameters and print their values
  for (const auto &pair : v16mm.named_parameters()) {
    const string &name = pair.name;
    const torch::Tensor &tensor = pair.value;
    if (tensor.sizes().size() <= 1) {
      cout << name << "(" << tensor.sizes()
           << "): " << tensor_to_string(tensor, 5) << endl;
    } else if (tensor.sizes().size() <= 2) {
      cout << name << "(" << tensor.sizes()
           << "): " << tensor_to_string(tensor[0], 5) << endl;
    } else if (tensor.sizes().size() <= 3) {
      cout << name << "(" << tensor.sizes()
           << "): " << tensor_to_string(tensor[0][0], 5) << endl;
    } else if (tensor.sizes().size() <= 4) {
      cout << name << "(" << tensor.sizes()
           << "): " << tensor_to_string(tensor[0][0][0], 5) << endl;
    } else {
      cout << name << "(" << tensor.sizes()
           << "): " << tensor_to_string(tensor[0][0][0][0], 5) << endl;
    }
  }

  v16mm.eval();
  v16mm.to(torch::kCUDA);

  /*
  3. Preprocess the input image:
  */
  // Load and preprocess the input image using OpenCV
  cv::Mat image = cv::imread(image_path);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  cv::resize(image, image, cv::Size(426, 224));

  // Convert the OpenCV image to a Torch tensor
  torch::Tensor tensor_image = torch::from_blob(
      image.data, {1, image.rows, image.cols, 3}, torch::kByte);

  // Permute dimensions to (batch, channels, height, width)
  tensor_image = tensor_image.permute({0, 3, 1, 2});

  tensor_image = tensor_image.to(torch::kFloat32);

  // https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
  tensor_image[0][0] = tensor_image[0][0].sub(0.485).div(0.229);
  tensor_image[0][1] = tensor_image[0][1].sub(0.456).div(0.224);
  tensor_image[0][2] = tensor_image[0][2].sub(0.406).div(0.225);

  vector<torch::jit::IValue> input;
  input.emplace_back(tensor_image.to(torch::kCUDA));
  // Execute the model and turn its output into a tensor.
  at::Tensor output = v16mm.forward(input).toTensor();

  cout << "raw output: " << output.cpu() << endl;
  /*cout << "pred: " << get<0>(torch::max(output.cpu(), 1, true))
            << endl;*/
}