#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp> // Include the OpenCV header for image processing
#include <torch/script.h>     // One-stop header.

int main(int argc, const char *argv[]) {
  if (argc != 3) {
    std::cerr
        << "usage: model.cpp <path-to-exported-script-module> <image-path>\n";
    return -1;
  }
  torch::jit::script::Module module;
  try {

    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    std::cerr << e.what() << std::endl;
    return -1;
  }
  module.eval();
  module.to(torch::kCUDA);
  /*
  3. Preprocess the input image:
  */
  // Load and preprocess the input image using OpenCV
  cv::Mat image = cv::imread(argv[2]);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  cv::resize(image, image, cv::Size(426, 224));
  at::Tensor tensor_image =
      torch::from_blob(image.data, {image.rows, image.cols, 3}, at::kByte);

  tensor_image = tensor_image.to(at::kFloat).div(255).unsqueeze(0);
  tensor_image = tensor_image.permute({0, 3, 1, 2});
  tensor_image.sub_(0.5).div_(0.5);

  tensor_image = tensor_image.to(torch::kCUDA);
  // Execute the model and turn its output into a tensor.
  auto output = module.forward({tensor_image}).toTensor();
  std::cout << "raw output: " << output.cpu() << std::endl;
  /*std::cout << "pred: " << std::get<0>(torch::max(output.cpu(), 1, true))
            << std::endl;*/
}