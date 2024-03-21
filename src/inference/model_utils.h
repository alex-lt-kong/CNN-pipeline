#ifndef VBCP_MODEL_UTILS_H
#define VBCP_MODEL_UTILS_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <opencv2/opencv.hpp>
#include <torch/script.h> // One-stop header.
#pragma GCC diagnostic pop

#define NUM_OUTPUT_CLASSES 2

const cv::Size target_img_size = cv::Size(640, 360); // size is in (w, h)

std::vector<torch::jit::script::Module>
load_models(const std::vector<std::string> &model_ids,
            const std::string &device_string = "cuda:0");

torch::Tensor cv_mat_to_tensor(cv::Mat image);

std::string tensor_to_string_like_pytorch(const torch::Tensor &t,
                                          const long index,
                                          const long ele_count);

#endif // VBCP_MODEL_UTILS_H
