#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H

#include <opencv2/opencv.hpp>
#include <torch/script.h> // One-stop header.

#define NUM_OUTPUT_CLASSES 2

const cv::Size target_img_size = cv::Size(426, 224); // size is in (w, h)
extern std::string torch_script_serialization;

std::vector<torch::jit::script::Module>
load_models(const std::vector<std::string> &model_ids,
            const std::string &device_string = "cuda:0");

torch::Tensor cv_mat_to_tensor(cv::Mat image);

std::string tensor_to_string_like_pytorch(const torch::Tensor &t,
                                          const long index,
                                          const long ele_count);

#endif /* MODEL_UTILS_H */
