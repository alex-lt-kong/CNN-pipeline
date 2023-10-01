#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H

#include <opencv2/opencv.hpp>
#include <torch/script.h> // One-stop header.

#define NUM_OUTPUT_CLASSES 2

std::vector<torch::jit::script::Module>
load_models(const std::string &torch_script_serialization,
            std::vector<std::string> model_ids);

torch::Tensor cv_mat_to_tensor(cv::Mat image, const cv::Size &target_img_size);

std::string tensor_to_string_like_pytorch(const torch::Tensor &t,
                                          const long index,
                                          const long ele_count);

#endif /* MODEL_UTILS_H */
