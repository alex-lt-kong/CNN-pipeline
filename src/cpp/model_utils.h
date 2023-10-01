#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H
#include <torch/script.h> // One-stop header.

std::vector<torch::jit::script::Module>
load_models(const std::string &torch_script_serialization,
            std::vector<std::string> model_ids);

#endif /* MODEL_UTILS_H */
