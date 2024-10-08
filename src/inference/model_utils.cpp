#define FMT_HEADER_ONLY

#include "model_utils.h"
#include "global_vars.h"

#include <ATen/core/interned_strings.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <spdlog/spdlog.h>

#include <regex>
#include <sstream>

namespace CnnPipeline::ModelUtils {
using namespace std;

// const float target_img_means[] = {0.485, 0.456, 0.406};
// const float target_img_stds[] = {0.229, 0.224, 0.225};

vector<torch::jit::script::Module>
load_models(const vector<string> &model_ids, const string ts_model_path,
            const string cuda_device_string) {
  vector<torch::jit::script::Module> models;
  spdlog::info("A total of {} models will be loaded", model_ids.size());
  // Unfortunately, torch::jit::script::Module's RAII won't automatically
  // release the GPU memory used by it, so we need to call emptyCache() manually
  // to reclaim the unused memory.
  // Without RAII, if torch::jit::load(model_path, torch::kCUDA) throws
  // exception, emptyCache() might not be called after the loop, so we want to
  // call the function both before and after model loading.
  c10::cuda::CUDACachingAllocator::emptyCache();
  for (size_t i = 0; i < model_ids.size(); ++i) {
    string model_path =
        regex_replace(ts_model_path, regex("\\{id\\}"), model_ids[i]);
    spdlog::info("Deserializing {}-th model from {}", i, model_path);

    // https://pytorch.org/docs/stable/generated/torch.jit.load.html
    models.emplace_back(torch::jit::load(model_path, cuda_device_string));
    // models[i].to(cuda_device_string);
    models[i].eval();
  }
  c10::cuda::CUDACachingAllocator::emptyCache();
  return models;
}

torch::Tensor cv_mat_to_tensor(cv::Mat image, cv::Size target_img_size) {
  auto img_clone = image.clone();
  cv::cvtColor(image, img_clone, cv::COLOR_BGR2RGB);

  // cv::INTER_LINEAR is the OpenCV equivalent of
  // torchvision.transforms.InterpolationMode.BILINEAR
  // However, it is important to note that the resize() results from OpenCV
  // and PIL are not identical
  cv::resize(img_clone, img_clone, target_img_size, 0, 0, cv::INTER_LINEAR);

  // Convert the OpenCV image to a Torch tensor
  torch::Tensor image_tensor = torch::from_blob(
      img_clone.data, {img_clone.rows, img_clone.cols, 3}, torch::kByte);

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
  /*
  for (size_t i = 0; i < 3; ++i) {
    image_tensor[i] =
        image_tensor[i].sub(target_img_means[i]).div(target_img_stds[i]);
  }
  */
  return image_tensor;
}

string tensor_to_string_like_pytorch(const torch::Tensor &t, const long index,
                                     const long ele_count) {
  ostringstream oss;
  oss << "tensor([";
  bool small_non_zero_numbers = false;
  for (long i = index; i < index + ele_count && i < t.sizes()[0]; ++i) {
    if (t[i].abs().item<float>() < 0.000001 && t[i].abs().item<float>() != 0) {
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
} // namespace CnnPipeline::ModelUtils