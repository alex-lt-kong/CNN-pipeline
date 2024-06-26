#include "global_vars.h"

using namespace std;
namespace CnnPipeline::GlobalVariables {

volatile sig_atomic_t ev_flag = 0;

mutex ext_program_mtx;
mutex swagger_mtx;
mutex models_mtx;
mutex model_ids_mtx;

// These three sizes can't be too large, otherwise GPUs with smaller VRAM
// may not run the program successfully
const size_t pre_detection_size = 6;
const size_t inference_batch_size = 24;
const size_t post_detection_size = 6;
const size_t gif_frame_count =
    pre_detection_size + inference_batch_size + post_detection_size;
const size_t image_queue_size = gif_frame_count * 4;

moodycamel::BlockingReaderWriterCircularBuffer<SnapshotMsg> snapshot_pc_queue =
    moodycamel::BlockingReaderWriterCircularBuffer<SnapshotMsg>(
        image_queue_size);

nlohmann::json settings;
vector<string> model_ids;
PercentileTracker<float> pt = PercentileTracker<float>(10000);

vector<torch::jit::script::Module> models;
std::string ts_model_path;

std::string last_inference_at;

std::string cuda_device_string;
} // namespace CnnPipeline::GlobalVariables