#include "global_vars.h"

using namespace std;
namespace CnnPipeline::GlobalVariables {

volatile sig_atomic_t ev_flag = 0;

mutex ext_program_mtx;
mutex swagger_mtx;
mutex models_mtx;
mutex model_ids_mtx;

size_t inference_batch_size;
const size_t pre_detection_size = 6;
const size_t post_detection_size = 6;
size_t gif_frame_count;
size_t image_queue_size;

moodycamel::BlockingReaderWriterCircularBuffer<SnapshotMsg> snapshot_pc_queue =
    moodycamel::BlockingReaderWriterCircularBuffer<SnapshotMsg>(0);

nlohmann::json settings;
vector<string> model_ids;
PercentileTracker<float> pt = PercentileTracker<float>(10000);

vector<torch::jit::script::Module> models;
std::string ts_model_path;

std::string last_inference_at;

std::string cuda_device_string;
} // namespace CnnPipeline::GlobalVariables