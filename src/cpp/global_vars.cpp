#include "global_vars.h"

using namespace std;

volatile sig_atomic_t ev_flag = 0;

mutex ext_program_mtx;
mutex swagger_mtx;
mutex models_mtx;
mutex model_ids_mtx;

const size_t gif_frame_count = 36;
const size_t inference_batch_size = 24;
const size_t pre_detection_size = 4;
const size_t image_queue_min_len =
    pre_detection_size + inference_batch_size + gif_frame_count;
const size_t image_queue_max_len = image_queue_min_len * 4;

moodycamel::BlockingReaderWriterCircularBuffer<SnapshotMsg> snapshot_queue =
    moodycamel::BlockingReaderWriterCircularBuffer<SnapshotMsg>(
        image_queue_max_len);

nlohmann::json settings;
vector<string> model_ids;
std::atomic<uint32_t> inference_interval_ms = 60000;
std::unordered_map<uint32_t, PercentileTracker<float>> pt_dict;

vector<torch::jit::script::Module> models;
std::string torch_script_serialization;

std::string last_inference_at;