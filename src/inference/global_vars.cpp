#include "global_vars.h"

using namespace std;
namespace CnnPipeline::GlobalVariables {

volatile sig_atomic_t ev_flag = 0;

mutex ext_program_mtx;
mutex swagger_mtx;
mutex models_mtx;
mutex model_ids_mtx;

size_t inference_batch_size;
size_t pc_queue_size;

moodycamel::BlockingReaderWriterCircularBuffer<SnapshotMsg> snapshot_pc_queue =
    moodycamel::BlockingReaderWriterCircularBuffer<SnapshotMsg>(0);
moodycamel::BlockingReaderWriterCircularBuffer<InferenceResultMsg>
    inference_result_pc_queue =
        moodycamel::BlockingReaderWriterCircularBuffer<InferenceResultMsg>(0);

nlohmann::json settings;
vector<string> model_ids;
PercentileTracker<float> pt = PercentileTracker<float>(10000);

vector<torch::jit::script::Module> models;
std::string ts_model_path;

std::string cuda_device_string;
} // namespace CnnPipeline::GlobalVariables