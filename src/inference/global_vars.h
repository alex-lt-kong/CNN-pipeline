#ifndef CP_GLOBAL_VARS_H
#define CP_GLOBAL_VARS_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "inference_result.pb.h"
#include "snapshot.pb.h"
#pragma GCC diagnostic pop
#include "percentile_tracker.h"

#include <torch/torch.h> // all-in-one torch header
#include <nlohmann/json.hpp>
#include <readerwriterqueue/readerwritercircularbuffer.h>

#include <mutex>
#include <csignal>
#include <string>
#include <vector>

namespace CnnPipeline::GlobalVariables {

extern volatile sig_atomic_t ev_flag;

extern std::mutex ext_program_mtx;
extern std::mutex swagger_mtx;
extern std::mutex models_mtx;
extern std::mutex model_ids_mtx;

extern moodycamel::BlockingReaderWriterCircularBuffer<SnapshotMsg>
snapshot_pc_queue;
extern moodycamel::BlockingReaderWriterCircularBuffer<InferenceResultMsg>
inference_result_pc_queue;

extern size_t inference_batch_size;
extern size_t pc_queue_size;

extern nlohmann::json settings;
extern std::vector<std::string> model_ids;

extern std::vector<std::string> model_ids;
extern PercentileTracker<float> pt;
extern std::vector<torch::jit::script::Module> models;

extern std::string ts_model_path;

extern std::string cuda_device_string;

} // namespace CnnPipeline::GlobalVariables

#endif // CP_GLOBAL_VARS_H
