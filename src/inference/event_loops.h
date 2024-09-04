#ifndef CP_EVENT_LOOPS_H
#define CP_EVENT_LOOPS_H

namespace CnnPipeline::EventLoops {

void inference_ev_loop();
// Consumer image data from upstream data provider
void zeromq_consumer_ev_loop();
// Produce image and inference result to downstream data user
void zeromq_producer_ev_loop();

} // namespace CnnPipeline::EventLoops
#endif // CP_EVENT_LOOPS_H
