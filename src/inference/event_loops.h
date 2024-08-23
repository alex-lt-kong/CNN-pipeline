#ifndef CP_EVENT_LOOPS_H
#define CP_EVENT_LOOPS_H

namespace CnnPipeline::EventLoops {

void inference_ev_loop();
void zeromq_ev_loop();

} // namespace CnnPipeline::EventLoops
#endif // CP_EVENT_LOOPS_H
