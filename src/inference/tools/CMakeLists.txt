include(${PROJECT_SOURCE_DIR}/cmake/proto.cmake)

add_executable(torch_playground torch_playground.cpp)
target_link_libraries(torch_playground torch_cpu c10)

add_executable(zeromq_test_consumer zeromq_test_consumer.cpp
  ${PROTO_INFER_SRCS} ${PROTO_INFER_HDRS})
target_link_libraries(zeromq_test_consumer
  utils
  spdlog zmq 
  ${Protobuf_LIBRARIES}
)

add_executable(classify_images classify_images.cpp)
target_link_libraries(classify_images
  model_utils global_vars
  ${OpenCV_LIBS} 
  ${Protobuf_LIBRARIES}
  torch_cpu c10
  spdlog
)

add_executable(classify_video classify_video.cpp)
target_link_libraries(classify_video
  model_utils global_vars utils
  ${OpenCV_LIBS}
  ${Protobuf_LIBRARIES}
  torch_cpu c10
  spdlog
)

add_executable(three_class_inference_consumer
  three_class_inference_consumer.cpp
  ${PROTO_INFER_SRCS} ${PROTO_INFER_HDRS}
)
target_link_libraries(three_class_inference_consumer
  utils
  ${OpenCV_LIBS}
  ${Protobuf_LIBRARIES}
  ${ImageMagick_LIBRARIES}
  spdlog zmq
)
