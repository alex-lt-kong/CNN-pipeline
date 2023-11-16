
#include "ErrorHandler.hpp"

ErrorHandler::ErrorHandler(
    const std::shared_ptr<oatpp::data::mapping::ObjectMapper> &objectMapper)
    : m_objectMapper(objectMapper) {}

std::shared_ptr<ErrorHandler::OutgoingResponse>
ErrorHandler::handleError(const Status &status, const oatpp::String &message,
                          const Headers &headers) {
  auto error = RespDto::createShared();
  error->success = false;
  error->responseText = message;

  auto response =
      ResponseFactory::createResponse(status, error, m_objectMapper);

  for (const auto &pair : headers.getAll()) {
    response->putHeader(pair.first.toString(), pair.second.toString());
  }

  return response;
}