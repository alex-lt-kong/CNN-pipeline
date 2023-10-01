#include <string>

void initialize_rest_api(std::string host, int port,
                         std::string advertised_host);
void finalize_rest_api();