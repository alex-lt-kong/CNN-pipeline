#ifndef CP_OATPP_ENTRY_H
#define CP_OATPP_ENTRY_H

#include <string>

void initialize_rest_api(std::string host, int port,
                         std::string advertised_host);
void finalize_rest_api();

#endif // CP_OATPP_ENTRY_H
