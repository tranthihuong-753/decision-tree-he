// SEAL version: 4.1.2

#include "seal/seal.h"
#include <iostream>

int main() {
    std::cout << "SEAL version: "
              << SEAL_VERSION_MAJOR << "."
              << SEAL_VERSION_MINOR << "."
              << SEAL_VERSION_PATCH << std::endl;
    return 0;
}
