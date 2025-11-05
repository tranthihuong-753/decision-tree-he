# Install script for directory: C:/decision-tree-SEAL/SEAL/native/src/seal

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/SEAL")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SEAL-4.1/seal" TYPE FILE FILES
    "C:/decision-tree-SEAL/SEAL/native/src/seal/batchencoder.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/ciphertext.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/ckks.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/modulus.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/context.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/decryptor.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/dynarray.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/encryptionparams.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/encryptor.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/evaluator.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/galoiskeys.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/keygenerator.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/kswitchkeys.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/memorymanager.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/plaintext.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/publickey.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/randomgen.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/randomtostd.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/relinkeys.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/seal.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/secretkey.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/serializable.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/serialization.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/valcheck.h"
    "C:/decision-tree-SEAL/SEAL/native/src/seal/version.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/decision-tree-SEAL/SEAL_Build/native/src/seal/util/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/decision-tree-SEAL/SEAL_Build/native/src/seal/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
