# Install script for directory: C:/hu/decision-tree-he/SEAL/native/src/seal/util

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SEAL-4.1/seal/util" TYPE FILE FILES
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/blake2.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/blake2-impl.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/clang.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/clipnormal.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/common.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/croots.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/defines.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/dwthandler.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/fips202.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/galois.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/gcc.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/globals.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/hash.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/hestdparms.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/iterator.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/locks.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/mempool.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/msvc.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/numth.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/pointer.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/polyarithsmallmod.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/polycore.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/rlwe.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/rns.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/scalingvariant.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/ntt.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/streambuf.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/uintarith.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/uintarithmod.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/uintarithsmallmod.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/uintcore.h"
    "C:/hu/decision-tree-he/SEAL/native/src/seal/util/ztools.h"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/hu/decision-tree-he/SEAL_Build/native/src/seal/util/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
