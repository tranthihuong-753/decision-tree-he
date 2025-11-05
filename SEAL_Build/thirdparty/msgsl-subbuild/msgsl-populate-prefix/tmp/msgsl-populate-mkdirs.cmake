# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/decision-tree-SEAL/SEAL_Build/thirdparty/msgsl-src")
  file(MAKE_DIRECTORY "C:/decision-tree-SEAL/SEAL_Build/thirdparty/msgsl-src")
endif()
file(MAKE_DIRECTORY
  "C:/decision-tree-SEAL/SEAL_Build/thirdparty/msgsl-build"
  "C:/decision-tree-SEAL/SEAL_Build/thirdparty/msgsl-subbuild/msgsl-populate-prefix"
  "C:/decision-tree-SEAL/SEAL_Build/thirdparty/msgsl-subbuild/msgsl-populate-prefix/tmp"
  "C:/decision-tree-SEAL/SEAL_Build/thirdparty/msgsl-subbuild/msgsl-populate-prefix/src/msgsl-populate-stamp"
  "C:/decision-tree-SEAL/SEAL_Build/thirdparty/msgsl-subbuild/msgsl-populate-prefix/src"
  "C:/decision-tree-SEAL/SEAL_Build/thirdparty/msgsl-subbuild/msgsl-populate-prefix/src/msgsl-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/decision-tree-SEAL/SEAL_Build/thirdparty/msgsl-subbuild/msgsl-populate-prefix/src/msgsl-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/decision-tree-SEAL/SEAL_Build/thirdparty/msgsl-subbuild/msgsl-populate-prefix/src/msgsl-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
