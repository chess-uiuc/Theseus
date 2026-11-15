#!/usr/bin/env python3
"""Verify real source-target language properties across CMake directory scopes."""
from pathlib import Path
import subprocess
import tempfile
import unittest


class SourceKernelLanguageTests(unittest.TestCase):
    def test_registration_translation_unit_is_cuda_in_tests_directory(self):
        source = Path(__file__).resolve().parents[1]
        definitions = (source/'tests/CMakeLists.txt').read_text()
        block = definitions.split('# General physical-source evaluation.\n', 1)[1]
        block = block.split('# Source kernel language regression', 1)[0]
        # Exercise the production target definition in a real child directory.
        block = block.replace('source_terms_tests.cpp',
                              '"${REPOSITORY_SOURCE_DIR}/tests/source_terms_tests.cpp"')
        with tempfile.TemporaryDirectory(prefix='source-language-', dir=Path.cwd()) as directory:
            root = Path(directory)
            (root/'child').mkdir()
            (root/'CMakeLists.txt').write_text('''
cmake_minimum_required(VERSION 3.18)
project(SourceLanguageScope LANGUAGES CXX)
enable_testing()
set(REPOSITORY_SOURCE_DIR "''' + source.as_posix() + '''")
set(PROJECT_SOURCE_DIR "${REPOSITORY_SOURCE_DIR}")
set(ENABLE_CUDA ON)
add_library(test_mfem INTERFACE)
add_library(MPI::MPI_CXX INTERFACE IMPORTED)
set(THESEUS_MFEM_TARGET test_mfem)
set(MPIEXEC_EXECUTABLE "${CMAKE_COMMAND}")
# Reproduce the parent setting, which is insufficient in the child directory.
set_source_files_properties("${REPOSITORY_SOURCE_DIR}/src/SourceConfiguration.cpp"
  PROPERTIES LANGUAGE CUDA)
add_subdirectory(child)
''')
            (root/'child/CMakeLists.txt').write_text(block + '''
get_source_file_property(registration_language
  "${REPOSITORY_SOURCE_DIR}/src/SourceConfiguration.cpp" LANGUAGE)
if(NOT registration_language STREQUAL "CUDA")
  message(FATAL_ERROR "SourceConfiguration.cpp language in tests directory: ${registration_language}; expected CUDA")
endif()
get_source_file_property(test_language
  "${REPOSITORY_SOURCE_DIR}/tests/source_terms_tests.cpp" LANGUAGE)
if(NOT test_language STREQUAL "CUDA")
  message(FATAL_ERROR "Source fixture language: ${test_language}; expected CUDA")
endif()
# Reset only after checking CUDA selection, allowing CPU-only generation.
set_source_files_properties(
  "${REPOSITORY_SOURCE_DIR}/src/SourceConfiguration.cpp"
  "${REPOSITORY_SOURCE_DIR}/tests/source_terms_tests.cpp"
  PROPERTIES LANGUAGE CXX)
''')
            result = subprocess.run(['cmake', '-S', str(root), '-B', str(root/'build')],
                                    text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == '__main__':
    unittest.main()
