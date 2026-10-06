# Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
#
# This file is part of the unit testing infrastructure for Theseus
#
# SPDX-License-Identifier: BSD-3-Clause
#
#################
# Example on how to add unit tests
# add_executable(suite_test_exe
#     unit_test_main.cpp
#     suite_tests.cpp
# )
#
# Run the whole suite of tests like this:
# add_test(NAME test_suite_name COMMAND suite_test_exe)
# Select a single test from the suite like this:
# add_test(NAME suite::test1 COMMAND suite_test_exe test1)
#

# Unit testing self-test suite
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(NAME ValidationRunnerTests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/validation_runner_tests.py)

add_test(
    NAME RunTheseusHarnessTests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_theseus_harness_tests.py
)

add_test(
    NAME CompareVizTests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/compare_viz_tests.py
)

add_executable(unit_testing_stub
    unit_test_main.cpp
    test_stub.cpp
)

add_test(NAME UnitTestingInfrastructure COMMAND unit_testing_stub)
add_test(NAME SingleUnitTestInfrastructure COMMAND unit_testing_stub unit_testing_funcs)

# Gas Physics testing suite
add_executable(gas_tests
   unit_test_main.cpp
   eos_tests.cpp
   transport_tests.cpp
   gas_model_tests.cpp
)

target_include_directories(gas_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

# Propagate SUTHERLAND compile definition to gas_tests if enabled
if(SUTHERLAND)
    target_compile_definitions(gas_tests PRIVATE SUTHERLAND)
endif()

target_link_libraries(gas_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)

add_test(NAME GasPhysicsTestingSuite COMMAND gas_tests)

# State unit testing suite
add_executable(state_tests
   unit_test_main.cpp
   state_tests.cpp
)

target_include_directories(state_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(state_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)

add_test(NAME StateTestingSuite COMMAND state_tests)

# Visualization configuration tests
add_executable(visualization_config_tests
   unit_test_main.cpp
   visualization_config_tests.cpp
)

target_include_directories(visualization_config_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

add_test(NAME VisualizationConfigTests COMMAND visualization_config_tests)

add_executable(visualization_mesh_tests
   unit_test_main.cpp
   visualization_mesh_tests.cpp
)

target_link_libraries(visualization_mesh_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)

add_test(NAME VisualizationMeshTests COMMAND visualization_mesh_tests)

add_executable(checkpoint_config_tests
   unit_test_main.cpp
   checkpoint_config_tests.cpp
)

target_include_directories(checkpoint_config_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

add_test(NAME CheckpointConfigTests COMMAND checkpoint_config_tests)

add_executable(time_step_control_tests
   unit_test_main.cpp
   time_step_control_tests.cpp
)

target_include_directories(time_step_control_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(time_step_control_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)

add_test(NAME TimeStepControlTests COMMAND time_step_control_tests)

# Axisymmetry configuration and mesh contract tests
add_executable(axisymmetry_config_tests
   unit_test_main.cpp
   axisymmetry_config_tests.cpp
)

target_include_directories(axisymmetry_config_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(axisymmetry_config_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)

add_test(NAME AxisymmetryConfigTests COMMAND axisymmetry_config_tests)

add_executable(axisymmetric_geometry_tests
   unit_test_main.cpp
   axisymmetric_geometry_tests.cpp
)

target_include_directories(axisymmetric_geometry_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)
if(ENABLE_CUDA)
    set_source_files_properties(
        axisymmetric_geometry_tests.cpp
        PROPERTIES LANGUAGE CUDA
    )

    target_include_directories(axisymmetric_geometry_tests PRIVATE
        ${MPI_CXX_INCLUDE_DIRS}
    )

    target_compile_options(axisymmetric_geometry_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-extended-lambda
            --expt-relaxed-constexpr
        >
    )
endif()

target_link_libraries(axisymmetric_geometry_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)

add_test(NAME AxisymmetricGeometryTests COMMAND axisymmetric_geometry_tests)

# Riemann solvers tests
add_executable(rflux_tests
   unit_test_main.cpp
   test_riemann_fluxes.cpp
)

target_include_directories(rflux_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(rflux_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)

add_test(NAME RiemannTestingSuite COMMAND rflux_tests)

if(THESEUS_WITH_PLATO)

   # Table lookup unit testing suite
   add_executable(table_lookup_tests
      unit_test_main.cpp
      table_lookup_test.cpp
   )

   if(ENABLE_CUDA)
	 set_source_files_properties(
		table_lookup_test.cpp
         	PROPERTIES LANGUAGE CUDA
    	 )

    	 target_include_directories(table_lookup_tests PRIVATE
              ${MPI_CXX_INCLUDE_DIRS}
    	 )

         target_compile_options(table_lookup_tests PRIVATE
              $<$<COMPILE_LANGUAGE:CUDA>:
                --expt-extended-lambda
                --expt-relaxed-constexpr
               >
         )
   endif()

   target_compile_definitions(table_lookup_tests PRIVATE USE_PLATO)
   if(SUTHERLAND)
     target_compile_definitions(table_lookup_tests PRIVATE SUTHERLAND)
   endif()

   target_include_directories(table_lookup_tests
         PRIVATE
         ${PROJECT_SOURCE_DIR}/include
         ${CMAKE_PREFIX_PATH}/include
         ${PROJECT_BINARY_DIR}/generated
   )

   target_link_directories(table_lookup_tests
         PUBLIC ${CMAKE_PREFIX_PATH}/lib
   )

   target_link_libraries(table_lookup_tests
          PRIVATE
          ${THESEUS_MFEM_TARGET}
          MPI::MPI_CXX
          plato
        )

   if(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES)
          target_link_libraries(table_lookup_tests PRIVATE
            ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}
          )
        endif()
        if(CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES)
          target_link_directories(table_lookup_tests PRIVATE
          ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}
        )
  endif()

        add_test(NAME ThermoTableTestingSuite COMMAND table_lookup_tests)
endif()

add_executable(radial_profile_tests radial_profile_tests.cpp)
target_include_directories(radial_profile_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME RadialProfileTests COMMAND radial_profile_tests
  ${PROJECT_SOURCE_DIR}/TestCases/Axisymmetric/NavierStokes/PXChamber/Tuvw_jet_inlet_profile.dat)

# Host conversion API uses an analytic LTE fixture; no PLATO dependency.
add_executable(physical_state_tests unit_test_main.cpp physical_state_tests.cpp)
target_include_directories(physical_state_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(physical_state_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
add_test(NAME PhysicalStateConversionTests COMMAND physical_state_tests)

if(THESEUS_WITH_PLATO)
  add_test(NAME PhysicalStatePLATOTests COMMAND table_lookup_tests plato_PhysicalState_conversion_test)
endif()

foreach(transport_variant IN ITEMS standard sutherland)
  set(property_test_target gas_property_tests_${transport_variant})
  add_executable(${property_test_target} unit_test_main.cpp gas_property_tests.cpp)
  target_include_directories(${property_test_target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(${property_test_target} PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
  if(transport_variant STREQUAL "sutherland")
    target_compile_definitions(${property_test_target} PRIVATE SUTHERLAND)
  endif()
  add_test(NAME GasPropertyTests_${transport_variant} COMMAND ${property_test_target})
endforeach()

add_executable(visualization_fields_tests visualization_fields_tests.cpp)
target_include_directories(visualization_fields_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(visualization_fields_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
add_test(NAME VisualizationFieldsTests COMMAND visualization_fields_tests)
