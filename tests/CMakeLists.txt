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

set(AXISYMMETRIC_RESTART_ARGUMENT)
set(AXISYMMETRIC_TEST_DEVICE "cpu" CACHE STRING
    "MFEM device used by axisymmetric integration tests")

if(AXISYMMETRIC)
    set(AXISYMMETRIC_RESTART_ARGUMENT --axisymmetric)
endif()

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

add_test(
    NAME TimestepCFLIntegration
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/timestep_cfl_integration_test.py
            --executable $<TARGET_FILE:theseus>
            --source ${PROJECT_SOURCE_DIR}
            --mpiexec ${MPIEXEC_EXECUTABLE}
            --numproc-flag=${MPIEXEC_NUMPROC_FLAG}
)

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

if(AXISYMMETRIC)
    add_test(
        NAME AxisymmetricUniformFlowIntegration
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/axisymmetric_uniform_flow_integration_test.py
                --executable $<TARGET_FILE:theseus>
                --source ${PROJECT_SOURCE_DIR}
                --mpiexec ${MPIEXEC_EXECUTABLE}
                --numproc-flag=${MPIEXEC_NUMPROC_FLAG}
                --device ${AXISYMMETRIC_TEST_DEVICE}
    )
    add_test(
        NAME AxisymmetricEntropyWaveConvergence
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/axisymmetric_entropy_wave_convergence_test.py
                --executable $<TARGET_FILE:theseus>
                --source ${PROJECT_SOURCE_DIR}
                --mpiexec ${MPIEXEC_EXECUTABLE}
                --numproc-flag=${MPIEXEC_NUMPROC_FLAG}
    )
    add_test(
        NAME AxisymmetricInviscidSphereIntegration
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/axisymmetric_flow_over_sphere_integration_test.py
                --executable $<TARGET_FILE:theseus>
                --source ${PROJECT_SOURCE_DIR}
                --mpiexec ${MPIEXEC_EXECUTABLE}
                --numproc-flag=${MPIEXEC_NUMPROC_FLAG}
                --device ${AXISYMMETRIC_TEST_DEVICE}
                --case inviscid
    )
    add_test(
        NAME AxisymmetricViscousSphereIntegration
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/axisymmetric_flow_over_sphere_integration_test.py
                --executable $<TARGET_FILE:theseus>
                --source ${PROJECT_SOURCE_DIR}
                --mpiexec ${MPIEXEC_EXECUTABLE}
                --numproc-flag=${MPIEXEC_NUMPROC_FLAG}
                --device ${AXISYMMETRIC_TEST_DEVICE}
                --case viscous
    )
endif()

add_test(
    NAME CheckpointRestartIntegration
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/restart_integration_test.py
            --executable $<TARGET_FILE:theseus>
            --source ${PROJECT_SOURCE_DIR}
            --mpiexec ${MPIEXEC_EXECUTABLE}
            --numproc-flag=${MPIEXEC_NUMPROC_FLAG}
            ${AXISYMMETRIC_RESTART_ARGUMENT}
)

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
