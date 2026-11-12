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

add_test(NAME NumericalFluxRegressionCheckerTests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/numflux_regression_checker_tests.py)

add_test(NAME TestRunnerTests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_runner_tests.py)

add_test(
    NAME RunTheseusHarnessTests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_theseus_harness_tests.py
)

add_test(
    NAME CompareVizTests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/compare_viz_tests.py
)

add_test(
    NAME PerformanceTimerParserTests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/performance_timers_tests.py
)

add_test(
    NAME ComparePerformanceTests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/compare_performance_tests.py
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

add_executable(viscous_volume_tests viscous_volume_tests.cpp)
target_include_directories(viscous_volume_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(viscous_volume_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
if(ENABLE_CUDA)
  set_source_files_properties(viscous_volume_tests.cpp PROPERTIES LANGUAGE CUDA)
  target_include_directories(viscous_volume_tests PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  target_compile_options(viscous_volume_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
endif()
add_test(NAME ViscousVolumeTests COMMAND viscous_volume_tests)

# Debugging guard contract: invalid metadata, NaN, and even tiny nonzero mass
# flux must terminate with an explanation, including in NDEBUG builds.
add_executable(viscous_invariant_probe viscous_invariant_probe.cpp)
target_include_directories(viscous_invariant_probe PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(viscous_invariant_probe PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
target_compile_definitions(viscous_invariant_probe PRIVATE NDEBUG)
add_test(NAME ViscousInvariantTests
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/viscous_invariant_tests.py
          $<TARGET_FILE:viscous_invariant_probe>)

# Analytic face-assembly oracle covering both face execution modes,
# LTE/CPG gradient conventions, output bounds, and axis regularization.
foreach(transport_variant IN ITEMS standard sutherland)
  set(normal_face_target normal_viscous_face_tests_${transport_variant})
  add_executable(${normal_face_target} normal_viscous_face_tests.cpp)
  target_include_directories(${normal_face_target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(${normal_face_target} PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
  if(transport_variant STREQUAL "sutherland")
    target_compile_definitions(${normal_face_target} PRIVATE SUTHERLAND)
  endif()
  if(ENABLE_CUDA)
    set_source_files_properties(normal_viscous_face_tests.cpp PROPERTIES LANGUAGE CUDA)
    target_include_directories(${normal_face_target} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_compile_options(${normal_face_target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
    add_test(NAME NormalViscousFaces_${transport_variant}_CUDA
      COMMAND ${normal_face_target} cuda)
  endif()
  add_test(NAME NormalViscousFaces_${transport_variant}_CPU
    COMMAND ${normal_face_target} cpu)
endforeach()

add_test(NAME LTETGVComparisonContractTests
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/lte_tgv_checker_tests.py)

# Bicubic Helmholtz polynomial/derivatives, independent analytic references.
add_executable(helmholtz_interpolation_tests helmholtz_interpolation_tests.cpp)
target_include_directories(helmholtz_interpolation_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(helmholtz_interpolation_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
if(ENABLE_CUDA)
  set_source_files_properties(helmholtz_interpolation_tests.cpp PROPERTIES LANGUAGE CUDA)
  target_include_directories(helmholtz_interpolation_tests PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  target_compile_options(helmholtz_interpolation_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
  add_test(NAME HelmholtzInterpolation_CUDA COMMAND helmholtz_interpolation_tests cuda)
endif()
add_test(NAME HelmholtzInterpolation_CPU COMMAND helmholtz_interpolation_tests cpu)

add_executable(helmholtz_eos_tests helmholtz_eos_tests.cpp)
target_include_directories(helmholtz_eos_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(helmholtz_eos_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
if(ENABLE_CUDA)
  set_source_files_properties(helmholtz_eos_tests.cpp PROPERTIES LANGUAGE CUDA)
  target_include_directories(helmholtz_eos_tests PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  target_compile_options(helmholtz_eos_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
  add_test(NAME HelmholtzEOS_CUDA COMMAND helmholtz_eos_tests cuda)
endif()
add_test(NAME HelmholtzEOS_CPU COMMAND helmholtz_eos_tests cpu)

foreach(variant IN ITEMS standard sutherland)
  set(target helmholtz_gas_tests_${variant})
  add_executable(${target} helmholtz_gas_tests.cpp)
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/generated)
  target_link_libraries(${target} PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
  if(variant STREQUAL "sutherland")
    target_compile_definitions(${target} PRIVATE SUTHERLAND)
  endif()
  if(THESEUS_WITH_PLATO AND THESEUS_PLATO_HAS_ENTROPY)
    target_compile_definitions(${target} PRIVATE USE_PLATO THESEUS_PLATO_HAS_ENTROPY)
    target_link_libraries(${target} PRIVATE Plato::Plato ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
    target_link_directories(${target} PRIVATE ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
  endif()
  if(ENABLE_CUDA)
    set_source_files_properties(helmholtz_gas_tests.cpp PROPERTIES LANGUAGE CUDA)
    target_include_directories(${target} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
    add_test(NAME HelmholtzGas_${variant}_CUDA COMMAND ${target} cuda)
  endif()
  add_test(NAME HelmholtzGas_${variant}_CPU COMMAND ${target} cpu)
endforeach()
if(THESEUS_WITH_PLATO AND THESEUS_PLATO_HAS_ENTROPY)
  add_test(NAME HelmholtzGas_AIR5 COMMAND helmholtz_gas_tests_standard plato)
endif()

foreach(variant IN ITEMS standard sutherland)
  set(target lteec_flux_tests_${variant})
  add_executable(${target} lteec_flux_tests.cpp)
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(${target} PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
  if(variant STREQUAL "sutherland")
    target_compile_definitions(${target} PRIVATE SUTHERLAND)
  endif()
  if(ENABLE_CUDA)
    set_source_files_properties(lteec_flux_tests.cpp PROPERTIES LANGUAGE CUDA)
    target_include_directories(${target} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
    add_test(NAME LTEECFlux_${variant}_CUDA COMMAND ${target} cuda)
  endif()
  add_test(NAME LTEECFlux_${variant}_CPU COMMAND ${target} cpu)
endforeach()

# This factory fixture creates a 3D Cartesian periodic mesh/configuration.
# Kernel/EOS tests above remain registered in both geometry builds.
if(NOT AXISYMMETRIC)
  set(_helmholtz_available 0)
  set(_plato_available 0)
  if(THESEUS_WITH_PLATO)
    set(_plato_available 1)
    if(THESEUS_PLATO_HAS_ENTROPY)
      set(_helmholtz_available 1)
    endif()
  endif()
  add_test(NAME HelmholtzFactorySelection
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/helmholtz_selection_test.py
      --source ${PROJECT_SOURCE_DIR} --executable $<TARGET_FILE:theseus>
      --available ${_helmholtz_available} --plato ${_plato_available})
endif()

# Optional large-batch numerical/reference checks. No historical source required.
if(THESEUS_BUILD_KERNEL_STRESS)
  add_test(NAME KernelStressReference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/helmholtz_stress_reference_tests.py)
  set_tests_properties(KernelStressReference PROPERTIES LABELS "kernel-stress;reference")

  foreach(kind IN ITEMS reference channel)
    set(target helmholtz_mean_stress_tests_${kind})
    add_executable(${target} helmholtz_mean_stress_tests.cpp)
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${target} PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
    if(kind STREQUAL "channel")
      target_compile_definitions(${target} PRIVATE THESEUS_STRESS_CHANNEL_CASES)
    endif()
    if(ENABLE_CUDA)
      set_source_files_properties(helmholtz_mean_stress_tests.cpp PROPERTIES LANGUAGE CUDA)
      target_include_directories(${target} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
      add_test(NAME HelmholtzMeanStress_${kind}_CUDA COMMAND ${target} cuda full)
      set_tests_properties(HelmholtzMeanStress_${kind}_CUDA PROPERTIES
        LABELS "kernel-stress;cuda" TIMEOUT 1800)
    endif()
    add_test(NAME HelmholtzMeanStress_${kind}_CPU COMMAND ${target} cpu full)
    set_tests_properties(HelmholtzMeanStress_${kind}_CPU PROPERTIES
      LABELS "kernel-stress;cpu" TIMEOUT 1800)
  endforeach()

  foreach(variant IN ITEMS standard sutherland)
    set(target lteec_flux_stress_tests_${variant})
    add_executable(${target} lteec_flux_stress_tests.cpp)
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${target} PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
    if(variant STREQUAL "sutherland")
      target_compile_definitions(${target} PRIVATE SUTHERLAND)
    endif()
    if(ENABLE_CUDA)
      set_source_files_properties(lteec_flux_stress_tests.cpp PROPERTIES LANGUAGE CUDA)
      target_include_directories(${target} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
      add_test(NAME LTEECFlux_stress_${variant}_CUDA COMMAND ${target} cuda full)
      set_tests_properties(LTEECFlux_stress_${variant}_CUDA PROPERTIES
        LABELS "kernel-stress;cuda" TIMEOUT 1800)
    endif()
    add_test(NAME LTEECFlux_stress_${variant}_CPU COMMAND ${target} cpu full)
    set_tests_properties(LTEECFlux_stress_${variant}_CPU PROPERTIES
      LABELS "kernel-stress;cpu" TIMEOUT 1800)
  endforeach()
endif()

# Channel initialization and regression-checker contracts are cheap default checks.
add_executable(channel_flow_case_tests channel_flow_case_tests.cpp)
target_include_directories(channel_flow_case_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(channel_flow_case_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
add_test(NAME ChannelFlowInitialCondition_CPU COMMAND channel_flow_case_tests)
add_test(NAME ChannelFlowCheckerTests
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/channel_flow_checker_tests.py)

add_executable(research_channel_source_tests research_channel_source_tests.cpp)
target_include_directories(research_channel_source_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(research_channel_source_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
if(ENABLE_CUDA)
  set_source_files_properties(research_channel_source_tests.cpp PROPERTIES LANGUAGE CUDA)
  target_include_directories(research_channel_source_tests PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  target_compile_options(research_channel_source_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
  add_test(NAME ResearchChannelSource_CUDA COMMAND research_channel_source_tests cuda)
endif()
add_test(NAME ResearchChannelSource_CPU COMMAND research_channel_source_tests cpu)
add_test(NAME ResearchChannelSource_MPI_CPU COMMAND ${MPIEXEC_EXECUTABLE}
  ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:research_channel_source_tests> ${MPIEXEC_POSTFLAGS} cpu)
add_executable(research_channel_compatibility_tests research_channel_compatibility_tests.cpp)
target_include_directories(research_channel_compatibility_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(research_channel_compatibility_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
target_compile_definitions(research_channel_compatibility_tests PRIVATE SUTHERLAND CHANNEL_RESEARCH_COMPATIBILITY)
if(ENABLE_CUDA)
  set_source_files_properties(research_channel_compatibility_tests.cpp PROPERTIES LANGUAGE CUDA)
  target_include_directories(research_channel_compatibility_tests PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  target_compile_options(research_channel_compatibility_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
  add_test(NAME ResearchChannelCompatibility_CUDA COMMAND research_channel_compatibility_tests cuda)
endif()
add_test(NAME ResearchChannelCompatibility_CPU COMMAND research_channel_compatibility_tests cpu)
