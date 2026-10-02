// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "VisualizationFields.hpp"

#include <iostream>
#include <stdexcept>

using namespace Theseus;

namespace
{
  struct CountingGas : IdealGasModel
  {
    using IdealGasModel::IdealGasModel;
    mutable int evaluations = 0;
    mutable GasPropertyRequest last_request;

    template<typename StateView>
    GasProperties EvaluateProperties(const StateView &state,
                                     const GasPropertyRequest &request) const
    {
      ++evaluations;
      last_request = request;
      return IdealGasModel::EvaluateProperties(state, request);
    }
  };

  void Require(bool condition, const std::string &message)
  {
    if (!condition)
      {
        throw std::runtime_error(message);
      }
  }

  void CheckValue(mfem::DataCollection &collection, const char *field, int point,
                  mfem::real_t expected)
  {
    const auto *values = collection.GetField(field);
    Require(values != nullptr, std::string("Missing field: ") + field);
    const auto actual = (*values)[point];
    Require(std::isfinite(actual) && std::abs(actual - expected) <= 1e-10,
            std::string("Incorrect field: ") + field);
  }

  void CheckFields(int dimension)
  {
    auto mesh = dimension == 1 ? mfem::Mesh::MakeCartesian1D(2) :
                dimension == 2 ? mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL) :
                                 mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::HEXAHEDRON);
    mfem::ParMesh parallel_mesh(MPI_COMM_WORLD, mesh);
    mfem::DG_FECollection collection(2, dimension, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace scalar_space(&parallel_mesh, &collection);
    mfem::ParFiniteElementSpace vector_space(&parallel_mesh, &collection, dimension);
    const int points = scalar_space.GetNDofs();
    const StateLayout layout(dimension, points);
    auto gas = std::make_shared<CountingGas>(PhysicsConstants(1.4, 0.72, 287, 0.02), layout);
    GasModelInterfaceT<CountingGas> interface(gas);
    mfem::Vector solution(points * layout.nequations());
    mfem::ParGridFunction density(&scalar_space);
    density.MakeRef(&scalar_space, solution, 0);
    mfem::ParGridFunction blending(&scalar_space);
    blending = 0.25;

    const auto runtime = nlohmann::json::parse(R"({
      "visualization": {
        "fields": ["density", "velocity", "pressure", "blending_coefficient", "temperature",
                   "sound_speed", "mach_number", "specific_internal_energy",
                   "internal_energy_density", "specific_total_energy", "viscosity",
                   "thermal_conductivity"]
      }
    })");
    const auto config = VisualizationConfig::FromRuntime(runtime, true);
    VisualizationFields fields(config, scalar_space, vector_space, density, &blending);
    mfem::ParaViewDataCollection paraview("test-paraview", &parallel_mesh);
    mfem::VisItDataCollection visit("test-visit", &parallel_mesh);
    fields.Register(paraview);
    fields.Register(visit);
    Require(paraview.GetFieldMap().size() == 12, "Incorrect ParaView field count");
    Require(visit.GetFieldMap().size() == 12, "Incorrect VisIt field count");
    Require(paraview.GetField("Density") == &density, "Density should alias the solution");
    Require(paraview.GetField("Blending Coeff") == &blending, "Blending should alias its field");

    const auto *temperature_field = paraview.GetField("Temperature");
    for (int update = 0; update < 2; ++update)
      {
        for (int point = 0; point < points; ++point)
          {
            PhysicalStateInput input;
            input.thermo = DensityTemperature{1 + 0.01 * point, 300.0 + point + update};
            input.velocity.assign(dimension, -4);
            input.velocity[0] = 3;
            mfem::Vector conservative;
            gas->ConservativeFromPhysical(input, conservative);
            for (int equation = 0; equation < layout.nequations(); ++equation)
              {
                solution[equation * points + point] = conservative[equation];
              }
          }

        gas->evaluations = 0;
        fields.Update(solution, interface);
        Require(gas->evaluations == points, "Expected one combined property request per point");
        Require(gas->last_request.pressure && gas->last_request.temperature &&
                gas->last_request.sound_speed && gas->last_request.viscosity &&
                gas->last_request.thermal_conductivity, "Missing combined property dependency");
        Require(paraview.GetField("Temperature") == temperature_field, "Field address changed");

        for (int point = 0; point < points; ++point)
          {
            const DofStateView state{solution.HostRead(), point};
            const auto rho = gas->density(state);
            const auto internal_energy = gas->specific_internal_energy(state);
            const auto speed = std::sqrt(9.0 + 16.0 * (dimension - 1));
            for (auto *output : {static_cast<mfem::DataCollection *>(&paraview),
                                 static_cast<mfem::DataCollection *>(&visit)})
              {
                CheckValue(*output, "Density", point, rho);
                for (int component = 0; component < dimension; ++component)
                  {
                    const auto velocity = component == 0 ? 3.0 : -4.0;
                    CheckValue(*output, "Velocity", point + component * points, velocity);
                  }
                CheckValue(*output, "Blending Coeff", point, 0.25);
                CheckValue(*output, "Pressure", point, gas->pressure(state));
                CheckValue(*output, "Temperature", point, 300.0 + point + update);
                CheckValue(*output, "Sound Speed", point, gas->sound_speed(state));
                CheckValue(*output, "Mach Number", point, speed / gas->sound_speed(state));
                CheckValue(*output, "Specific Internal Energy", point, internal_energy);
                CheckValue(*output, "Internal Energy Density", point, rho * internal_energy);
                CheckValue(*output, "Specific Total Energy", point, state.energy(layout) / rho);
                CheckValue(*output, "Viscosity", point, gas->viscosity(state));
                CheckValue(*output, "Thermal Conductivity", point, gas->thermal_conductivity(state));
              }
          }
      }

    for (const auto &selection : {"density", "velocity", "specific_internal_energy"})
      {
        nlohmann::json selected;
        selected["visualization"]["fields"] = {selection};
        const auto selection_config = VisualizationConfig::FromRuntime(selected, false);
        VisualizationFields selected_fields(selection_config, scalar_space, vector_space,
                                             density, nullptr);
        gas->evaluations = 0;
        selected_fields.Update(solution, interface);
        Require(gas->evaluations == 0, "Algebraic output should not evaluate EOS properties");
        mfem::DataCollection output("selected", &parallel_mesh);
        selected_fields.Register(output);
        Require(output.GetFieldMap().size() == 1, "Unexpected allocated/registered field");
      }
  }
}

int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  try
    {
      for (int dimension = 1; dimension <= 3; ++dimension)
        {
          CheckFields(dimension);
        }
      std::cout << "Visualization registration, values, dependencies and lifetime: PASS\n";
    }
  catch (const std::exception &error)
    {
      std::cerr << error.what() << '\n';
      return 1;
    }
  return 0;
}
