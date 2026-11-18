// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "mfem.hpp"
#include "GasModel.hpp"
#include "LTEGasModel.hpp"
#include "LocalCharacteristicBoundary.hpp"
#include "BoundaryStateConfig.hpp"
#include "bc_kernels.hpp"
#include "LaxFriedrichsFlux.hpp"
#include <iostream>
#include <stdexcept>

using namespace Theseus;

void Require(bool value, const char *message)
{
  if (!value)
    {
      throw std::runtime_error(message);
    }
}

void Close(double actual, double expected, double scale, const char *message)
{
  Require(std::isfinite(actual) && std::abs(actual - expected) <= 2e-9 * scale, message);
}

template<class Gas>
struct Context
{
  Gas gas;
  LaxFriedrichsFlux::InviscidFlux iflux;
  int dim;
  int num_equations;
  bool axisymmetric = false;
  const mfem::real_t *bc_vector_d;
  const mfem::real_t *bc_scalar_d = nullptr;
};

template<class Gas>
void CheckFamilies(const Gas &gas, double R, double offset, double entropy_normalization, bool tabulated_sound = false, double gamma = 1.4)
{
  const int dim = gas.L.dim;
  const int neq = gas.L.nequations();
  const int stride = 4 * neq + 1;
  const double rho = 1.5;
  const double pressure = rho * R * 600;
  const double sound = std::sqrt(gamma * R * 600);
  for (const auto family : {BCType::CharacteristicReference, BCType::CharacteristicPressure, BCType::CharacteristicTotal})
    {
      for (const auto sign : {-1, 1})
        {
          for (const auto qi : {-80.0, 120.0})
            {
              mfem::Vector input(neq), data(neq + 2 + dim), normal(dim), result(stride);
              const auto p_a = family == BCType::CharacteristicTotal ? 1.1 * pressure : 0.98 * pressure;
              const auto T_a = family == BCType::CharacteristicTotal ? 700.0 : 500.0;
              const auto rho_a = p_a / (R * T_a);
              const auto qa = family == BCType::CharacteristicTotal ? 0.0 : -60.0;
              input = 0.0;
              data = 0.0;
              normal = 0.0;
              input[0] = rho;
              input[neq - 1] = rho * (offset + R * 600 / (gamma - 1) + qi * qi / 2);
              data[0] = rho_a;
              data[neq - 1] = rho_a * (offset + R * T_a / (gamma - 1) + qa * qa / 2);
              for (int d = 0; d < dim; ++d)
                {
                  const double n = dim == 1 ? sign : (dim == 2 ? (d == 0 ? 0.6 * sign : 0.8 * sign)
                      : (d == 0 ? 0.48 * sign : (d == 1 ? 0.64 * sign : 0.6 * sign)));
                  normal[d] = 3 * n;
                  input[d + 1] = rho * qi * n;
                  data[d + 1] = rho_a * qa * n;
                  data[neq + 2 + d] = -n;
                }
              data[neq] = p_a;
              data[neq + 1] = T_a;
              input.UseDevice();
              data.UseDevice();
              normal.UseDevice();
              result.UseDevice();
              const auto *u = input.Read();
              const auto *a = data.Read();
              const auto *n = normal.Read();
              auto *out = result.Write();
              Context<Gas> dc{gas, {}, dim, neq, false, a};
              BCDescriptor bc{};
              bc.type = int(family);
              mfem::forall(1, [=] MFEM_HOST_DEVICE (int)
              {
                out[4 * neq] = BC::LocalCharacteristicState(gas, family, u, n, a, out);
                mfem::real_t entropy[MAXEQ];
                PointStateViewRW entropy_view{entropy};
                gas.entropy_state(PointStateView{u}, entropy_view);
                BC::ComputeBdrFaceGradFlux(dc, bc, entropy, out + neq, u, n);
                BC::ApplyBoundaryConditionInviscid(dc, bc, u, n, out + 2 * neq);
                mfem::real_t zero[MAXEQ] = {};
                BC::ApplyViscousBoundaryCondition(dc, bc, u, zero, zero, zero, n, 0, out + 3 * neq);
              });
              const auto *actual = result.HostRead();
              auto p_b = p_a;
              auto q_b = qi - (p_b - pressure) / (rho * sound);
              double rho_b, T_b;
              if (family == BCType::CharacteristicTotal && qi < 0)
                {
                  // Independently reduced aligned-CPG limit of HEGEL's quadratic:
                  // selected c_b=c_i and V_b=-q_i. No production root duplicated.
                  const auto factor = 1 + (gamma - 1) * qi * qi / 2 / (sound * sound);
                  p_b = p_a * std::pow(factor, -gamma / (gamma - 1));
                  T_b = T_a / factor;
                  rho_b = p_b / (R * T_b);
                  q_b = qi;
                }
              else
                {
                  if (family == BCType::CharacteristicReference && qi <= 0)
                    {
                      p_b = (p_a + pressure + rho * sound * (qi - qa)) / 2;
                      q_b = qi - (p_b - pressure) / (rho * sound);
                    }
                  const bool outgoing = q_b >= 0;
                  if (outgoing || family == BCType::CharacteristicReference)
                    {
                      rho_b = (outgoing ? rho : rho_a)
                              + (p_b - (outgoing ? pressure : p_a)) / (sound * sound);
                      T_b = p_b / (R * rho_b);
                    }
                  else
                    {
                      T_b = T_a;
                      rho_b = p_b / (R * T_b);
                    }
                }
              Close(actual[0], rho_b, rho, "density disagrees with HEGEL/local equations");
              Close(actual[neq - 1], rho_b * (offset + R * T_b / (gamma - 1) + q_b * q_b / 2),
                    rho * R * 600, "EOS conserved energy");
              Require(bool(actual[4 * neq]) == (q_b >= 0), "material direction");
              for (int d = 0; d < dim; ++d)
                {
                  Close(actual[d + 1] / actual[0], q_b * normal.HostRead()[d] / 3, sound,
                        "normal velocity/orientation");
                  double gradient_velocity = q_b;
                  if (family == BCType::CharacteristicPressure)
                    {
                      gradient_velocity = qi;
                    }
                  const auto expected = q_b >= 0 ? 0 : entropy_normalization * normal.HostRead()[d] / 3
                      * (gradient_velocity / T_b - qi / 600);
                  Close(actual[neq + d + 1], expected, entropy_normalization * sound / 500,
                        "entropy momentum lifting");
                }
              Close(actual[neq], 0, 1, "mass entropy component unconstrained");
              const auto expected_energy_lift = q_b >= 0 ? 0 : entropy_normalization * (1.0 / 600 - 1.0 / T_b);
              Close(actual[2 * neq - 1], expected_energy_lift, entropy_normalization / 500,
                    "entropy thermal lifting");
              const auto lambda = 3 * std::max(std::abs(qi) + sound,
                  std::abs(q_b) + std::sqrt(gamma * R * T_b));
              const auto mass_flux = 1.5 * (rho * qi + rho_b * q_b) - lambda * (rho_b - rho) / 2;
              // Manufactured LTE linearly interpolates c; use the actual represented
              // c for LLF in that case rather than an analytic sqrt(T) comparison.
              const auto lower_t = 200 + 50 * std::floor((T_b - 200) / 50);
              const auto weight = (T_b - lower_t) / 50;
              const auto c_b_table = tabulated_sound
                  ? (1 - weight) * std::sqrt(gamma * R * lower_t)
                    + weight * std::sqrt(gamma * R * (lower_t + 50))
                  : std::sqrt(gamma * R * T_b);
              const auto table_lambda = 3 * std::max(std::abs(qi) + sound, std::abs(q_b) + c_b_table);
              const auto represented_mass = mass_flux + (lambda - table_lambda) * (rho_b - rho) / 2;
              Close(actual[2 * neq], represented_mass, rho * sound, "signed LLF mass flux");
              if (family != BCType::CharacteristicTotal)
                {
                  Require(q_b >= 0 ? actual[2 * neq] > 0 : actual[2 * neq] < 0, "mixed mass direction");
                }
              for (int q = 0; q < neq; ++q)
                {
                  Close(actual[3 * neq + q], actual[2 * neq + q],
                        std::max(1.0, std::abs(actual[2 * neq + q])), "zero-gradient viscous flux");
                }
              std::cout << "state," << dim << ',' << int(family) << ',' << R << ',' << gamma
                        << ',' << sign << ',' << qi << ',' << q_b << ',' << actual[0] << ',' << rho_b
                        << ',' << actual[neq - 1] << ','
                        << rho_b * (offset + R * T_b / (gamma - 1) + q_b * q_b / 2)
                        << ',' << actual[2 * neq] << ',' << represented_mass << '\n';
            }
        }
    }
}

template<class Gas>
void CheckShearHeat(const Gas &gas, double R, double conductivity, bool lte)
{
  mfem::Vector result(8);
  result.UseDevice();
  auto *out = result.Write();
  mfem::forall(1, [=] MFEM_HOST_DEVICE (int)
  {
    const auto thermo = gas.boundary_p_T(1.5 * R * 400, 400);
    mfem::real_t state[] = {1.5, 120, 0, 1.5 * (thermo.specific_internal_energy + 3200)};
    mfem::real_t normal[] = {-2, 0};
    mfem::real_t gx[] = {0, 0, 3, lte ? 4 : 1.5 * R * 4};
    mfem::real_t zero[MAXEQ] = {};
    Context<Gas> dc{gas, {}, 2, 4, false, state};
    BCDescriptor bc{};
    bc.type = int(BCType::CharacteristicReference);
    BC::ApplyBoundaryConditionInviscid(dc, bc, state, normal, out);
    BC::ApplyViscousBoundaryCondition(dc, bc, state, gx, zero, zero, normal, 0, out + 4);
  });
  const auto *v = result.HostRead();
  Close(v[4] - v[0], 0, 1, "viscous mass flux");
  Close(v[5] - v[1], 0, 1, "viscous normal stress");
  Close(v[6] - v[2], 2 * 0.02 * 3, 1, "independent shear flux");
  Close(v[7] - v[3], 2 * conductivity * 4, conductivity * 8, "independent heat flux");
}

void CheckPressureDrivenBackflowTrace()
{
  IdealGasModel gas{PhysicsConstants(1.4, 0.72, 287, 0.02), StateLayout(2, 7)};
  mfem::Vector result(3 * 4);
  result.UseDevice();
  auto *out = result.Write();
  mfem::forall(3, [=] MFEM_HOST_DEVICE (int family)
  {
    const mfem::real_t rho = 1.5;
    const mfem::real_t pressure = rho * 287 * 600;
    mfem::real_t input[] = {rho, rho * 10, rho * 7,
                            rho * (287 * 600 / 0.4 + (10 * 10 + 7 * 7) / 2)};
    mfem::real_t reference[] = {1.2 * pressure / (287 * 500), 0, 0,
                                1.2 * pressure / 0.4, 1.2 * pressure, 500, -1, 0};
    mfem::real_t normal[] = {3, 0};
    Context<IdealGasModel> dc{gas, {}, 2, 4, false, reference};
    BCDescriptor bc{};
    bc.type = int(BCType::CharacteristicReference) + family;
    mfem::real_t entropy[4];
    PointStateViewRW entropy_view{entropy};
    gas.entropy_state(PointStateView{input}, entropy_view);
    mfem::real_t correction[4];
    BC::ComputeBdrFaceGradFlux(dc, bc, entropy, correction, input, normal);
    out[family * 4] = correction[1] + 10 * correction[3];
    out[family * 4 + 1] = correction[3];
    mfem::real_t target[4];
    out[family * 4 + 2] = BC::LocalCharacteristicState(gas, static_cast<BCType>(bc.type),
                                                     input, normal, reference, target);
    out[family * 4 + 3] = target[1] / target[0];
  });
  const auto *v = result.HostRead();
  for (int family = 0; family < 3; ++family)
    {
      Close(v[family * 4], 0, 1, "pressure-driven backflow natural normal velocity");
      Require(v[family * 4 + 1] != 0, "pressure-driven incoming T missing");
      Require(v[family * 4 + 2] == 0 && v[family * 4 + 3] < 0, "pressure-driven reversal not selected");
    }
}

void CheckVariableCaloricLTE()
{
  LTETable::Data data;
  LTETable::LTETables tables(9, 17);
  data.rho_grid.SetSize(9);
  data.T_grid.SetSize(17);
  data.e_grid.SetSize(17);
  data.lte_table.SetSize(9 * 9 * 17);
  data.inv_table.SetSize(9 * 17);
  for (int i = 0; i < 9; ++i)
    {
      data.rho_grid[i] = 0.25 + 0.5 * i;
    }
  for (int j = 0; j < 17; ++j)
    {
      const auto t = 200 + 50.0 * j;
      const auto cv = 400 + 0.2 * t;
      const auto gamma = 1 + 200 / cv;
      data.T_grid[j] = t;
      data.e_grid[j] = 10000 + 400 * t + 0.1 * t * t;
      for (int i = 0; i < 9; ++i)
        {
          const double values[] = {data.rho_grid[i] * 200 * t, data.e_grid[j], cv, cv + 200,
                                   200, gamma, std::sqrt(gamma * 200 * t), 0.02, 20};
          for (int k = 0; k < 9; ++k)
            {
              data.lte_table[tables.L.property_index(k, i, j)] = values[k];
            }
          data.inv_table[tables.L.property_index(0, i, j)] = t;
        }
    }
  data.lte_table.UseDevice();
  data.inv_table.UseDevice();
  data.rho_grid.UseDevice();
  data.T_grid.UseDevice();
  data.e_grid.UseDevice();
  tables.tables = {data.lte_table.Read(), data.inv_table.Read(), data.rho_grid.Read(),
                   data.T_grid.Read(), data.e_grid.Read()};
  LTEGas gas{PhysicsConstants(1.4, 0.72, 200, 0.02), StateLayout(2, 7), tables};
  mfem::Vector result(4);
  result.UseDevice();
  auto *out = result.Write();
  mfem::forall(1, [=] MFEM_HOST_DEVICE (int)
  {
    mfem::real_t input[] = {1.5, 120, 0, 1.5 * (286000 + 3200)};
    mfem::real_t reference[] = {0.75, 0, 0, 0.75 * 394000, 120000, 800, 1, 0};
    mfem::real_t normal[] = {-3, 0};
    BC::LocalCharacteristicState(gas, BCType::CharacteristicTotal, input, normal, reference, out);
  });
  const auto gamma = 1 + 200.0 / 520;
  const auto factor = 1 + (gamma - 1) * 80 * 80 / (2 * gamma * 200 * 600);
  const auto temperature = 800 / factor;
  const auto pressure = 120000 * std::pow(factor, -gamma / (gamma - 1));
  const auto rho = pressure / (200 * temperature);
  const auto lower = 200 + 50 * std::floor((temperature - 200) / 50);
  const auto weight = (temperature - lower) / 50;
  const auto energy_low = 10000 + 400 * lower + 0.1 * lower * lower;
  const auto energy_high = 10000 + 400 * (lower + 50) + 0.1 * (lower + 50) * (lower + 50);
  const auto energy = (1 - weight) * energy_low + weight * energy_high;
  const auto *v = result.HostRead();
  Close(v[0], rho, rho, "local gamma from calorically imperfect LTE table");
  Close(v[1] / v[0], 80, 400, "frozen total inlet speed");
  Close(v[3], rho * (energy + 3200), rho * energy, "calorically imperfect actual EOS energy");
  const auto h0 = 394000 + 200 * 800;
  std::cout << "frozen_total_enthalpy_relative_error,"
            << (energy + 200 * temperature + 3200 - h0) / h0 << '\n';
}

void CheckLTE()
{
  LTETable::Data data;
  LTETable::LTETables tables(9, 17);
  data.rho_grid.SetSize(9);
  data.T_grid.SetSize(17);
  data.e_grid.SetSize(17);
  data.lte_table.SetSize(9 * 9 * 17);
  data.inv_table.SetSize(9 * 17);
  for (int i = 0; i < 9; ++i)
    {
      data.rho_grid[i] = 0.25 + i * 0.5;
    }
  for (int j = 0; j < 17; ++j)
    {
      const double t = 200 + 50 * j;
      data.T_grid[j] = t;
      data.e_grid[j] = 10000 + 500 * t;
      for (int i = 0; i < 9; ++i)
        {
          const auto rho = data.rho_grid[i];
          const double properties[] = {rho * 200 * t, 10000 + 500 * t, 500, 700, 200, 1.4,
                                       std::sqrt(1.4 * 200 * t), 0.02, 20};
          for (int k = 0; k < 9; ++k)
            {
              data.lte_table[tables.L.property_index(k, i, j)] = properties[k];
            }
          data.inv_table[tables.L.property_index(0, i, j)] = t;
        }
    }
  data.lte_table.UseDevice();
  data.inv_table.UseDevice();
  data.rho_grid.UseDevice();
  data.T_grid.UseDevice();
  data.e_grid.UseDevice();
  tables.tables = {data.lte_table.Read(), data.inv_table.Read(), data.rho_grid.Read(),
                   data.T_grid.Read(), data.e_grid.Read()};
  LTEGas gas{PhysicsConstants(1.4, 0.72, 200, 0.02), StateLayout(2, 7), tables};
  CheckFamilies(gas, 200, 10000, 1, true);
  CheckShearHeat(gas, 200, 20, true);
  // Inversions at non-node coordinates must agree with independent p=rho*R*T.
  mfem::Vector result(6);
  result.UseDevice();
  auto *out = result.Write();
  mfem::forall(1, [=] MFEM_HOST_DEVICE (int)
  {
    const auto a = gas.boundary_p_T(153200, 613);
    const auto b = gas.boundary_rho_p(1.23, 157194);
    out[0] = a.density;
    out[1] = a.specific_internal_energy;
    out[2] = b.temperature;
    out[3] = b.specific_internal_energy;
    out[4] = gas.boundary_p_T(1e20, 613).density;
    out[5] = gas.boundary_rho_p(1.23, 1e20).temperature;
  });
  const auto *v = result.HostRead();
  Close(v[0], 153200.0 / (200 * 613), 1, "LTE pT inversion");
  Close(v[1], 10000 + 500 * 613, 400000, "LTE energy offset");
  Close(v[2], 639, 639, "LTE rhoP inversion");
  Close(v[3], 10000 + 500 * 639, 400000, "LTE rhoP energy");
  Require(std::isnan(v[4]) && std::isnan(v[5]), "outside-table roots must propagate NaN");
}

// Host configuration must reject an unsupported backend before touching tables.
// Keep existing Helmholtz kernel instantiations independent of old LTE layout.
void CheckBackendAndParsing()
{
  auto cpg = std::make_shared<IdealGasModel>(
      PhysicsConstants(1.4, 0.72, 287, 0.02), StateLayout(2, 7));
  GasModelInterfaceT<IdealGasModel> host(cpg);
  const nlohmann::json total = {{"type", "characteristic-total-inlet"},
      {"total_pressure", 100000}, {"total_temperature", 400}, {"direction", {0, -3}}};
  const auto packed = ParseLocalCharacteristicBoundary(total, host);
  Require(packed.values.Size() == 8, "total payload size");
  Close(packed.values[6], 0, 1, "normalized axial direction");
  Close(packed.values[7], -1, 1, "normalized radial direction");
  for (const auto *type : {"characteristic-inlet", "characteristic-pressure-outlet"})
    {
      const nlohmann::json config = {{"type", type}, {"state", {
          {"pressure", 100000}, {"temperature", 400}, {"velocity", {20, 0}}}}};
      const auto reference = ParseLocalCharacteristicBoundary(config, host);
      Close(reference.values[0], 100000.0 / (287 * 400), 1, "reference EOS packing");
    }
  auto helmholtz = std::make_shared<LTEHelmholtzGas>();
  GasModelInterfaceT<LTEHelmholtzGas> unsupported(helmholtz);
  for (const auto *type : {"characteristic-inlet", "characteristic-radial-inlet",
                          "characteristic-pressure-outlet", "characteristic-total-inlet"})
    {
      bool rejected = false;
      try
        {
          ParseLocalCharacteristicBoundary({{"type", type}}, unsupported);
        }
      catch (const std::invalid_argument &error)
        {
          rejected = std::string(error.what()).find("LTE_Helmholtz") != std::string::npos;
        }
      Require(rejected, "Helmholtz local BC must be rejected during host configuration");
    }
  Require(!BC::LocalCharacteristicState(*helmholtz, BCType::CharacteristicReference,
      nullptr, nullptr, nullptr, nullptr), "unsupported backend path must not access old tables");
}

int main(int argc, char **argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(backend.c_str());
  device.Print();
  try
    {
      if (backend == "cuda")
        {
          Require(mfem::Device::Allows(mfem::Backend::CUDA), "CUDA fallback");
        }
      std::cout.precision(17);
      for (int dim = 1; dim <= 3; ++dim)
        {
          for (const auto gamma : {1.4, 5.0 / 3})
            {
              IdealGasModel gas{PhysicsConstants(gamma, 0.72, 287, 0.02), StateLayout(dim, 7)};
              CheckFamilies(gas, 287, 0, 1.0 / 287, false, gamma);
            }
        }
      CheckBackendAndParsing();
      CheckLTE();
      CheckVariableCaloricLTE();
      CheckPressureDrivenBackflowTrace();
      IdealGasModel viscous_gas{PhysicsConstants(1.4, 0.72, 287, 0.02), StateLayout(2, 7)};
      CheckShearHeat(viscous_gas, 287, 0.02 * 3.5 * 287 / 0.72, false);
    }
  catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      return 1;
    }
  return 0;
}
