// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "ViscousInvariants.hpp"
#include <cstring>
#include <limits>

int main(int argc, char **argv)
{
  if (argc != 2)
    {
      return 2;
    }
  // No table evaluation is required to test metadata or mass-flux rejection.
  mfem::real_t storage[36] = {};
  Theseus::LTETable::LTETables tables(2, 2);
  tables.tables = {storage, storage, storage, storage, storage};
  Theseus::StateLayout layout(3, 1);
  Theseus::PhysicsConstants constants{};
  Theseus::LTEGas gas(constants, layout, tables);
  mfem::real_t flux[Theseus::MAXEQ][Theseus::MAXDIM] = {};
  const char *mode = argv[1];
  if (std::strcmp(mode, "bad_dimension") == 0)
    {
      gas.L.dim = Theseus::MAXDIM + 1;
    }
  else if (std::strcmp(mode, "bad_mass_index") == 0)
    {
      gas.L.eq_mass = -1;
    }
  else if (std::strcmp(mode, "bad_energy_index") == 0)
    {
      gas.L.eq_energy = Theseus::MAXEQ;
    }
  else if (std::strcmp(mode, "bad_momentum_index") == 0)
    {
      gas.L.eq_mom[2] = Theseus::MAXEQ;
    }
  else if (std::strcmp(mode, "excess_equations") == 0)
    {
      gas.L.num_scalars = 1;
    }
  else if (std::strcmp(mode, "bad_table_shape") == 0)
    {
      gas.T.L.ny = 0;
    }
  else if (std::strcmp(mode, "table_offset_overflow") == 0)
    {
      gas.T.L.nx = INT_MAX;
    }
  else if (std::strcmp(mode, "bad_table_property") == 0)
    {
      gas.T.L.mu_idx = 9;
    }
  else if (std::strcmp(mode, "null_table") == 0)
    {
      gas.T.tables.inv_table = nullptr;
    }
  else if (std::strcmp(mode, "tiny_mass") == 0)
    {
      flux[0][2] = std::numeric_limits<mfem::real_t>::epsilon();
    }
  else if (std::strcmp(mode, "nan_mass") == 0)
    {
      flux[0][0] = std::numeric_limits<mfem::real_t>::quiet_NaN();
    }
  else if (std::strcmp(mode, "bad_projection") == 0)
    {
      Theseus::ViscousInvariants::Projection(gas, 0.25, "probe_projection");
      return 0;
    }
  else if (std::strcmp(mode, "valid") != 0)
    {
      return 2;
    }
  Theseus::ViscousInvariants::Layout(gas, "probe_layout");
  Theseus::ViscousInvariants::Tables(gas, "probe_tables");
  Theseus::ViscousInvariants::Mass(gas, flux, "probe_mass");
  Theseus::ViscousInvariants::Projection(gas, 0.0, "probe_projection");
  return 0;
}
