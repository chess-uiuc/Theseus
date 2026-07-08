// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cassert>
// StateLayout and State Views for Gas Simulations
//
// Goal:
//   - Centralize the mapping from (equation, dof) -> flat index
//   - Provide indices and offsets for accessing model-specific
//     quantities from the solution data
//   - Provide general field-level view for future use
//
// Layout assumption (canonical, matches current code):
// Note that potentially we will change rho --> rho*Y_alpha.
//   U(eq, dof) is currently stored as equation-blocked:
//       U = [ rho       (block 0)
//           | rho*u_x   (block 1)
//           | rho*u_y   (block 2, if dim > 1)
//           | rho*u_z   (block 3, if dim > 2)
//           | rho*E     (block dim+1)
//           | scalars (maybe/future, blocks dim+2:nspecies+dim+1) ]
//
// Each block has length = num_dofs_scalar (wrt mesh or element)
//
#include "theseus_kernels.hpp"

namespace Theseus
{
  // -----------------------------------------------------------------------------
  // StateLayout: single point to define how state storage is arranged
  // -----------------------------------------------------------------------------
  
  struct StateLayout
  {
    int dim;              // spatial dimension (1,2,3)
    int num_dofs_scalar;  // DOFs per scalar field (block length)

    // Equation indices (0-based)
    int eq_mass;          // mass density
    int eq_mom0;          // first component of momentum
    int eq_mom[3];        // momentum density components: x,y,z
    int eq_energy;        // total energy density

    // Optional scalar support (forward-looking)
    int eq_scalar0;       // index of first scalar component (or -1 if none)
    int num_scalars;      // number of scalar components
    
    MFEM_HOST_DEVICE StateLayout() = default;
    /**
     * Set up after creation.
     *
     * This method exists to allow default construction of StateLayout objects,
     * which is sometimes required for use in containers (e.g., std::vector),
     * serialization frameworks, or APIs that require default-constructible types.
     * In such cases, the object is first default-constructed and then initialized
     * via this setup() method.
     *
     * Prefer using the parameterized constructor whenever possible to ensure
     * objects are fully initialized at construction time. Use setup() only when
     * default construction is unavoidable due to external constraints.
     *
     * @param dim_             Spatial dimension (1, 2, or 3)
     * @param num_dofs_scalar_ Number of DOFs per scalar field
     * @param num_scalars_     Number of scalar components (default: 0)
     */
    void setup(int dim_, int num_dofs_scalar_, int num_scalars_ = 0)
    {
      dim = dim_;
      num_dofs_scalar = num_dofs_scalar_;
      eq_mass = 0;
      eq_energy = dim_ + 1;
      eq_scalar0 = (num_scalars_ > 0 ? (dim_ + 2) : -1);
      num_scalars = num_scalars_;
      // Momentum components follow mass
      eq_mom0 = 1;
      for (int d = 0; d < 3; ++d)
        {
          eq_mom[d] = (d < dim_) ? (1 + d) : -1;
        }
    }

    // Canonical ordering:
    //   [rho, rho*u_0,-, rho*u_(dim-1), rho*E, scalars-]
    StateLayout(int dim_, int num_dofs_scalar_, int num_scalars_ = 0)
    { setup(dim_, num_dofs_scalar_, num_scalars_);}

    // Convenience for nequations
    MFEM_HOST_DEVICE inline int nequations() const
    { return dim + 2 + num_scalars; }

    // parameter validation routine
    MFEM_HOST_DEVICE inline int validate(int equation, int dof) const
    {
      return ((equation < nequations() && dof < num_dofs_scalar &&
               equation > -1 && dof > -1) ? 0 : 1);
    }

    // Flat index into the equation-blocked vector
    MFEM_HOST_DEVICE inline int index(int equation, int dof) const
    {
      assert(validate(equation, dof) == 0);
      return equation * num_dofs_scalar + dof;
    }
  };
  
  // -----------------------------------------------------------------------------
  // PointStateView: per-point view (immediate refactor tool).
  //
  // Usage pattern:
  //
  //   Vector state = [rho, rhoVx, rhoVy, rhoVz, rhoE]
  //   const mfem::real_t *q = state.GetData();
  //   PointStateView S{state.GetData()};
  //   double rho = S.mass(layout);
  //   double rhoU = S.momentum(layout, 0);
  //   double rhoE = S.energy(layout);
  //
  //   double rho = q[layout.eq_mass];  (same thing)
  //
  //
  // This is a small, value-type-like view with correct indexing.
  // -----------------------------------------------------------------------------
  struct PointStateView
  {
    const mfem::real_t* U;           // packed state data -> [rho, rhoVx, rhoVy, ..., rhoE]
    
    MFEM_HOST_DEVICE explicit PointStateView(const mfem::real_t* U_)
      : U(U_)
    { }

    MFEM_HOST_DEVICE inline bool is_valid() const
    {
      return (U != nullptr);
    }
 
    // Mass / density
    MFEM_HOST_DEVICE inline mfem::real_t mass(const StateLayout &L) const
    {
      const mfem::real_t rho = U[ L.eq_mass ];
      return rho;
    }
 
    // Momentum components: d = 0(x),1(y),2(z)
    MFEM_HOST_DEVICE inline mfem::real_t momentum(const StateLayout &L, int d) const
    {
      assert(d < L.dim);
      return U[ L.eq_mom[d] ];
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t momentum_x(const StateLayout &L) const
    {
      return momentum(L, 0);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t momentum_y(const StateLayout &L) const
    {
      return momentum(L, 1);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t momentum_z(const StateLayout &L) const
    {
      return momentum(L, 2);
    }
    
    // Velocity components: d = 0(x),1(y),2(z)
    MFEM_HOST_DEVICE inline mfem::real_t velocity(const StateLayout &L, int d) const
    {
      return momentum(L, d) / mass(L);
    }

    MFEM_HOST_DEVICE inline mfem::real_t velocity_x(const StateLayout &L) const
    {
      return momentum_x(L) / mass(L);
    }

    MFEM_HOST_DEVICE inline mfem::real_t velocity_y(const StateLayout &L) const
    {
      return momentum_y(L) / mass(L);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t velocity_z(const StateLayout &L) const
    {
      return momentum_z(L) / mass(L);
    }
    
    // Total energy
    MFEM_HOST_DEVICE inline mfem::real_t energy(const StateLayout &L) const
    {
      return U[ L.eq_energy ];
    }
    
    // Scalar components, if present
    MFEM_HOST_DEVICE inline mfem::real_t scalar(const StateLayout &L, int k) const
    {
      assert(L.num_scalars > 0);
      assert((k >= 0 && k < L.num_scalars) && "Invalid scalar index");
      return U[ L.eq_scalar0 + k ];
    }
  };

  // -----------------------------------------------------------------------------
  // PointStateViewRW: *writeable* per-point view
  //
  // Usage pattern:
  //
  //   Vector state = [rho, rhoVx, rhoVy, rhoVz, rhoE]
  //   PointStateView S{state.GetData()};
  //   double rho = S.mass(layout);
  //   double rhoU = S.momentum(layout, 0);
  //   double rhoE = S.energy(layout);
  //
  // This is a small, value-type-like view with correct indexing.
  // -----------------------------------------------------------------------------
  struct PointStateViewRW
  {
    mfem::real_t* U;                 // packed state data -> [rho, rhoVx, rhoVy, ..., rhoE]
    
    MFEM_HOST_DEVICE
    PointStateViewRW(mfem::real_t* U_)
      : U(U_)
    { }
    
    MFEM_HOST_DEVICE inline bool is_valid() const
    {
      return (U != nullptr);
    }
    
    // Mass density
    MFEM_HOST_DEVICE inline mfem::real_t mass(const StateLayout &L) const
    {
      const mfem::real_t rho = U[ L.eq_mass ];
      return rho;
    }
    
    // Set Mass density
    MFEM_HOST_DEVICE inline void set_mass(const StateLayout &L, mfem::real_t val)
    {
      U[ L.eq_mass ] = val;
    }

    // Momentum components: d = 0(x),1(y),2(z)
    MFEM_HOST_DEVICE inline mfem::real_t momentum(const StateLayout &L, int d) const
    {
      assert(d < L.dim);
      return U[ L.eq_mom[d] ];
    }
    
    // Set Momentum components: d = 0(x),1(y),2(z)
    MFEM_HOST_DEVICE inline void set_momentum(const StateLayout &L, int d, mfem::real_t val)
    {
      assert(d < L.dim);
      U[ L.eq_mom[d] ] = val;
    }

    MFEM_HOST_DEVICE inline mfem::real_t momentum_x(const StateLayout &L) const
    {
      return momentum(L, 0);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t momentum_y(const StateLayout &L) const
    {
      return momentum(L, 1);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t momentum_z(const StateLayout &L) const
    {
      return momentum(L, 2);
    }
    
    // Velocity components: d = 0(x),1(y),2(z)
    MFEM_HOST_DEVICE inline mfem::real_t velocity(const StateLayout &L, int d) const
    {
      return momentum(L, d) / mass(L);
    }

    MFEM_HOST_DEVICE inline mfem::real_t velocity_x(const StateLayout &L) const
    {
      return momentum_x(L) / mass(L);
    }

    MFEM_HOST_DEVICE inline mfem::real_t velocity_y(const StateLayout &L) const
    {
      return momentum_y(L) / mass(L);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t velocity_z(const StateLayout &L) const
    {
      return momentum_z(L) / mass(L);
    }
    
    // Total energy
    MFEM_HOST_DEVICE inline mfem::real_t energy(const StateLayout &L) const
    {
      return U[ L.eq_energy ];
    }
    
    // Set Total energy
    MFEM_HOST_DEVICE inline void set_energy(const StateLayout &L, mfem::real_t val)
    {
      U[ L.eq_energy ] = val;
    }

    // Scalar components, if present
    MFEM_HOST_DEVICE inline mfem::real_t scalar(const StateLayout &L, int k) const
    {
      assert(L.num_scalars > 0);
      assert((k >= 0 && k < L.num_scalars) && "Invalid scalar index");
      return U[ L.eq_scalar0 + k ];
    }

    // Set Scalar components, if present
    MFEM_HOST_DEVICE inline void set_scalar(const StateLayout &L, int k, mfem::real_t val)
    {
      assert(L.num_scalars > 0);
      assert((k >= 0 && k < L.num_scalars) && "Invalid scalar index");
      U[ L.eq_scalar0 + k ] = val;
    }
    
  };

  // -----------------------------------------------------------------------------
  // PointPrimitiveView: per-point view
  //
  // Usage pattern:
  //
  //   Vector state = [rho, Vx, Vy, Vz, p]
  //   PointStateView S{state.GetData()};
  //   double rho = S.mass(layout);
  //   double Vx  = S.velocity(layout, 0);
  //   double Vy  = S.velocity(layout, 1);
  //   double p   = S.pressure(layout);
  //
  // This is a small, value-type-like view with correct indexing.
  // -----------------------------------------------------------------------------
  struct PointPrimitiveView
  {
    const mfem::real_t* U;           // packed state data -> [rho, rhoVx, rhoVy, ..., rhoE]
    
    MFEM_HOST_DEVICE
    PointPrimitiveView(const mfem::real_t* U_)
      : U(U_)
    { }
    
    MFEM_HOST_DEVICE inline bool is_valid() const
    {
      return (U != nullptr);
    }

    // Mass / density
    MFEM_HOST_DEVICE inline mfem::real_t mass(const StateLayout &L) const
    {
      const mfem::real_t rho = U[ L.eq_mass ];
      return rho;
    }
    
    // Array offset to velocity components
    MFEM_HOST_DEVICE inline int velocity_loc(const StateLayout &L) const
    {
      return L.eq_mom0;
    }
    // Momentum components: d = 0(x),1(y),2(z)
    MFEM_HOST_DEVICE inline mfem::real_t velocity(const StateLayout &L, int d) const
    {
      assert(d < L.dim);
      return U[ L.eq_mom[d] ];
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t velocity_x(const StateLayout &L) const
    {
      return velocity(L, 0);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t velocity_y(const StateLayout &L) const
    {
      return velocity(L, 1);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t velocity_z(const StateLayout &L) const
    {
      return velocity(L, 2);
    }

    MFEM_HOST_DEVICE inline mfem::real_t pressure(const StateLayout &L) const
    {
      return U[ L.eq_energy ];
    }
    
    // Scalar components, if present
    MFEM_HOST_DEVICE inline mfem::real_t scalar(const StateLayout &L, int k) const
    {
      assert(L.num_scalars > 0);
      assert((k >= 0 && k < L.num_scalars) && "Invalid scalar index");
      return U[ L.eq_scalar0 + k ];
    }
  };

  // -----------------------------------------------------------------------------
  // PointPrimtiveViewRW: *writeable* per-point view
  //
  // Usage pattern:
  //
  //   Vector state = [rho, rhoVx, rhoVy, rhoVz, rhoE]
  //   PointStateView S{state.GetData()};
  //   double r =  S.mass(layout);
  //   double U = S.velocity(layout, 0);
  //   double V = S.velocity(layout, 1);
  //   double p = S.pressure(layout);
  //   S.set_pressure(layout, p);
  //
  // This is a small, value-type-like view with correct indexing.
  // -----------------------------------------------------------------------------
  struct PointPrimitiveViewRW
  {
    mfem::real_t* U;                 // packed state data -> [rho, rhoVx, rhoVy, ..., rhoE]
    
    MFEM_HOST_DEVICE
    PointPrimitiveViewRW(mfem::real_t* U_) : U(U_) { }

    MFEM_HOST_DEVICE inline bool is_valid() const
    {
      return (U != nullptr);
    }

    // Mass density
    MFEM_HOST_DEVICE inline mfem::real_t mass(const StateLayout &L) const
    {
      const mfem::real_t rho = U[ L.eq_mass ];
      return rho;
    }
    
    // Set Mass density
    MFEM_HOST_DEVICE inline void set_mass(const StateLayout &L, mfem::real_t val)
    {
      U[ L.eq_mass ] = val;
    }

    // Array offset to velocity components
    MFEM_HOST_DEVICE inline int velocity_loc(const StateLayout &L) const
    {
      return L.eq_mom0;
    }

    // Momentum components: d = 0(x),1(y),2(z)
    MFEM_HOST_DEVICE inline mfem::real_t velocity(const StateLayout &L, int d) const
    {
      assert(d < L.dim);
      return U[ L.eq_mom[d] ];
    }
    
    // Set Momentum components: d = 0(x),1(y),2(z)
    MFEM_HOST_DEVICE inline void set_velocity(const StateLayout &L, int d, mfem::real_t val)
    {
      assert(d < L.dim);
      U[ L.eq_mom[d] ] = val;
    }

    MFEM_HOST_DEVICE inline mfem::real_t velocity_x(const StateLayout &L) const
    {
      return velocity(L, 0);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t velocity_y(const StateLayout &L) const
    {
      return velocity(L, 1);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t velocity_z(const StateLayout &L) const
    {
      return velocity(L, 2);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t pressure(const StateLayout &L) const
    {
      return U[ L.eq_energy ];
    }
    
    MFEM_HOST_DEVICE inline void set_pressure(const StateLayout &L, mfem::real_t val)
    {
      U[ L.eq_energy ] = val;
    }

    // Scalar components, if present
    MFEM_HOST_DEVICE inline mfem::real_t scalar(const StateLayout &L, int k) const
    {
      assert(L.num_scalars > 0);
      assert((k >= 0 && k < L.num_scalars) && "Invalid scalar index");
      return U[ L.eq_scalar0 + k ];
    }

    // Set Scalar components, if present
    MFEM_HOST_DEVICE inline void set_scalar(const StateLayout &L, int k, mfem::real_t val)
    {
      assert(L.num_scalars > 0);
      assert((k >= 0 && k < L.num_scalars) && "Invalid scalar index");
      U[ L.eq_scalar0 + k ] = val;
    }
    
  };

  // -----------------------------------------------------------------------------
  // DofStateView: per-DOF view (Get conserved quantities for particular DOF)
  //
  // Usage pattern:
  //
  //   const double* U = sol->Read();
  //   StateLayout layout(dim, num_dofs_scalar);
  //   for (int i = 0; i < num_dofs_scalar; ++i) {
  //       DofStateView<const double> S{U, i};
  //       double rho  = S.mass(l);
  //       double rhoU = S.momentum(l, 0);
  //       double E    = S.energy(l);
  //   }
  //
  // This is a small, value-type-like view with correct indexing.
  // -----------------------------------------------------------------------------
  struct DofStateView
  {
    const mfem::real_t* U;           // pointer into equation-blocked storage
    int dof;                   // which DOF (0 .. num_dofs_scalar-1)
    
    MFEM_HOST_DEVICE
    DofStateView(const mfem::real_t* U_,
                 int dof_)
      : U(U_), dof(dof_)
    { }
    
    MFEM_HOST_DEVICE inline bool is_valid() const
    {
      return (U != nullptr);
    }

    // Mass / density
    MFEM_HOST_DEVICE inline mfem::real_t mass(const StateLayout &L) const
    {
      const mfem::real_t rho = U[ L.index(L.eq_mass, dof) ];
      return rho;
    }
    
    // Momentum components: d = 0(x),1(y),2(z)
    MFEM_HOST_DEVICE inline mfem::real_t momentum(const StateLayout &L, int d) const
    {
      assert(d < L.dim);
      return U[ L.index(L.eq_mom[d], dof) ];
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t momentum_x(const StateLayout &L) const
    {
      return momentum(L, 0);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t momentum_y(const StateLayout &L) const
    {
      return momentum(L, 1);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t momentum_z(const StateLayout &L) const
    {
      return momentum(L, 2);
    }
    
    // Velocity components: d = 0(x),1(y),2(z)
    MFEM_HOST_DEVICE inline mfem::real_t velocity(const StateLayout &L, int d) const
    {
      return momentum(L, d) / mass(L);
    }

    MFEM_HOST_DEVICE inline mfem::real_t velocity_x(const StateLayout &L) const
    {
      return velocity(L, 0);
    }

    MFEM_HOST_DEVICE inline mfem::real_t velocity_y(const StateLayout &L) const
    {
      return velocity(L, 1);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t velocity_z(const StateLayout &L) const
    {
      return velocity(L, 2);
    }
    
    // Total energy
    MFEM_HOST_DEVICE inline mfem::real_t energy(const StateLayout &L) const
    {
      return U[ L.index(L.eq_energy, dof) ];
    }
    
    // Scalar components, if present
    MFEM_HOST_DEVICE inline mfem::real_t scalar(const StateLayout &L, int k) const
    {
      assert(L.num_scalars > 0);
      assert(k >= 0 && k < L.num_scalars && "Invalid scalar index");
      return U[ L.index(L.eq_scalar0 + k, dof) ];
    }
  };

  // -----------------------------------------------------------------------------
  // FieldStateView: Field-level view.
  //
  // Wraps a mfem::real_t* + StateLayout and provides equation-blocked access over all DOFs.
  // For use in loops that already use (eq, i) patterns, or when needing more general
  // mechanism than DofStateView.
  // -----------------------------------------------------------------------------
  struct FieldStateView
  {
    mfem::real_t* data;                // raw pointer to equation-blocked storage
    
    MFEM_HOST_DEVICE
    FieldStateView(mfem::real_t* data_) : data(data_) { }

    MFEM_HOST_DEVICE inline bool is_valid() const
    {
      return (data != nullptr);
    }
    
    // Generic access by (equation, dof)
    MFEM_HOST_DEVICE inline mfem::real_t& u(const StateLayout &layout, int equation, int dof) const
    {
      return data[ layout.index(equation, dof) ];
    }
    
    // Named accessors for convenience
    MFEM_HOST_DEVICE inline mfem::real_t& mass(const StateLayout &layout, int dof) const
    {
      return u(layout, layout.eq_mass, dof);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t& momentum(const StateLayout &layout, int component, int dof) const
    {
      return u(layout, layout.eq_mom[component], dof);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t& momentum_x(const StateLayout &layout, int dof) const
    {
      return u(layout, layout.eq_mom[0], dof);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t& momentum_y(const StateLayout &layout, int dof) const
    {
      return u(layout, layout.eq_mom[1], dof);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t& momentum_z(const StateLayout &layout, int dof) const
    {
      return u(layout, layout.eq_mom[2], dof);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t velocity(const StateLayout &layout, int component, int dof) const
    {
      return momentum(layout, component, dof) / mass(layout, dof);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t velocity_x(const StateLayout &layout, int dof) const
    {
      return velocity(layout, 0, dof);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t velocity_y(const StateLayout &layout, int dof) const
    {
      return velocity(layout, 1, dof);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t velocity_z(const StateLayout &layout, int dof) const
    {
      return velocity(layout, 2, dof);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t& energy(const StateLayout &layout, int dof) const
    {
      return u(layout, layout.eq_energy, dof);
    }
    
    MFEM_HOST_DEVICE inline mfem::real_t& scalar(const StateLayout &layout, int k, int dof) const
    {
      assert(layout.num_scalars > 0);
      assert(k >= 0 && k < layout.num_scalars && "Invalid scalar index");
      return u(layout, layout.eq_scalar0 + k, dof);
    }
  };

  inline int offset_mass   (const Theseus::StateLayout &L) { return L.index(L.eq_mass, 0); }
  inline int offset_momentum  (const Theseus::StateLayout &L) { return L.index(L.eq_mom0, 0); }
  inline int offset_energy (const Theseus::StateLayout &L) { return L.index(L.eq_energy, 0); }
  inline int offset_scalars (const Theseus::StateLayout &L) { return L.index(L.eq_scalar0, 0); };

} // namespace Theseus
