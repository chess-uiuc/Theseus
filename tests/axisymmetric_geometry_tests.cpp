// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus testing suites.
//
// SPDX-License-Identifier: BSD-3-Clause
#include "unit_test.hpp"

#include "AxisymmetricGeometry.hpp"
#include "AxisymmetricSource.hpp"
#include "GasModel.hpp"
#include "bc_kernels.hpp"
#include "dgsem_cache_utilities.hpp"

namespace
{
  struct GeometryCache
  {
    bool axisymmetric = true;
    int dim = 2;
    int Np_x = 3;
    int Np_y = 3;
    int Np_z = 1;
    mfem::IntegrationRules rules{0, mfem::Quadrature1D::GaussLobatto};
    const mfem::IntegrationRule *ir_vol = nullptr;
    mfem::Vector elJac;
    mfem::Vector elMetric;
    mfem::Vector elQuadratureWeights;
    mfem::Vector elRadius;

    GeometryCache(int elements)
    {
      ir_vol = &rules.Get(mfem::Geometry::SQUARE, 3);
      const int points = elements * ir_vol->GetNPoints();
      elJac.SetSize(points);
      elMetric.SetSize(points * dim * dim);
      elQuadratureWeights.SetSize(points);
      elRadius.SetSize(points);
    }
  };

  struct ViscousAxisContext
  {
    Theseus::PhysicsConstants physics{1.4, 0.72, 287.0, 1.8e-5};
    Theseus::StateLayout layout{2, 1};
    Theseus::IdealGasModel gas{physics, layout};
    int dim = 2;
    int num_equations = 4;
    int Np_x = 3;
    int Np_y = 3;
    int ndof_scalar_el = 9;
    const mfem::real_t *bc_vector_d = nullptr;
    mfem::real_t differentiation[9] = {
      -1.5,  2.0, -0.5,
      -0.5,  0.0,  0.5,
       0.5, -2.0,  1.5
    };
    const mfem::real_t *D_d = differentiation;
  };

  void CacheRadii(mfem::Mesh &mesh, GeometryCache &cache)
  {
    for (int element = 0; element < mesh.GetNE(); ++element)
      {
        Theseus::AssembleElementVolumeGeometricTerms(
          *mesh.GetElementTransformation(element), &cache);
      }
  }
}

TEST(straight_mesh_radius_cache_matches_physical_coordinate)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    2, 2, mfem::Element::QUADRILATERAL, true, 2.0, 1.0);
  GeometryCache cache(mesh.GetNE());
  CacheRadii(mesh, cache);

  const int nq = cache.ir_vol->GetNPoints();
  mfem::Vector physical(2);
  for (int element = 0; element < mesh.GetNE(); ++element)
    {
      auto *transformation = mesh.GetElementTransformation(element);
      for (int point = 0; point < nq; ++point)
        {
          transformation->Transform(cache.ir_vol->IntPoint(point), physical);
          EXPECT_CLOSE(cache.elRadius[element*nq + point], physical(1), 1.0e-14);
        }
    }
  return 0;
}

TEST(curved_mesh_radius_cache_matches_physical_coordinate)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    2, 2, mfem::Element::QUADRILATERAL, true, 2.0, 1.0);
  mesh.SetCurvature(3);
  mfem::VectorFunctionCoefficient deformation(2, [](const mfem::Vector &x,
                                                     mfem::Vector &y) {
    y.SetSize(2);
    y(0) = x(0);
    y(1) = x(1) + 0.1*x(1)*(1.0 - x(1))*x(0)*(2.0 - x(0));
  });
  mesh.GetNodes()->ProjectCoefficient(deformation);

  GeometryCache cache(mesh.GetNE());
  CacheRadii(mesh, cache);

  const int nq = cache.ir_vol->GetNPoints();
  mfem::Vector physical(2);
  for (int element = 0; element < mesh.GetNE(); ++element)
    {
      auto *transformation = mesh.GetElementTransformation(element);
      for (int point = 0; point < nq; ++point)
        {
          transformation->Transform(cache.ir_vol->IntPoint(point), physical);
          EXPECT_CLOSE(cache.elRadius[element*nq + point], physical(1), 1.0e-13);
          EXPECT_TRUE(cache.elRadius[element*nq + point] >= 0.0);
        }
    }
  return 0;
}

TEST(radius_accessor_is_device_safe)
{
  mfem::Vector coordinates(4);
  coordinates[0] = 7.0;
  coordinates[1] = 0.0;
  coordinates[2] = 3.0;
  coordinates[3] = 2.5;
  mfem::Vector radii(2);
  const mfem::real_t *coordinates_d = coordinates.Read();
  mfem::real_t *radii_d = radii.Write();
  mfem::forall(2, [=] MFEM_HOST_DEVICE (int point) {
    radii_d[point] = Theseus::AxisymmetricGeometry::Radius(
      coordinates_d + 2*point);
  });
  EXPECT_CLOSE(radii[0], 0.0, 0.0);
  EXPECT_CLOSE(radii[1], 2.5, 0.0);
  return 0;
}

TEST(euler_geometric_source_matches_swirl_free_equations_away_from_axis)
{
  const mfem::real_t gamma = 1.4;
  Theseus::PhysicsConstants physics(gamma, 0.72, 287.0, 1.8e-5);
  Theseus::StateLayout layout(2, 1);
  Theseus::IdealGasModel gas(physics, layout);

  const mfem::real_t density = 2.0;
  const mfem::real_t axial_velocity = 3.0;
  const mfem::real_t radial_velocity = -0.5;
  const mfem::real_t pressure = 4.0;
  const mfem::real_t energy = pressure/(gamma - 1.0) +
    0.5*density*(axial_velocity*axial_velocity +
                 radial_velocity*radial_velocity);
  const mfem::real_t state[4] = {
    density, density*axial_velocity, density*radial_velocity, energy
  };
  mfem::real_t rate[4] = {0.0, 0.0, 0.0, 0.0};
  const mfem::real_t radius = 2.5;

  EXPECT_TRUE(Theseus::AddAxisymmetricEulerSourceAwayFromAxis(
    gas, state, radius, rate));
  EXPECT_CLOSE(rate[0], -density*radial_velocity/radius, 1.0e-14);
  EXPECT_CLOSE(rate[1], -density*axial_velocity*radial_velocity/radius,
               1.0e-14);
  EXPECT_CLOSE(rate[2], -density*radial_velocity*radial_velocity/radius,
               1.0e-14);
  EXPECT_CLOSE(rate[3], -(energy + pressure)*radial_velocity/radius,
               1.0e-14);
  return 0;
}

TEST(euler_geometric_source_defers_axis_limit)
{
  Theseus::PhysicsConstants physics(1.4, 0.72, 287.0, 1.8e-5);
  Theseus::StateLayout layout(2, 1);
  Theseus::IdealGasModel gas(physics, layout);
  const mfem::real_t state[4] = {1.0, 2.0, 0.0, 5.0};
  mfem::real_t rate[4] = {1.0, 2.0, 3.0, 4.0};

  EXPECT_TRUE(!Theseus::AddAxisymmetricEulerSourceAwayFromAxis(
    gas, state, 0.0, rate));
  EXPECT_CLOSE(rate[0], 1.0, 0.0);
  EXPECT_CLOSE(rate[1], 2.0, 0.0);
  EXPECT_CLOSE(rate[2], 3.0, 0.0);
  EXPECT_CLOSE(rate[3], 4.0, 0.0);
  return 0;
}

TEST(axisymmetric_viscous_stress_uses_cylindrical_velocity_divergence)
{
  const mfem::real_t gamma = 1.4;
  const mfem::real_t viscosity = 1.8e-5;
  Theseus::PhysicsConstants physics(gamma, 0.72, 287.0, viscosity);
  Theseus::StateLayout layout(2, 1);
  Theseus::IdealGasModel gas(physics, layout);

  const mfem::real_t density = 2.0;
  const mfem::real_t axial_velocity = 3.0;
  const mfem::real_t radial_velocity = -0.5;
  const mfem::real_t pressure = 4.0;
  const mfem::real_t energy = pressure/(gamma - 1.0) +
    0.5*density*(axial_velocity*axial_velocity +
                 radial_velocity*radial_velocity);
  const mfem::real_t state[4] = {
    density, density*axial_velocity, density*radial_velocity, energy
  };
  const mfem::real_t dprim_axial[4] = {0.1, 0.2, 0.3, 0.4};
  const mfem::real_t dprim_radial[4] = {-0.1, -0.4, 0.5, -0.2};
  const mfem::real_t dprim_unused[4] = {0.0, 0.0, 0.0, 0.0};
  mfem::real_t cartesian[Theseus::MAXEQ][Theseus::MAXDIM];
  mfem::real_t cylindrical[Theseus::MAXEQ][Theseus::MAXDIM];
  mfem::real_t azimuthal_stress = 0.0;
  const mfem::real_t radius = 2.5;

  Theseus::NavierStokesFlux::ComputeViscousFluxKernel(
    gas, state, dprim_axial, dprim_radial, dprim_unused, cartesian);
  Theseus::NavierStokesFlux::ComputeViscousFluxKernel(
    gas, state, dprim_axial, dprim_radial, dprim_unused, cylindrical,
    true, radius, &azimuthal_stress);

  const mfem::real_t radial_rate = radial_velocity/radius;
  const mfem::real_t diagonal_correction =
    -viscosity*physics.mu_bulk*radial_rate;
  EXPECT_CLOSE(cylindrical[layout.eq_mom[0]][0] -
               cartesian[layout.eq_mom[0]][0], diagonal_correction, 1.0e-18);
  EXPECT_CLOSE(cylindrical[layout.eq_mom[1]][1] -
               cartesian[layout.eq_mom[1]][1], diagonal_correction, 1.0e-18);
  EXPECT_CLOSE(cylindrical[layout.eq_mom[0]][1],
               cartesian[layout.eq_mom[0]][1], 1.0e-18);
  const mfem::real_t cylindrical_divergence = 0.2 + 0.5 + radial_rate;
  EXPECT_CLOSE(azimuthal_stress,
               viscosity*(2.0*radial_rate -
                          physics.mu_bulk*cylindrical_divergence),
               1.0e-18);
  return 0;
}

TEST(axisymmetric_viscous_source_matches_swirl_free_equations_away_from_axis)
{
  const mfem::real_t gamma = 1.4;
  Theseus::PhysicsConstants physics(gamma, 0.72, 287.0, 1.8e-5);
  Theseus::StateLayout layout(2, 1);
  Theseus::IdealGasModel gas(physics, layout);
  const mfem::real_t state[4] = {2.0, 6.0, -1.0, 19.25};
  const mfem::real_t dprim_axial[4] = {0.1, 0.2, 0.3, 0.4};
  const mfem::real_t dprim_radial[4] = {-0.1, -0.4, 0.5, -0.2};
  const mfem::real_t dprim_unused[4] = {0.0, 0.0, 0.0, 0.0};
  const mfem::real_t radius = 2.5;
  mfem::real_t flux[Theseus::MAXEQ][Theseus::MAXDIM];
  mfem::real_t azimuthal_stress = 0.0;
  Theseus::NavierStokesFlux::ComputeViscousFluxKernel(
    gas, state, dprim_axial, dprim_radial, dprim_unused, flux,
    true, radius, &azimuthal_stress);
  mfem::real_t source[4] = {0.0, 0.0, 0.0, 0.0};

  EXPECT_TRUE(Theseus::AddAxisymmetricViscousSourceAwayFromAxis(
    gas, state, dprim_axial, dprim_radial, dprim_unused, radius, source));
  EXPECT_CLOSE(source[layout.eq_mass], 0.0, 0.0);
  EXPECT_CLOSE(source[layout.eq_mom[0]],
               flux[layout.eq_mom[0]][1]/radius, 1.0e-18);
  EXPECT_CLOSE(source[layout.eq_mom[1]],
               (flux[layout.eq_mom[1]][1] - azimuthal_stress)/radius,
               1.0e-18);
  EXPECT_CLOSE(source[layout.eq_energy], flux[layout.eq_energy][1]/radius,
               1.0e-18);

  mfem::real_t axis_source[4] = {1.0, 2.0, 3.0, 4.0};
  EXPECT_TRUE(!Theseus::AddAxisymmetricViscousSourceAwayFromAxis(
    gas, state, dprim_axial, dprim_radial, dprim_unused, 0.0,
    axis_source));
  EXPECT_CLOSE(axis_source[0], 1.0, 0.0);
  EXPECT_CLOSE(axis_source[1], 2.0, 0.0);
  EXPECT_CLOSE(axis_source[2], 3.0, 0.0);
  EXPECT_CLOSE(axis_source[3], 4.0, 0.0);
  return 0;
}

TEST(axisymmetric_viscous_stress_has_finite_axis_limit)
{
  Theseus::PhysicsConstants physics(1.4, 0.72, 287.0, 1.8e-5);
  Theseus::StateLayout layout(2, 1);
  Theseus::IdealGasModel gas(physics, layout);
  const mfem::real_t state[4] = {2.0, 6.0, 0.0, 19.0};
  const mfem::real_t dprim_axial[4] = {0.0, 0.2, 0.0, 0.0};
  const mfem::real_t dprim_radial[4] = {0.0, 0.0, 0.5, 0.0};
  const mfem::real_t dprim_unused[4] = {0.0, 0.0, 0.0, 0.0};
  mfem::real_t flux[Theseus::MAXEQ][Theseus::MAXDIM];
  mfem::real_t azimuthal_stress = 0.0;

  Theseus::NavierStokesFlux::ComputeViscousFluxKernel(
    gas, state, dprim_axial, dprim_radial, dprim_unused, flux,
    true, 0.0, &azimuthal_stress);

  const mfem::real_t divergence = 0.2 + 2.0*0.5;
  const mfem::real_t radial_stress = physics.mu *
    (2.0*0.5 - physics.mu_bulk*divergence);
  EXPECT_CLOSE(flux[layout.eq_mom[0]][0],
               physics.mu*(2.0*0.2 - physics.mu_bulk*divergence),
               1.0e-18);
  EXPECT_CLOSE(flux[layout.eq_mom[1]][1], radial_stress, 1.0e-18);
  EXPECT_CLOSE(azimuthal_stress, radial_stress, 1.0e-18);
  return 0;
}

TEST(axisymmetric_viscous_source_axis_limit_uses_radial_flux_derivative)
{
  ViscousAxisContext ctx;
  mfem::real_t state[36] = {0.0};
  mfem::real_t grad_axial[36] = {0.0};
  mfem::real_t grad_radial[36] = {0.0};
  mfem::real_t grad_unused[36] = {0.0};
  mfem::real_t radius[9];
  mfem::real_t jacobian[9];
  mfem::real_t metric[36] = {0.0};
  const mfem::real_t axial_curvature = 0.7;
  const mfem::real_t pressure = 4.0;
  for (int j = 0; j < 3; ++j)
    {
      for (int i = 0; i < 3; ++i)
        {
          const int point = j*3 + i;
          const mfem::real_t r = 0.5*j;
          const mfem::real_t axial_velocity = axial_curvature*r*r;
          radius[point] = r;
          jacobian[point] = 0.25;
          metric[4*point] = 0.5;
          metric[4*point + 3] = 0.5;
          state[point] = 1.0;
          state[point + 9] = axial_velocity;
          state[point + 18] = 0.0;
          state[point + 27] = pressure/(ctx.physics.gamma - 1.0) +
            0.5*axial_velocity*axial_velocity;
          grad_radial[point + 9] = 2.0*axial_curvature*r;
        }
    }

  const mfem::real_t derivative = Theseus::RadialViscousFluxDerivative(
    ctx, state, grad_axial, grad_radial, grad_unused, radius, jacobian,
    metric, 0, ctx.layout.eq_mom[0]);
  EXPECT_CLOSE(derivative, 2.0*ctx.physics.mu*axial_curvature, 1.0e-18);
  return 0;
}

TEST(axis_boundary_gradient_flux_enforces_radial_odd_parity)
{
  ViscousAxisContext ctx;
  Theseus::BCDescriptor axis{};
  axis.type = int(Theseus::BCType::Axis);
  const mfem::real_t entropy_state[4] = {1.0, 2.0, -3.0, 4.0};
  mfem::real_t gradient_flux[4];
  Theseus::BC::ComputeBdrFaceGradFlux(
    ctx, axis, entropy_state, gradient_flux);
  EXPECT_CLOSE(gradient_flux[0], 0.0, 0.0);
  EXPECT_CLOSE(gradient_flux[1], 0.0, 0.0);
  EXPECT_CLOSE(gradient_flux[2], 6.0, 0.0);
  EXPECT_CLOSE(gradient_flux[3], 0.0, 0.0);
  return 0;
}

TEST(euler_geometric_source_axis_limit_matches_parity)
{
  const mfem::real_t gamma = 1.4;
  Theseus::PhysicsConstants physics(gamma, 0.72, 287.0, 1.8e-5);
  Theseus::StateLayout layout(2, 1);
  Theseus::IdealGasModel gas(physics, layout);
  const mfem::real_t density = 2.0;
  const mfem::real_t axial_velocity = 3.0;
  const mfem::real_t pressure = 4.0;
  const mfem::real_t energy = pressure/(gamma - 1.0) +
    0.5*density*axial_velocity*axial_velocity;
  const mfem::real_t state[4] = {
    density, density*axial_velocity, 0.0, energy
  };
  mfem::real_t rate[4] = {0.0, 0.0, 0.0, 0.0};
  const mfem::real_t radial_momentum_derivative = 1.25;

  Theseus::AddAxisymmetricEulerSourceAtAxis(
    gas, state, radial_momentum_derivative, rate);
  EXPECT_CLOSE(rate[0], -radial_momentum_derivative, 1.0e-14);
  EXPECT_CLOSE(rate[1], -axial_velocity*radial_momentum_derivative,
               1.0e-14);
  EXPECT_CLOSE(rate[2], 0.0, 0.0);
  EXPECT_CLOSE(rate[3],
               -(energy + pressure)/density*radial_momentum_derivative,
               1.0e-14);
  return 0;
}

TEST(curved_element_axis_derivative_uses_physical_radial_direction)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    1, 1, mfem::Element::QUADRILATERAL, true, 2.0, 1.0);
  mesh.SetCurvature(3);
  mfem::VectorFunctionCoefficient deformation(2, [](const mfem::Vector &x,
                                                     mfem::Vector &y) {
    y.SetSize(2);
    y(0) = x(0) + 0.15*x(1)*(1.0 - x(1));
    y(1) = x(1) + 0.1*x(1)*(1.0 - x(1))*x(0)*(2.0 - x(0));
  });
  mesh.GetNodes()->ProjectCoefficient(deformation);
  GeometryCache geometry(mesh.GetNE());
  CacheRadii(mesh, geometry);

  const mfem::real_t derivative_data[9] = {
    -3.0, 4.0, -1.0,
    -1.0, 0.0, 1.0,
    1.0, -4.0, 3.0
  };
  Theseus::PhysicsConstants physics(1.4, 0.72, 287.0, 1.8e-5);
  Theseus::StateLayout layout(2, 9);
  struct Context
  {
    int Np_x = 3;
    int Np_y = 3;
    int ndof_scalar_el = 9;
    int dim = 2;
    const mfem::real_t *D_d;
    Theseus::IdealGasModel gas;
  } context{3, 3, 9, 2, derivative_data,
            Theseus::IdealGasModel(physics, layout)};

  mfem::real_t state[36] = {};
  const mfem::real_t expected_derivative = 1.75;
  for (int point = 0; point < 9; ++point)
    {
      state[2*9 + point] = expected_derivative*geometry.elRadius[point];
    }
  for (int point = 0; point < 9; ++point)
    {
      if (geometry.elRadius[point] <=
          Theseus::AxisymmetricGeometry::radius_tolerance)
        {
          EXPECT_CLOSE(Theseus::RadialMomentumDerivative(
                         context, state, geometry.elJac.GetData(),
                         geometry.elMetric.GetData(), point),
                       expected_derivative, 1.0e-12);
        }
    }
  return 0;
}

TEST(axis_boundary_reflects_only_radial_momentum)
{
  Theseus::PhysicsConstants physics(1.4, 0.72, 287.0, 1.8e-5);
  Theseus::StateLayout layout(2, 1);
  Theseus::IdealGasModel gas(physics, layout);
  const mfem::real_t interior[4] = {2.0, 6.0, -1.0, 14.0};
  mfem::real_t exterior[4] = {};

  Theseus::BC::ReflectAxisState(gas, interior, exterior);
  EXPECT_CLOSE(exterior[0], interior[0], 0.0);
  EXPECT_CLOSE(exterior[1], interior[1], 0.0);
  EXPECT_CLOSE(exterior[2], -interior[2], 0.0);
  EXPECT_CLOSE(exterior[3], interior[3], 0.0);
  return 0;
}

TEST(axis_boundary_geometry_must_lie_on_zero_radius)
{
  struct BoundaryCache
  {
    int num_face_points = 2;
    mfem::Array<int> bnd_marker_index;
    mfem::Array<Theseus::BCDescriptor> bc_descriptors;
    mfem::Vector bnd_radius;
  } cache;
  cache.bnd_marker_index.SetSize(1);
  cache.bnd_marker_index[0] = 0;
  cache.bc_descriptors.SetSize(1);
  cache.bc_descriptors[0].type = int(Theseus::BCType::Axis);
  cache.bnd_radius.SetSize(2);
  cache.bnd_radius[0] = 0.0;
  cache.bnd_radius[1] = 0.0;

  EXPECT_TRUE(Theseus::AxisBoundaryGeometryIsValid(cache));
  cache.bnd_radius[1] = 0.25;
  EXPECT_TRUE(!Theseus::AxisBoundaryGeometryIsValid(cache));
  cache.bc_descriptors[0].type = int(Theseus::BCType::SlipWall);
  EXPECT_TRUE(Theseus::AxisBoundaryGeometryIsValid(cache));
  cache.bnd_radius.SetSize(0);
  EXPECT_TRUE(Theseus::AxisBoundaryGeometryIsValid(cache));
  return 0;
}

TEST(axisymmetric_measure_integrates_revolved_constant_state)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    2, 2, mfem::Element::QUADRILATERAL, true, 2.0, 1.0);
  GeometryCache geometry(mesh.GetNE());
  CacheRadii(mesh, geometry);

  const mfem::real_t density = 3.0;
  mfem::real_t cartesian_mass = 0.0;
  mfem::real_t axisymmetric_mass = 0.0;
  for (int point = 0; point < geometry.elRadius.Size(); ++point)
    {
      const mfem::real_t quadrature_weight =
        geometry.elQuadratureWeights[point];
      cartesian_mass += density*quadrature_weight*
        Theseus::AxisymmetricGeometry::MeasureMultiplier(
          false, geometry.elRadius[point]);
      axisymmetric_mass += density*quadrature_weight*
        Theseus::AxisymmetricGeometry::MeasureMultiplier(
          true, geometry.elRadius[point]);
    }

  EXPECT_CLOSE(cartesian_mass, 6.0, 1.0e-13);
  EXPECT_CLOSE(axisymmetric_mass,
               6.0*3.141592653589793238462643383279502884,
               1.0e-12);
  return 0;
}

TEST(axis_primitive_gradient_projection_enforces_parity)
{
  Theseus::PhysicsConstants physics(1.4, 0.72, 287.0, 1.8e-5);
  Theseus::StateLayout layout(2, 1);
  Theseus::IdealGasModel gas(physics, layout);
  mfem::real_t axial_gradient[4] = {1.0, 2.0, 3.0, 4.0};
  mfem::real_t radial_gradient[4] = {5.0, 6.0, 7.0, 8.0};

  EXPECT_TRUE(Theseus::ProjectAxisPrimitiveGradientDirection(
		     gas, 0.0, Theseus::AxisymmetricGeometry::axial_coordinate,
		     axial_gradient));
  EXPECT_CLOSE(axial_gradient[layout.eq_mass], 1.0, 0.0);
  EXPECT_CLOSE(axial_gradient[layout.eq_mom[0]], 2.0, 0.0);
  EXPECT_CLOSE(axial_gradient[layout.eq_mom[1]], 0.0, 0.0);
  EXPECT_CLOSE(axial_gradient[layout.eq_energy], 4.0, 0.0);

  EXPECT_TRUE(Theseus::ProjectAxisPrimitiveGradientDirection(
		     gas, 0.0, Theseus::AxisymmetricGeometry::radial_coordinate,
		     radial_gradient));
  EXPECT_CLOSE(radial_gradient[layout.eq_mass], 0.0, 0.0);
  EXPECT_CLOSE(radial_gradient[layout.eq_mom[0]], 0.0, 0.0);
  EXPECT_CLOSE(radial_gradient[layout.eq_mom[1]], 7.0, 0.0);
  EXPECT_CLOSE(radial_gradient[layout.eq_energy], 0.0, 0.0);
  mfem::real_t off_axis[4] = {9.0, 10.0, 11.0, 12.0};
  EXPECT_TRUE(!Theseus::ProjectAxisPrimitiveGradientDirection(
                      gas, 0.1, Theseus::AxisymmetricGeometry::radial_coordinate,
		      off_axis));
  for (int equation = 0; equation < 4; ++equation)
    {
      EXPECT_CLOSE(off_axis[equation], 9.0 + equation, 0.0);
    }
  return 0;
}

TEST(axisymmetric_viscous_energy_flux_uses_zero_radial_axis_velocity)
{
  Theseus::PhysicsConstants physics(1.4, 0.72, 287.0, 1.8e-5);
  Theseus::StateLayout layout(2, 1);
  Theseus::IdealGasModel gas(physics, layout);
  const mfem::real_t density = 2.0;
  const mfem::real_t axial_velocity = 3.0;
  const mfem::real_t radial_velocity = 0.25;
  const mfem::real_t pressure = 4.0;
  const mfem::real_t energy = pressure/(physics.gamma - 1.0) +
    0.5*density*(axial_velocity*axial_velocity +
		 radial_velocity*radial_velocity);
  const mfem::real_t state[4] = {
    density, density*axial_velocity, density*radial_velocity, energy
  };
  const mfem::real_t dprim_axial[4] = {0.0, 0.2, 0.0, 0.0};
  const mfem::real_t dprim_radial[4] = {0.0, 0.0, 0.5, 0.0};
  const mfem::real_t dprim_unused[4] = {0.0, 0.0, 0.0, 0.0};
  mfem::real_t flux[Theseus::MAXEQ][Theseus::MAXDIM];
  Theseus::NavierStokesFlux::ComputeViscousFluxKernel(
	gas, state, dprim_axial, dprim_radial, dprim_unused, flux, true, 0.0);
  EXPECT_CLOSE(flux[layout.eq_mom[0]][1], 0.0, 0.0);
  EXPECT_CLOSE(flux[layout.eq_energy][1], 0.0, 0.0);
  return 0;
}
