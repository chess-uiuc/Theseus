// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus testing suites.
//
// SPDX-License-Identifier: BSD-3-Clause
#include "unit_test.hpp"

#include "AxisymmetricGeometry.hpp"
#include "AxisymmetricSource.hpp"
#include "GasModel.hpp"
#include "StabilityEstimate.hpp"
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

  struct SegmentGeometryCache
  {
    bool axisymmetric = false;
    int dim = 1;
    int Np_x = 4;
    int Np_y = 1;
    int Np_z = 1;
    mfem::IntegrationRules rules{0, mfem::Quadrature1D::GaussLobatto};
    const mfem::IntegrationRule *ir_vol = nullptr;
    mfem::Vector elJac;
    mfem::Vector elMetric;
    mfem::Vector elQuadratureWeights;
    mfem::Vector elRadius;

    explicit SegmentGeometryCache(int elements)
    {
      const int integration_order = 2*Np_x - 3;
      ir_vol = &rules.Get(mfem::Geometry::SEGMENT, integration_order);
      const int points = elements*ir_vol->GetNPoints();
      elJac.SetSize(points);
      elMetric.SetSize(points);
      elQuadratureWeights.SetSize(points);
    }
  };

  struct SubcellMetricCache
  {
    int dim = 3;
    int p = 3;
    int Np_x = 4;
    int Np_y = 4;
    int Np_z = 4;
    int ndof_scalar_el = 64;
    int num_elements = 0;
    mfem::IntegrationRules rules{0, mfem::Quadrature1D::GaussLobatto};
    const mfem::IntegrationRule *ir = nullptr;
    const mfem::IntegrationRule *ir_vol = nullptr;
    mfem::Vector D;
    mfem::Vector elMetric;
    mfem::Vector subcellMetricXi;
    mfem::Vector subcellMetricEta;
    mfem::Vector subcellMetricZeta;
    mfem::Vector subcellWeights;
  };

  void PrepareSubcellMetricCache(mfem::Mesh &mesh, SubcellMetricCache &cache)
  {
    cache.num_elements = mesh.GetNE();
    const int integration_order = 2*cache.Np_x - 3;
    cache.ir = &cache.rules.Get(mfem::Geometry::SEGMENT, integration_order);
    cache.ir_vol = &cache.rules.Get(mfem::Geometry::CUBE, integration_order);
    cache.elMetric.SetSize(cache.num_elements*cache.ndof_scalar_el
                           *cache.dim*cache.dim);

    mfem::Vector unused_jac(cache.num_elements*cache.ndof_scalar_el);
    mfem::Vector unused_weights(cache.num_elements*cache.ndof_scalar_el);
    mfem::Vector unused_radius;
    GeometryCache geometry_cache(0);
    geometry_cache.dim = cache.dim;
    geometry_cache.Np_x = cache.Np_x;
    geometry_cache.Np_y = cache.Np_y;
    geometry_cache.Np_z = cache.Np_z;
    geometry_cache.ir_vol = cache.ir_vol;
    geometry_cache.elJac.MakeRef(unused_jac, 0, unused_jac.Size());
    geometry_cache.elMetric.MakeRef(cache.elMetric, 0, cache.elMetric.Size());
    geometry_cache.elQuadratureWeights.MakeRef(unused_weights, 0,
                                                unused_weights.Size());
    geometry_cache.elRadius.MakeRef(unused_radius, 0, 0);
    geometry_cache.axisymmetric = false;
    for (int element = 0; element < mesh.GetNE(); ++element)
      Theseus::AssembleElementVolumeGeometricTerms(
        *mesh.GetElementTransformation(element), &geometry_cache);

    mfem::Vector barycentric(cache.Np_x);
    barycentric = 1.0;
    for (int i = 1; i < cache.Np_x; ++i)
      for (int j = 0; j < i; ++j)
        {
          barycentric(j) *= cache.ir->IntPoint(j).x - cache.ir->IntPoint(i).x;
          barycentric(i) *= cache.ir->IntPoint(i).x - cache.ir->IntPoint(j).x;
        }
    barycentric.Reciprocal();
    mfem::DenseMatrix derivative(cache.Np_x);
    derivative = 0.0;
    for (int row = 0; row < cache.Np_x; ++row)
      for (int column = 0; column < cache.Np_x; ++column)
        if (row != column)
          {
            derivative(column, row) = barycentric(row)/barycentric(column)
              /(cache.ir->IntPoint(column).x-cache.ir->IntPoint(row).x);
            derivative(column, column) -= derivative(column, row);
          }
    derivative.Transpose();
    cache.D.SetSize(cache.Np_x*cache.Np_x);
    std::memcpy(cache.D.HostWrite(), derivative.Data(),
                sizeof(mfem::real_t)*cache.D.Size());
  }

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

TEST(affine_segment_geometry_gives_expected_cfl_and_mesh_scaling)
{
  constexpr int order = 3;
  constexpr mfem::real_t velocity = 0.75;
  constexpr mfem::real_t sound_speed = 1.25;
  constexpr mfem::real_t target_cfl = 0.4;

  auto rate_for_length = [&](const mfem::real_t length) {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(1, length);
    SegmentGeometryCache cache(mesh.GetNE());
    Theseus::AssembleElementVolumeGeometricTerms(
      *mesh.GetElementTransformation(0), &cache);

    // MFEM places its Gauss-Lobatto rule on Geometry::SEGMENT == [0,1].
    EXPECT_CLOSE(cache.ir_vol->IntPoint(0).x, 0.0, 1.0e-14);
    EXPECT_CLOSE(cache.ir_vol->IntPoint(cache.ir_vol->GetNPoints()-1).x,
                 1.0, 1.0e-14);

    const mfem::real_t expected_rate =
      Theseus::ReferenceAdvectionSpectralScale(order)
      * (std::abs(velocity) + sound_speed) / length;
    for (int point = 0; point < cache.ir_vol->GetNPoints(); ++point)
      {
        // For x = x_left + length*xi with xi in [0,1], J=length and
        // adj(J)=1.  These are the exact arrays consumed by EstimateStability.
        EXPECT_CLOSE(cache.elJac[point], length, 1.0e-14);
        EXPECT_CLOSE(cache.elMetric[point], 1.0, 1.0e-14);

        const mfem::real_t computed_rate =
          Theseus::ReferenceAdvectionSpectralScale(order)
          * Theseus::MappedDirectionalAcousticRate(
              velocity*cache.elMetric[point],
              cache.elMetric[point]*cache.elMetric[point], sound_speed,
              1.0/cache.elJac[point]);
        EXPECT_CLOSE(computed_rate, expected_rate, 1.0e-13);
      }

    Theseus::StabilityEstimate estimate{expected_rate, 0.0, 0.0};
    const mfem::real_t timestep = target_cfl/estimate.TotalRate();
    EXPECT_CLOSE(estimate.CFL(timestep), target_cfl, 1.0e-14);
    return expected_rate;
  };

  const mfem::real_t rate_h = rate_for_length(2.0);
  const mfem::real_t rate_2h = rate_for_length(4.0);
  EXPECT_CLOSE(rate_2h, 0.5*rate_h, 1.0e-13);
  return 0;
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

TEST(subcell_metrics_match_legacy_construction_on_curved_mesh)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
    2, 1, 1, mfem::Element::HEXAHEDRON, 2.0, 1.0, 1.0);
  mesh.SetCurvature(3);
  mfem::VectorFunctionCoefficient deformation(3, [](const mfem::Vector &x,
                                                     mfem::Vector &y) {
    y.SetSize(3);
    y(0) = x(0) + 0.04*x(1)*x(2);
    y(1) = x(1) + 0.06*x(0)*(2.0 - x(0))*x(2);
    y(2) = x(2) + 0.05*x(0)*x(1)*(1.0 - x(2));
  });
  mesh.GetNodes()->ProjectCoefficient(deformation);

  mfem::DG_FECollection collection(3, 3, mfem::BasisType::GaussLobatto);
  mfem::FiniteElementSpace fes(&mesh, &collection);
  SubcellMetricCache cache;
  PrepareSubcellMetricCache(mesh, cache);
  Theseus::ComputeSubcellMetrics(&fes, &cache);

  const mfem::real_t *D = cache.D.HostRead();
  const mfem::real_t *xi = cache.subcellMetricXi.HostRead();
  const mfem::real_t *eta = cache.subcellMetricEta.HostRead();
  const mfem::real_t *zeta = cache.subcellMetricZeta.HostRead();
  const int n = cache.Np_x;
  const int nq = n*n*n;
  const int n_metric_xi = (n + 1)*n*n;
  const int n_metric_eta = n*(n + 1)*n;
  const int n_metric_zeta = n*n*(n + 1);
  mfem::Vector left(3), point_metric(3);

  for (int element = 0; element < mesh.GetNE(); ++element)
    {
      auto *transformation = mesh.GetElementTransformation(element);
      for (int direction = 0; direction < 3; ++direction)
        for (int transverse_b = 0; transverse_b < n; ++transverse_b)
          for (int transverse_a = 0; transverse_a < n; ++transverse_a)
            for (int face = 1; face < n; ++face)
              {
                const int line = direction == 0
                  ? transverse_b*n*n + transverse_a*n
                  : (direction == 1 ? transverse_b*n*n + transverse_a
                                    : transverse_b*n + transverse_a);
                transformation->SetIntPoint(&cache.ir_vol->IntPoint(line));
                transformation->AdjugateJacobian().GetRow(direction, left);
                for (int l = 0; l < face; ++l)
                  {
                    mfem::Vector sum(3);
                    sum = 0.0;
                    for (int m = 0; m < n; ++m)
                      {
                        const int point = line + (direction == 0 ? m
                          : (direction == 1 ? m*n : m*n*n));
                        transformation->SetIntPoint(&cache.ir_vol->IntPoint(point));
                        transformation->AdjugateJacobian().GetRow(direction,
                                                                   point_metric);
                        point_metric *= D[l*n + m];
                        sum += point_metric;
                      }
                    sum *= cache.ir->IntPoint(l).weight;
                    left += sum;
                  }

                const int point = direction == 0 ? line + face
                  : (direction == 1 ? transverse_b*n*n + face*n + transverse_a
                                    : face*n*n + line);
                const int stride = direction == 0 ? n_metric_xi
                  : (direction == 1 ? n_metric_eta : n_metric_zeta);
                const mfem::real_t *computed = direction == 0 ? xi
                  : (direction == 1 ? eta : zeta);
                for (int component = 0; component < 3; ++component)
                  EXPECT_CLOSE(computed[((element*stride + point)*3)+component],
                               left(component), 2.0e-14);
              }
    }
  EXPECT_EQ(cache.elMetric.Size(), mesh.GetNE()*nq*9);
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
