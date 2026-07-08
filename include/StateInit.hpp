// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ConditionFactory.hpp"

namespace Prandtl
{

  // Isentropic Vortex initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> LTEVortexIC(mfem::real_t radius,
                                                                      mfem::real_t vel_inf,
                                                                      mfem::real_t beta,
                                                                      mfem::real_t rho_inf,
                                                                      mfem::real_t temp_inf)
  {
    return [=](const mfem::Vector &x, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 2, "");

      const mfem::real_t xc = 0.0;
      const mfem::real_t yc = 0.0;

      // Using CPG constants only to shape the initial field.
      const mfem::real_t gamma = 1.4;
      const mfem::real_t R_gas = 287.05;
      const mfem::real_t gm1   = gamma - 1.0;
      const mfem::real_t cp    = gamma * R_gas / gm1;

      // using the perfect-gas relations
      const mfem::real_t pres_inf = rho_inf * R_gas * temp_inf;

      mfem::real_t dx = x(0) - xc;
      mfem::real_t dy = x(1) - yc;

      mfem::real_t r2rad = (dx*dx + dy*dy) / (radius * radius);

      const mfem::real_t exp_half = std::exp(-0.5 * r2rad);
      const mfem::real_t exp_full = std::exp(-r2rad);

      // Vortex velocity field
      const mfem::real_t velX = vel_inf * (1.0 - beta * dy / radius * exp_half);
      const mfem::real_t velY = vel_inf * (      beta * dx / radius * exp_half);
      const mfem::real_t vel2 = velX * velX + velY * velY;

      // Gaussian temperature perturbation
      const mfem::real_t temp =
        temp_inf - 0.5 * (vel_inf * beta) * (vel_inf * beta) / cp * exp_full;

      // Safety clamp so the IC never goes nonphysical
      const mfem::real_t temp_safe = std::max(temp, 0.2 * temp_inf);

      // CPG isentropic relations used only to generate the spatial field
      const mfem::real_t den  = rho_inf * std::pow(temp_safe / temp_inf, 1.0 / gm1);
      const mfem::real_t pres = pres_inf * std::pow(temp_safe / temp_inf, gamma / gm1);

      const mfem::real_t rhoe = pres / gm1;
      const mfem::real_t rhoE = rhoe + 0.5 * den * vel2;

      y(0) = den;
      y(1) = den * velX;
      y(2) = den * velY;
      y(3) = rhoE;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterLTEVortex
  {
    RegisterLTEVortex()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition5("LTEVortexIC", LTEVortexIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterLTEVortex regLTEVortex;

  // LTE Blob initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> LTEBlobIC(mfem::real_t radius,
                                                                    mfem::real_t T_inf,
                                                                    mfem::real_t T_blob,
                                                                    mfem::real_t P_inf)
  {
    return [=](const mfem::Vector &x, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 2, "");

      const mfem::real_t xc = 0.0;
      const mfem::real_t yc = 0.0;

      mfem::real_t dx = x(0) - xc;
      mfem::real_t dy = x(1) - yc;

      mfem::real_t r2rad = (dx*dx + dy*dy) / (radius * radius);

      const mfem::real_t exp_full = std::exp(-r2rad);

      // Quiescent velocity field
      const mfem::real_t velX = 0.0;
      const mfem::real_t velY = 0.0;
      const mfem::real_t vel2 = velX * velX + velY * velY;

      // Gaussian temperature perturbation
      const mfem::real_t T = T_inf + (T_blob - T_inf) * exp_full;

      mfem::real_t R_gas = 287.05;
      const mfem::real_t gamma = 1.4;

      const mfem::real_t den   = P_inf / (R_gas * T);
      const mfem::real_t rhoe  = P_inf / (gamma-1.0);
      const mfem::real_t rhoE  = rhoe + 0.5 * den * vel2;

      y(0) = den;
      y(1) = den * velX;
      y(2) = den * velY;
      y(3) = rhoE;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterLTEBlob
  {
    RegisterLTEBlob()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition4("LTEBlobIC", LTEBlobIC);
    }
  };

  // Global static instance to ensure registration happens at startup.
  static RegisterLTEBlob regLTEBlob;

  // Taylor Green Vortex initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> TaylorGreenVortexIC(mfem::real_t gamma, mfem::real_t Ma)
  {
    return [gamma, Ma](const mfem::Vector &x, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 3, "");

      mfem::real_t den, velX, velY, velZ, energy, p, p0 = 1.0 / (gamma * Ma * Ma);

      den = 1.0;
      velX = std::sin(x(0)) * std::cos(x(1)) * std::cos(x(2));
      velY = -std::cos(x(0)) * std::sin(x(1)) * std::cos(x(2));
      velZ = 0.0;
      p = p0 + 1.0 / 16.0 * (std::cos(2.0 * x(0)) + std::cos(2.0 * x(1))) * (std::cos(2.0 * x(2)) + 2);

      energy = p / (gamma - 1.0) + 0.5 * den * (velX * velX + velY * velY + velZ * velZ);

      y(0) = den;
      y(1) = den * velX;
      y(2) = den * velY;
      y(3) = den * velZ;
      y(4) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterTaylorGreenVortex
  {
    RegisterTaylorGreenVortex()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition2("TaylorGreenVortexIC", TaylorGreenVortexIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterTaylorGreenVortex regTaylorGreenVortex;





  // Taylor Green Vortex initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> TaylorGreenVortex2DIC(mfem::real_t gamma, mfem::real_t Ma)
  {
    return [gamma, Ma](const mfem::Vector &x, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 2, "");
      MFEM_ASSERT(y.Size() == 4, "");

      mfem::real_t den, velX, velY, energy, p, p0 = 1.0 / (gamma * Ma * Ma);

      den = 1.0;
      velX = std::sin(x(0)) * std::cos(x(1));
      velY = -std::cos(x(0)) * std::sin(x(1));
      p = p0 + 0.25 * (std::cos(2.0 * x(0)) + std::cos(2.0 * x(1)));

      energy = p / (gamma - 1.0) + 0.5 * den * (velX * velX + velY * velY);

      y(0) = den;
      y(1) = den * velX;
      y(2) = den * velY;
      y(3) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterTaylorGreenVortex2D
  {
    RegisterTaylorGreenVortex2D()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition2("TaylorGreenVortex2DIC", TaylorGreenVortex2DIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterTaylorGreenVortex2D regTaylorGreenVortex2D;





  // Lid-driven Cavity initial condition function
  std::function<void(const mfem::Vector&, mfem::Vector&)> LidDrivenCavityIC(mfem::real_t Ma, mfem::real_t gamma)
  {
    return [Ma, gamma] (const mfem::Vector &x, mfem::Vector &y)
    {
      mfem::real_t p = 1.0 / (Ma * Ma * gamma);
      y(0) = 1.0;
      y(1) = 0.0;
      y(2) = 0.0;
      y(3) = p / (gamma - 1.0);
    };
  }

  // Lid-driven Cavity heat flow boundary condition scalar function for adiabatic walls and lid
  std::function<mfem::real_t(const mfem::Vector&)> LidDrivenCavityAdiaBCFunction()
  {
    return [] (const mfem::Vector &x)
    {
      return 0.0;
    };
  }

  // Lid-driven Cavity heat flow boundary condition scalar for adiabatic walls and lid
  const Prandtl::BC_Scalar LidDrivenCavityAdiaBCScalar = 0.0;

  // Lid-driven Cavity velocity boundary condition function for walls
  std::function<void(const mfem::Vector&, mfem::Vector&)> LidDrivenCavityWallVelBCFunction()
  {
    return [] (const mfem::Vector &x, mfem::Vector &vel)
    {
      vel(0) = 0.0;
      vel(1) = 0.0;
    };
  }
  // Lid-driven Cavity velocity boundary condition vector for walls
  const Prandtl::BC_Vector LidDrivenCavityWallVelBCVector({0.0, 0.0});

  // Lid-driven Cavity velocity boundary condition function for lid
  std::function<void(const mfem::Vector&, mfem::Vector&)> LidDrivenCavityLidVelBCFunction(mfem::real_t Re, mfem::real_t mu)
  {
    return [Re, mu] (const mfem::Vector &x, mfem::Vector &vel)
    {
      vel(0) = Re * mu / 2.0;
      vel(1) = 0.0;
    };
  }

  // Lid-driven Cavity velocity boundary condition vector for lid
  const Prandtl::BC_Vector LidDrivenCavityLidVelBCVector({1.0, 0.0});

  // Registration helper that automatically registers these functions
  // along with associated boundary marker arrays.
  struct RegisterLidDrivenCavity
  {
    RegisterLidDrivenCavity()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition2("LidDrivenCavityIC", LidDrivenCavityIC);

      // Register boundary conditions with functions.
      Prandtl::ConditionFactory::Instance().RegisterScalarFunctionBoundaryCondition0("LidDrivenCavityAdiaBCFunction",
                                                                                     LidDrivenCavityAdiaBCFunction);
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition0("LidDrivenCavityWallVelBCFunction",
                                                                                     LidDrivenCavityWallVelBCFunction);
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition2("LidDrivenCavityLidVelBCFunction",
                                                                                     LidDrivenCavityLidVelBCFunction);

      // Register boundary conditions with constant scalars/vectors.
      Prandtl::ConditionFactory::Instance().RegisterScalarBoundaryCondition("LidDrivenCavityAdiaBCScalar",
                                                                            LidDrivenCavityAdiaBCScalar);
      Prandtl::ConditionFactory::Instance().RegisterVectorBoundaryCondition("LidDrivenCavityWallVelBCVector",
                                                                            LidDrivenCavityWallVelBCVector);
      Prandtl::ConditionFactory::Instance().RegisterVectorBoundaryCondition("LidDrivenCavityLidVelBCVector",
                                                                            LidDrivenCavityLidVelBCVector);

    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterLidDrivenCavity regLidDrivenCavity;


  // Triple Point Shock Interaction initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> TriplePointShockInteractionIC(mfem::real_t gammaM1Inverse)
  {
    return [gammaM1Inverse] (const mfem::Vector &x, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 2, "Triple Point Shock Interaction is a 2D problem");
      mfem::real_t density, velocity_x, velocity_y, pressure, energy;
      if (x(0) <= 1.0)
        {
          density = 1.0;
          pressure = 1.0;
        }
      else if (x(0) <= 7.0 && x(1) <= 1.5)
        {
          density = 1.0;
          pressure = 0.1;
        }
      else
        {
          density = 0.125;
          pressure = 0.1;
        }

      velocity_x = 0.0;
      velocity_y = 0.0;
      energy = pressure * gammaM1Inverse;

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = density * velocity_y;
      y(3) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterTriplePointShockInteraction
  {
    RegisterTriplePointShockInteraction()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("TriplePointShockInteractionIC", TriplePointShockInteractionIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterTriplePointShockInteraction regTriplePointShockInteraction;

  // Isentropic Vortex initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> IsentropicVortexIC(mfem::real_t radius, mfem::real_t Minf,
                                                                             mfem::real_t beta, mfem::real_t R_gas,
                                                                             mfem::real_t gamma)
  {
    return [radius, Minf, beta, R_gas, gamma](const mfem::Vector &x, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 2, "");

      const mfem::real_t xc = 0.0, yc = 0.0;

      // Nice units
      const mfem::real_t vel_inf = 1.;
      const mfem::real_t den_inf = 1.;

      // Derive remainder of background state from this and Minf
      const mfem::real_t pres_inf = (den_inf / gamma) * (vel_inf / Minf) * (vel_inf / Minf);
      const mfem::real_t temp_inf = pres_inf / (den_inf * R_gas);

      mfem::real_t r2rad = 0.0;
      r2rad += (x(0) - xc) * (x(0) - xc);
      r2rad += (x(1) - yc) * (x(1) - yc);
      r2rad /= (radius * radius);

      const mfem::real_t shrinv1 = 1.0 / (gamma - 1.0);

      const mfem::real_t velX = vel_inf * (1 - beta * (x(1) - yc) / radius * std::exp(-0.5 * r2rad));
      const mfem::real_t velY = vel_inf * beta * (x(0) - xc) / radius * std::exp(-0.5 * r2rad);
      const mfem::real_t vel2 = velX * velX + velY * velY;

      const mfem::real_t specific_heat = R_gas * gamma * shrinv1;
      const mfem::real_t temp = temp_inf - 0.5 * (vel_inf * beta) * (vel_inf * beta) / specific_heat * std::exp(-r2rad);

      const mfem::real_t den = den_inf * std::pow(temp / temp_inf, shrinv1);
      const mfem::real_t pres = den * R_gas * temp;
      const mfem::real_t energy = shrinv1 * pres / den + 0.5 * vel2;

      y(0) = den;
      y(1) = den * velX;
      y(2) = den * velY;
      y(3) = den * energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterIsentropicVortex
  {
    RegisterIsentropicVortex()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition5("IsentropicVortexIC", IsentropicVortexIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterIsentropicVortex regIsentropicVortex;

  // Backward Facing Step initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> BackwardFacingStepIC(mfem::real_t gammaM1Inverse)
  {
    return [gammaM1Inverse] (const mfem::Vector &x, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 2, "Backward Facing Step is a 2D problem");
      mfem::real_t density, velocity_x, velocity_y, pressure, energy;
      if (x(0) < 0.5)
        {
          density = 5.9970;
          velocity_x = 98.5914;
          pressure = 11666.5;
        }
      else
        {
          density = 1.0;
          velocity_x = 0.0;
          pressure = 1.0;
        }
      velocity_y = 0.0;
      energy = pressure * gammaM1Inverse + density * 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = density * velocity_y;
      y(3) = energy;
    };
  }

  // Backward Facing Step boundary condition
  const Prandtl::BC_Vector BackwardFacingStepLeftBCVector({5.9970, 5.9970 * 98.5914, 0.0,
      11666.5 / (1.4 - 1.0) + 0.5 * 5.9970 * 98.5914 * 98.5914});

  // Registration helper that automatically registers these functions
  struct RegisterBackwardFacingStep
  {
    RegisterBackwardFacingStep()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("BackwardFacingStepIC", BackwardFacingStepIC);
      // Register boundary condition.
      Prandtl::ConditionFactory::Instance().RegisterVectorBoundaryCondition("BackwardFacingStepLeftBCVector",
                                                                            BackwardFacingStepLeftBCVector);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterBackwardFacingStep regBackwardFacingStep;

  // Double Mach Reflection initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> DoubleMachReflectionIC(mfem::real_t gammaM1Inverse)
  {
    return [gammaM1Inverse] (const mfem::Vector &x, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 2, "DMR is a 2D problem");

      if (x(0) < 1.0 / 6.0 + x(1) / std::sqrt(3))
        {
          y(0) = 8.0;
          y(1) = 8.0 * 7.144709581221619;
          y(2) = -8.0 * 4.125;
          y(3) = 116.5 * gammaM1Inverse  + 0.5 * y(0) * (7.144709581221619 * 7.144709581221619 + 4.125 * 4.125);
        }
      else
        {
          y(0) = 1.4;
          y(1) = 0.0;
          y(2) = 0.0;
          y(3) = 1.0 * gammaM1Inverse;
        }
    };
  }

  // Double Mach reflection boundary condition for top boundary
  std::function<void(const mfem::Vector&, mfem::real_t, mfem::Vector&)> DoubleMachReflectionTopBCFunction(const mfem::real_t gammaM1Inverse)
  {
    return [gammaM1Inverse] (const mfem::Vector &x, mfem::real_t t, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 2, "DMR is a 2D problem");

      if (x(0) < 1.0 / 6.0 + (x(1) + 20.0 * t) / std::sqrt(3))
        {
          y(0) = 8.0;
          y(1) = 8.0 * 7.144709581221619;
          y(2) = -8.0 * 4.125;
          y(3) = 116.5 * gammaM1Inverse + 0.5 * y(0) * (7.144709581221619 * 7.144709581221619 + 4.125 * 4.125);
        }
      else
        {
          y(0) = 1.4;
          y(1) = 0.0;
          y(2) = 0.0;
          y(3) = 1.0 * gammaM1Inverse;
        }
    };
  }

  // Double Mach Reflection conservative state boundary condition vector for left and bottom boundaries
  const Prandtl::BC_Vector DoubleMachReflectionLeftBottom1BCVector({8.0, 8.0 * 7.144709581221619, -8.0 * 4.125,
      116.5 * 1.0 / (1.4 - 1.0) + 0.5 * 8.0 * (7.144709581221619 * 7.144709581221619 + 4.125 * 4.125)});

  // Registration helper that automatically registers these functions
  struct RegisterDoubleMachReflection
  {
    RegisterDoubleMachReflection()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("DoubleMachReflectionIC", DoubleMachReflectionIC);

      // Register boundary conditions
      Prandtl::ConditionFactory::Instance().RegisterVectorBoundaryCondition("DoubleMachReflectionLeftBottom1BCVector",
                                                                            DoubleMachReflectionLeftBottom1BCVector);
      Prandtl::ConditionFactory::Instance().RegisterVectorTDFunctionBoundaryCondition1("DoubleMachReflectionTopBCFunction",
                                                                                       DoubleMachReflectionTopBCFunction);
    }
  };

  // Global static instance to ensure registration happens at startup.
  static RegisterDoubleMachReflection regDoubleMachReflection;

  // Supersonic Freestream initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> NagashimaIC()
  {
    static mfem::real_t gamma = 1.4;
    static mfem::real_t R_gas = 287.05;
    static mfem::real_t Ma = 2.0;                                      // Mach number
    static mfem::real_t p0 = 303975;                                   // stagnation pressure (Pa)
    static mfem::real_t a0 = 360.63;                                   // stagnation speed (m/s)
    static mfem::real_t C = 1 + 0.5 * (gamma-1) * Ma * Ma;             // 1 + (gamma-1)/2 * M^2
    static mfem::real_t a = a0/std::sqrt(C);                           // speed of sound
    static mfem::real_t ua = Ma * a;                                   // freestream velocity
    static mfem::real_t pa = p0 / std::pow(C, gamma/(gamma-1));        // freestream pressure
    static mfem::real_t pi = pa;                                       // pressure for the inlet and initilization
    static mfem::real_t Ta = (a * a)/(gamma * R_gas);                  // freestream temperature
    static mfem::real_t rhoa = pa / (R_gas * Ta);                      // freestream density
    static mfem::real_t E = pi/(gamma - 1.0) + 0.5 * rhoa * ua * ua;   // Specific total energy
    return [] (const mfem::Vector &x, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 2, "Nagashima Ramjet is a 2D problem");

      y(0) = rhoa;
      y(1) = rhoa * ua;
      y(2) = 0.0;
      y(3) = E;

    };
  }

  // Inlet velocity ramping boundary condition
  std::function<void(const mfem::Vector&, mfem::real_t, mfem::Vector&)>NagashimaRampingInletBC(mfem::real_t t_ramp)
  {
    static mfem::real_t gamma = 1.4;
    static mfem::real_t R_gas = 287.05;
    static mfem::real_t Ma = 2.0;                                      // Mach number
    static mfem::real_t p0 = 303975;                                   // stagnation pressure (Pa)
    static mfem::real_t a0 = 360.63;                                   // stagnation speed (m/s)
    static mfem::real_t C = 1 + 0.5 * (gamma-1) * Ma * Ma;             // 1 + (gamma-1)/2 * M^2
    static mfem::real_t a = a0/std::sqrt(C);                           // speed of sound
    static mfem::real_t ua = Ma * a;                                   // freestream velocity
    static mfem::real_t pa = p0 / std::pow(C, gamma/(gamma-1));        // freestream pressure
    static mfem::real_t pi = pa;                                       // pressure for the inlet and initilization
    static mfem::real_t Ta = (a * a)/(gamma * R_gas);                  // freestream temperature
    static mfem::real_t rhoa = pa / (R_gas * Ta);                      // freestream density
    static mfem::real_t E = pi/(gamma - 1.0) + 0.5 * rhoa * ua * ua;   // Specific total energy
    return [t_ramp] (const mfem::Vector &x, mfem::real_t t, mfem::Vector &y)
    {
      mfem::real_t s    = std::tanh((t - 0.5*t_ramp)/(0.1*t_ramp));
      mfem::real_t ramp = std::clamp(0.5*(1.0 + s), 0.0, 1.0);
      mfem::real_t u   = Ma * a * ramp;
      y(0) = rhoa;
      y(1) = rhoa*u;
      y(2) = 0.0;
      y(3) = pi/(gamma-1.0) + 0.5*rhoa*u*u;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterNagashima
  {
    RegisterNagashima()
    {
      static mfem::real_t gamma = 1.4;
      static mfem::real_t R_gas = 287.05;
      static mfem::real_t Ma = 2.0;                                      // Mach number
      static mfem::real_t p0 = 303975;                                   // stagnation pressure (Pa)
      static mfem::real_t a0 = 360.63;                                   // stagnation speed (m/s)
      static mfem::real_t C = 1 + 0.5 * (gamma-1) * Ma * Ma;             // 1 + (gamma-1)/2 * M^2
      static mfem::real_t a = a0/std::sqrt(C);                           // speed of sound
      static mfem::real_t ua = Ma * a;                                   // freestream velocity
      static mfem::real_t pa = p0 / std::pow(C, gamma/(gamma-1));        // freestream pressure
      static mfem::real_t pi = pa;                                       // pressure for the inlet and initilization
      static mfem::real_t Ta = (a * a)/(gamma * R_gas);                  // freestream temperature
      static mfem::real_t rhoa = pa / (R_gas * Ta);                      // freestream density
      static mfem::real_t E = pi/(gamma - 1.0) + 0.5 * rhoa * ua * ua;   // Specific total energy
      // Subsonic outflow
      const mfem::real_t NagashimaOutletPressure = pa;
      // Supersonic Freestream boundary condition
      const Prandtl::BC_Vector NagashimaInletBCVector({rhoa, rhoa * ua, 0.0, E});

      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition0("NagashimaIC", NagashimaIC);
      // Register boundary condition.
      Prandtl::ConditionFactory::Instance().RegisterVectorBoundaryCondition("NagashimaInletBCVector",
                                                                            NagashimaInletBCVector);
      Prandtl::ConditionFactory::Instance().RegisterVectorTDFunctionBoundaryCondition1("NagashimaRampingInletBC",
                                                                                       NagashimaRampingInletBC);
      Prandtl::ConditionFactory::Instance().RegisterScalarBoundaryCondition("NagashimaOutletPressure",
                                                                            NagashimaOutletPressure);

    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterNagashima registerNagashima;


  // Kelvin Helmholtz initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> KelvinHelmholtzInstabilyIC(mfem::real_t gammaM1Inverse)
  {
    return [gammaM1Inverse] (const mfem::Vector &x, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 2, "KHI is a 2D problem");

      mfem::real_t density, velocity_x, velocity_y, pressure, energy, B;

      B = std::tanh(15.0 * x(1) + 7.5) - std::tanh(15.0 * x(1) - 7.5);
      density = 0.5 + 0.75 * B;
      velocity_x = 0.5 * (B - 1.0);
      velocity_y = 0.1 * std::sin(2.0 * M_PI * x(0));
      pressure = 1.0;
      energy = pressure * gammaM1Inverse + density * 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = density * velocity_y;
      y(3) = energy;

      // if (x(1) < 0.5 && x(1) > -0.5)
      // {
      //     y(0) = 2.0;
      //     y(1) = -1.0;
      //     y(2) = 2.0 * 0.01 * std::sin(M_PI * x(0));
      //     y(3) = 2.5 * gammaM1Inverse  + 0.5 * (y(1) * y(1) + y(2) * y(2)) / y(0);
      // }
      // else
      // {
      //     y(0) = 1.0;
      //     y(1) = 0.5;
      //     y(2) = 0.01 * std::sin(M_PI * x(0));;
      //     y(3) = 2.5 * gammaM1Inverse + 0.5 * (y(1) * y(1) + y(2) * y(2)) / y(0);
      // }
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterKelvinHelmholtzInstability
  {
    RegisterKelvinHelmholtzInstability()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("KelvinHelmholtzInstabilityIC",
                                                                      KelvinHelmholtzInstabilyIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterKelvinHelmholtzInstability regKelvinHelmholtzInstability;


  // Supersonic Freestream initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> RampIC()
  {
    return [] (const mfem::Vector &x, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 2, "Ramp is a 2D problem");
      const mfem::real_t gamma = 1.4;
      const mfem::real_t Ma = 2.0;
      const mfem::real_t a = 1; //340.294;
      const mfem::real_t rho = 1; //0.225;
      const mfem::real_t u = Ma * a;
      const mfem::real_t p = rho*a*a/1.4;
      y(0) = rho;
      y(1) = rho * u;
      y(2) = 0.0;
      y(3) = (p / (gamma-1)) + 0.5 * rho * u * u;
    };
  }


  const Prandtl::BC_Vector RampInletBCVector({1.0, 2.0, 0.0, ((1.0/1.4)/0.4) + 0.5 * 1.0 * 2.0 * 2.0});
  // std::function<void(const mfem::Vector&, mfem::real_t, mfem::Vector&)>RampingInletBC(mfem::real_t t_ramp)
  // {
  //     return [t_ramp] (const mfem::Vector &x, mfem::real_t t, mfem::Vector &y)
  //     {
  //         const mfem::real_t gamma = 1.4;
  //         const mfem::real_t Ma = 2.0;
  //         const mfem::real_t a = 1;
  //         const mfem::real_t rho = 1.0;
  //         mfem::real_t s    = std::tanh((t - 0.5*t_ramp)/(0.1*t_ramp));
  //         mfem::real_t ramp = std::clamp(0.5*(1.0 + s), 0.0, 1.0);
  //         mfem::real_t u   = Ma * a * ramp;
  //         mfem::real_t p   = 1.0;
  //         y(0) = rho;
  //         y(1) = rho*u;
  //         y(2) = 0.;
  //         y(3) = p/(1.4-1.0) + 0.5*rho*u*u;
  //     };
  // }


  // Registration helper that automatically registers these functions
  struct RegisterRamp
  {
    RegisterRamp()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition0("RampIC", RampIC);
      // Register boundary condition.
      Prandtl::ConditionFactory::Instance().RegisterVectorBoundaryCondition("RampInletBCVector", RampInletBCVector);
      // Prandtl::ConditionFactory::Instance().RegisterVectorTDFunctionBoundaryCondition1("RampingInletBC", RampingInletBC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterRamp registerRamp;

  // Forward Facing Step initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> ForwardFacingStepIC(mfem::real_t gammaM1Inverse)
  {
    return [gammaM1Inverse] (const mfem::Vector &x, mfem::Vector &y)
    {
      MFEM_ASSERT(x.Size() == 2, "Forward Facing Step is a 2D problem");

      y(0) = 1.4;
      y(1) = 1.4 * 3.0;
      y(2) = 0.0;
      y(3) = 1.0 * gammaM1Inverse + 0.5 * 1.4 * 3.0 * 3.0;
    };
  }

  // Forward Facing Step boundary condition
  const Prandtl::BC_Vector ForwardFacingStepLeftBCVector({1.4, 1.4 * 3.0, 0.0, 1.0 / (1.4 - 1.0) + 0.5 * 1.4 * 3.0 * 3.0});

  // Registration helper that automatically registers these functions
  struct RegisterForwardFacingStep
  {
    RegisterForwardFacingStep()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("ForwardFacingStepIC", ForwardFacingStepIC);
      // Register boundary condition.
      Prandtl::ConditionFactory::Instance().RegisterVectorBoundaryCondition("ForwardFacingStepLeftBCVector",
                                                                            ForwardFacingStepLeftBCVector);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterForwardFacingStep regForwardFacingStep;

  // Woodward-Colella Blast Wave initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> WoodwardColellaBlastWaveIC(mfem::real_t gamma)
  {
    return [gamma](const mfem::Vector &x, mfem::Vector &y)
    {
      mfem::real_t density, velocity_x, pressure, energy;
      MFEM_ASSERT(x.Size() == 1, "");

      density = 1.0;
      velocity_x = 0.0;

      if (x(0) <= 0.1)
        {
          pressure = 1000.0;
        }
      else if (x(0) <= 0.9)
        {
          pressure = 0.01;
        }
      else
        {
          pressure = 100.0;
        }

      energy = pressure / (gamma - 1.0) + density * 0.5 * (velocity_x * velocity_x);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterWoodwardColellaBlastWave
  {
    RegisterWoodwardColellaBlastWave()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("WoodwardColellaBlastWaveIC",
                                                                      WoodwardColellaBlastWaveIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterWoodwardColellaBlastWave regWoodwardColellaBlastWave;

  // 123 Problem initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> Problem123IC(mfem::real_t gamma)
  {
    return [gamma](const mfem::Vector &x, mfem::Vector &y)
    {
      mfem::real_t density, velocity_x, pressure, energy;
      MFEM_ASSERT(x.Size() == 1, "");

      if (x(0) < 0.5)
        {
          density = 1.0;
          velocity_x = -2.0;
          pressure = 0.4;
        }
      else
        {
          density = 1.0;
          velocity_x = 2.0;
          pressure = 0.4;
        }

      energy = pressure / (gamma - 1.0) + density * 0.5 * (velocity_x * velocity_x);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterProblem123
  {
    RegisterProblem123()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("Problem123IC", Problem123IC);

      // Register boundary condition for left and right boundaries.
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("Problem123LeftBC", Problem123IC);
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("Problem123RightBC", Problem123IC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterProblem123 regProblem123;

  // Lax Shock Tube initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> LaxShockTubeIC(mfem::real_t gamma)
  {
    return [gamma](const mfem::Vector &x, mfem::Vector &y)
    {
      mfem::real_t density, velocity_x, pressure, energy;
      MFEM_ASSERT(x.Size() == 1, "");

      if (x(0) < 0.5)
        {
          density = 0.445;
          velocity_x = 0.698;
          pressure = 3.528;
        }
      else
        {
          density = 0.5;
          velocity_x = 0.0;
          pressure = 0.571;
        }

      energy = pressure / (gamma - 1.0) + density * 0.5 * (velocity_x * velocity_x);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterLaxShockTube
  {
    RegisterLaxShockTube()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("LaxShockTubeIC", LaxShockTubeIC);

      // Register boundary condition for left and right boundaries.
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("LaxShockTubeLeftBC", LaxShockTubeIC);
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("LaxShockTubeRightBC", LaxShockTubeIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterLaxShockTube regLaxShockTube;

  // Sod Shock Tube initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> ModifiedSodShockTubeIC(mfem::real_t gamma)
  {
    return [gamma](const mfem::Vector &x, mfem::Vector &y)
    {
      mfem::real_t density, velocity_x, pressure, energy;
      MFEM_ASSERT(x.Size() == 1, "");

      if (x(0) < 0.5)
        {
          density = 1.0;
          velocity_x = 0.75;
          pressure = 1.0;
        }
      else
        {
          density = 0.125;
          velocity_x = 0.0;
          pressure = 0.1;
        }

      energy = pressure / (gamma - 1.0) + density * 0.5 * (velocity_x * velocity_x);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterModifiedSodShockTube
  {
    RegisterModifiedSodShockTube()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("ModifiedSodShockTubeIC", ModifiedSodShockTubeIC);

      // Register boundary condition for left and right boundaries.
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("ModifiedSodShockTubeLeftBC",
                                                                                     ModifiedSodShockTubeIC);
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("ModifiedSodShockTubeRightBC",
                                                                                     ModifiedSodShockTubeIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterModifiedSodShockTube regModifiedSodShockTube;

  // Woodward-Colella Blast Wave Collision initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> WoodwardColellaBlastWaveCollisionIC(mfem::real_t gamma)
  {
    return [gamma](const mfem::Vector &x, mfem::Vector &y)
    {
      mfem::real_t density, velocity_x, pressure, energy;
      MFEM_ASSERT(x.Size() == 1, "");

      if (x(0) < 0.5)
        {
          density = 5.99924;
          velocity_x = 19.5975;
          pressure = 460.894;
        }
      else
        {
          density = 5.99242;
          velocity_x = -6.19633 ;
          pressure = 46.0950;
        }

      energy = pressure / (gamma - 1.0) + density * 0.5 * (velocity_x * velocity_x);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterWoodwardColellaBlastWaveCollision
  {
    RegisterWoodwardColellaBlastWaveCollision()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("WoodwardColellaBlastWaveCollisionIC",
                                                                      WoodwardColellaBlastWaveCollisionIC);

      // Register boundary condition for left and right boundaries.
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("WoodwardColellaBlastWaveCollisionLeftBC",
                                                                                     WoodwardColellaBlastWaveCollisionIC);
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("WoodwardColellaBlastWaveCollisionRightBC",
                                                                                     WoodwardColellaBlastWaveCollisionIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterWoodwardColellaBlastWaveCollision regWoodwardColellaBlastWaveCollision;


  // Acoustic Wave initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> AcousticWaveIC(mfem::real_t gamma)
  {
    return [gamma](const mfem::Vector &x, mfem::Vector &y)
    {
      mfem::real_t density, velocity_x, pressure, energy, V;
      mfem::real_t a_inf, M_inf = 0.5, rho_inf = 1.225, p_inf = 1.0;
      a_inf = std::sqrt(gamma * p_inf / rho_inf);
      // MFEM_ASSERT(x.Size() == 1, "");


      // V = 1.0 / (2.0 * M_PI) * std::sin(2.0 * M_PI * x(0));

      // velocity_x = a_inf * (M_inf + 2.0 / ((gamma + 1.0) * a_inf) * V);
      // density = rho_inf * std::pow(1.0 + (gamma - 1.0) / 2.0 * (velocity_x / a_inf - M_inf), 2.0 / (gamma - 1.0));
      // pressure = p_inf * std::pow(1.0 + (gamma - 1.0) / 2.0 * (velocity_x / a_inf - M_inf), 2.0 * gamma / (gamma - 1.0));

      density = rho_inf + 0.2 * std::sin(2.0 * M_PI * x(0));
      velocity_x = a_inf * M_inf;
      pressure = p_inf;

      energy = pressure / (gamma - 1.0) + density * 0.5 * (velocity_x * velocity_x);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = energy;
    };
  }

  std::function<void(const mfem::Vector&, mfem::real_t, mfem::Vector&)> AcousticWaveExactSolution(mfem::real_t gamma)
  {
    return [gamma](const mfem::Vector &x, mfem::real_t t, mfem::Vector &y)
    {
      mfem::real_t density, velocity_x, pressure, energy, V, v_;
      mfem::real_t a_inf, M_inf = 0.5, rho_inf = 1.225, p_inf = 1.0;
      a_inf = std::sqrt(gamma * p_inf / rho_inf);
      // MFEM_ASSERT(x.Size() == 1, "");

      // v_ = 0.0;
      // for (int i = 0; i < 200; i++)
      // {
      //     V = 1.0 / (2.0 * M_PI) * std::sin(2.0 * M_PI * (x(0) - v_ * t));
      //     if (std::abs(V - v_) < 1e-14)
      //     {
      //         break;
      //     }
      //     v_ = V;
      // }

      // velocity_x = a_inf * (M_inf + 2.0 / ((gamma + 1.0) * a_inf) * V);
      // density = rho_inf * std::pow(1.0 + (gamma - 1.0) / 2.0 * (velocity_x / a_inf - M_inf), 2.0 / (gamma - 1.0));
      // pressure = p_inf * std::pow(1.0 + (gamma - 1.0) / 2.0 * (velocity_x / a_inf - M_inf), 2.0 * gamma / (gamma - 1.0));

      density = rho_inf + 0.2 * std::sin(2.0 * M_PI * (x(0) - a_inf * M_inf * t));
      velocity_x = a_inf * M_inf;
      pressure = p_inf;

      energy = pressure / (gamma - 1.0) + density * 0.5 * (velocity_x * velocity_x);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterAcousticWave
  {
    RegisterAcousticWave()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("AcousticWaveIC", AcousticWaveIC);

      // Register exact solution.
      Prandtl::ConditionFactory::Instance().RegisterVectorTDFunctionBoundaryCondition1("AcousticWaveExactSolution",
                                                                                       AcousticWaveExactSolution);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterAcousticWave regAcousticWave;

  // Shu-Osher Shock initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> ShuOsherShockIC(mfem::real_t gamma)
  {
    return [gamma](const mfem::Vector &x, mfem::Vector &y)
    {
      mfem::real_t density, velocity_x, pressure, energy;
      MFEM_ASSERT(x.Size() == 1, "");

      if (x(0) < 1.0)
        {
          density = 3.857;
          velocity_x = 2.629;
          pressure = 10.333;
        }
      else
        {
          density = 1.0 + 0.2 * std::sin(5.0 * (x(0) - 5.0));
          velocity_x = 0.0;
          pressure = 1.0;
        }

      energy = pressure / (gamma - 1.0) + density * 0.5 * (velocity_x * velocity_x);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterShuOsherShock
  {
    RegisterShuOsherShock()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("ShuOsherShockIC", ShuOsherShockIC);

      // Register boundary condition functions.
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("ShuOsherShockLeftBC", ShuOsherShockIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterShuOsherShock regShuOsherShock;

  // Woodward-Colella Blast Wave Left Half initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> WoodwardColellaBlastWaveLeftIC(mfem::real_t gamma)
  {
    return [gamma](const mfem::Vector &x, mfem::Vector &y)
    {
      mfem::real_t density, velocity_x, pressure, energy;
      MFEM_ASSERT(x.Size() == 1, "");

      if (x(0) < 0.5)
        {
          density = 1.0;
          velocity_x = 0.0;
          pressure = 1000.0;
        }
      else
        {
          density = 1.0;
          velocity_x = 0.0;
          pressure = 0.01;
        }

      energy = pressure / (gamma - 1.0) + density * 0.5 * (velocity_x * velocity_x);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterWoodwardColellaBlastWaveLeft
  {
    RegisterWoodwardColellaBlastWaveLeft()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("WoodwardColellaBlastWaveLeftIC",
                                                                      WoodwardColellaBlastWaveLeftIC);

      // Register boundary condition for left and right boundaries.
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("WoodwardColellaBlastWaveLeftLeftBC",
                                                                                     WoodwardColellaBlastWaveLeftIC);
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("WoodwardColellaBlastWaveLeftRightBC",
                                                                                     WoodwardColellaBlastWaveLeftIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterWoodwardColellaBlastWaveLeft regWoodwardColellaBlastWaveLeft;


  // Woodward-Colella Blast Wave Right Half initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> WoodwardColellaBlastWaveRightIC(mfem::real_t gamma)
  {
    return [gamma](const mfem::Vector &x, mfem::Vector &y)
    {
      mfem::real_t density, velocity_x, pressure, energy;
      MFEM_ASSERT(x.Size() == 1, "");

      if (x(0) < 0.5)
        {
          density = 1.0;
          velocity_x = 0.0;
          pressure = 0.01;
        }
      else
        {
          density = 1.0;
          velocity_x = 0.0;
          pressure = 100.0;
        }

      energy = pressure / (gamma - 1.0) + density * 0.5 * (velocity_x * velocity_x);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterWoodwardColellaBlastWaveRight
  {
    RegisterWoodwardColellaBlastWaveRight()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("WoodwardColellaBlastWaveRightIC",
                                                                      WoodwardColellaBlastWaveRightIC);

      // Register boundary condition for left and right boundaries.
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("WoodwardColellaBlastWaveRightLeftBC",
                                                                                     WoodwardColellaBlastWaveRightIC);
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("WoodwardColellaBlastWaveRightRightBC",
                                                                                     WoodwardColellaBlastWaveRightIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterWoodwardColellaBlastWaveRight regWoodwardColellaBlastWaveRight;


  // Sod Shock Tube initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> SodShockTubeIC(mfem::real_t gamma)
  {
    return [gamma](const mfem::Vector &x, mfem::Vector &y)
    {
      mfem::real_t density, velocity_x, pressure, energy;
      MFEM_ASSERT(x.Size() == 1, "");

      if (x(0) < 0.5)
        {
          density = 1.0;
          velocity_x = 0.0;
          pressure = 1.0;
        }
      else
        {
          density = 0.125;
          velocity_x = 0.0;
          pressure = 0.1;
        }

      energy = pressure / (gamma - 1.0) + density * 0.5 * (velocity_x * velocity_x);

      y(0) = density;
      y(1) = density * velocity_x;
      y(2) = energy;
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterSodShockTube
  {
    RegisterSodShockTube()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("SodShockTubeIC", SodShockTubeIC);

      // Register boundary condition for left and right boundaries.
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("SodShockTubeLeftBC", SodShockTubeIC);
      Prandtl::ConditionFactory::Instance().RegisterVectorFunctionBoundaryCondition1("SodShockTubeRightBC", SodShockTubeIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterSodShockTube regSodShockTube;

  // LeBlanc Shock Tube initial condition
  std::function<void(const mfem::Vector&, mfem::Vector&)> LeBlancShockTubeIC(mfem::real_t gamma)
  {
    return [gamma](const mfem::Vector &x, mfem::Vector &y)
    {
      mfem::real_t density, velocity_x, pressure, energy;
      MFEM_ASSERT(x.Size() == 1, "");

      if (x(0) < 3.0)
        {
          y(0) = 1.0;
          y(1) = 0.0;
          y(2) = 0.1;
        }
      else
        {
          y(0) = 1e-3;
          y(1) = 0.0;
          y(2) = 1e-9;
        }
    };
  }

  // Registration helper that automatically registers these functions
  struct RegisterLeBlancShockTube
  {
    RegisterLeBlancShockTube()
    {
      // Register initial condition.
      Prandtl::ConditionFactory::Instance().RegisterInitialCondition1("LeBlancShockTubeIC", LeBlancShockTubeIC);
    }
  };
  // Global static instance to ensure registration happens at startup.
  static RegisterLeBlancShockTube regLeBlancShockTube;

}
