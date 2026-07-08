// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: MIT
#pragma once
#include "mfem.hpp"
#include "LTETable.hpp"
#include "plato_Cpp_library_interface.h"

using mfem::real_t;

struct PlatoMixture
{
    real_t rho    = 0.0;
    real_t T      = 0.0;
    real_t P      = 0.0;
    real_t e      = 0.0;
    real_t cv     = 0.0;
    real_t cp     = 0.0;
    real_t R_eq   = 0.0;
    real_t gam_eq = 0.0;
    real_t c      = 0.0;
    real_t mu     = 0.0;
    real_t lambda = 0.0;
};

static void
plato_set_state(const real_t rho, const real_t T, PlatoMixture &mix)
{
    mix.rho = rho;
    mix.T = T;

    real_t UKB = 1.380649e-23;

    int nb_comp = plato_get_nb_comp();
    int nb_spec = plato_get_nb_species();
    int nb_temp = plato_get_nb_temp();
    double X_tol = 1e-12;

    mfem::Vector yc(nb_comp), Xc(nb_comp);
    mfem::Vector Xi(nb_spec), Xitol(nb_spec), Xip(nb_spec), Xim(nb_spec), yi(nb_spec),
                  Ri(nb_spec), ei(nb_spec), hi(nb_spec), Ji(nb_spec),
                  Di(nb_spec),Dij(nb_spec*(nb_spec + 1)/2), di(nb_spec);
    mfem::Vector temp(nb_temp), lambda_int(nb_temp);

    real_t R, P, nb, e, betaT, alpha, cv, gam, cp, c,
            mu, sigma, lambda_trh, lambda_tre, lambda_reactive;

    plato_get_Ri(Ri.GetData());

    temp = T;

    // Computing LTE composition
    int flag = 0;
    yc = 1.0;
    Xc = 1.0;
    plato_get_eq_composition_mass(&rho, &T, yc.GetData(), yi.GetData(), &flag);
    plato_mass_to_mole_fractions(yi.GetData(), Xi.GetData());

    // Number density and pressure
    R  = Theseus::Kernels::Dot(nb_spec, Ri.GetData(), yi.GetData());
    P  = rho * R * T;
    nb = P / (UKB*T);

    // Energies and enthalpy per unit-mass
    plato_get_species_energy(temp.GetData(), ei.GetData());
    for(int sp=0; sp < nb_spec; sp++) hi[sp] = ei[sp] + Ri[sp]*T;
    e = Theseus::Kernels::Dot(nb_spec, yi.GetData(), ei.GetData());

    betaT = plato_get_eq_isoth_comp(&P, &T, Xi.GetData());
    alpha = plato_get_eq_coeff_th_exp(&P, &T, Xi.GetData());
    cv    = plato_get_eq_cv(&rho, &T, yi.GetData());
    gam   = 1.0 + alpha*alpha*T/(rho*betaT*cv);
    cp    = gam*cv;
    c     = std::sqrt(gam*P/rho);

    // Transport properties
    // dynamic viscosity, thermal conductivity, components of thermal conductivity
    plato_get_transp_coeff_comp(&nb, Xi.GetData(), temp.GetData(), &mu, &sigma, &lambda_trh, &lambda_tre, lambda_int.GetData(), Di.GetData());

    double eps = 1e-5;
    double Tepsp1 = T*(1.0 + eps);
    double Tepsm1 = T*(1.0 - eps);
    // reactive thermal conductivity
    // (NOTE: pass mole fractions with tolerance to procedure solving Stefan-Maxwell's equations)
    plato_get_eq_composition_mole(&P, &Tepsp1, Xc.GetData(), Xip.GetData(), &flag);
    plato_get_eq_composition_mole(&P, &Tepsm1, Xc.GetData(), Xim.GetData(), &flag);
    plato_get_bin_diff_coeff(&nb, &T, &T, Xi.GetData(), Dij.GetData());
    for(int sp=0; sp < nb_spec; sp++) di[sp] = -0.5 * (Xip[sp] - Xim[sp])/(eps*T);

    double sum_Xitol = 0.0, X;
    for(int sp=0; sp < nb_spec; sp++)
    {
      X = Xi[sp] + X_tol;
      Xitol[sp] = X;
      sum_Xitol += X;
    }

    sum_Xitol = 1.0/sum_Xitol;
    for(int sp=0; sp < nb_spec; sp++) Xitol[sp] *= sum_Xitol;

    plato_get_species_diff_flux(&T, &T, &nb, Xitol.GetData(), Dij.GetData(), di.GetData(), Ji.GetData());
    lambda_reactive = Theseus::Kernels::Dot(nb_spec, Ji.GetData(), hi.GetData());

    mix.P = P;
    mix.e = e;
    mix.cv = cv;
    mix.cp = cp;
    mix.R_eq = R;
    mix.gam_eq = gam;
    mix.c = c;
    mix.mu = mu;
    mix.lambda = lambda_trh + lambda_tre + lambda_int[0] + lambda_reactive;
}
