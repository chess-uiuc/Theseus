#!/usr/bin/env python3
"""Independent sampled audit of CNS forcing and transport contributions."""
from mms_paths import REPOSITORY_ROOT
import argparse
import json
from pathlib import Path
import numpy as np
from trigonometric_euler_study import exact_fields


def primitive(coordinates, time):
    state = exact_fields(coordinates, time)
    rho = state[:, 0]
    velocity = state[:, 1:3]/rho[:, None]
    pressure = .4*(state[:, 3] - .5*rho*np.sum(velocity**2, axis=1))
    return np.column_stack((rho, velocity, pressure))


def viscosity(values, mu, gas_constant, sutherland):
    result = np.broadcast_to(mu, len(values)).copy()
    if sutherland:
        temperature = values[:, 3]/(values[:, 0]*gas_constant)
        result *= (temperature/273.15)**1.5*(273.15 + 110.4)/(temperature + 110.4)
    return result


def viscous_parts(coordinates, time, mu, gas_constant, prandtl, sutherland=False):
    values = primitive(coordinates, time)
    rho, pressure = values[:, 0], values[:, 3]
    gradient = np.empty((len(values), 4, 2))
    for direction in range(2):
        shifted = coordinates.astype(complex)
        shifted[:, direction] += 1e-30j
        gradient[:, :, direction] = np.imag(primitive(shifted, time))/1e-30
    velocity_gradient = gradient[:, 1:3, :]
    divergence = np.trace(velocity_gradient, axis1=1, axis2=2)
    local_mu = viscosity(values, mu, gas_constant, sutherland)
    stress = local_mu[:, None, None]*(velocity_gradient + velocity_gradient.transpose(0, 2, 1)
                 - (2./3.)*divergence[:, None, None]*np.eye(2))
    work = np.einsum('ni,nij->nj', values[:, 1:3], stress)
    temperature_gradient = (gradient[:, 3, :]/rho[:, None]
                            - pressure[:, None]*gradient[:, 0, :]/rho[:, None]**2)/gas_constant
    conductivity = local_mu*3.5*gas_constant/prandtl
    return stress, work, conductivity[:, None]*temperature_gradient


def audit(mu=.02, gas_constant=1., prandtl=.72, sutherland=False):
    nodes, weights = np.polynomial.legendre.leggauss(20)
    x, y = np.meshgrid(nodes, nodes, indexing='ij')
    points = np.column_stack((x.ravel(), y.ravel()))
    measure = np.outer(weights, weights).ravel()/4.
    time = .173
    stress_divergence = np.zeros((len(points), 2))
    work_divergence = np.zeros(len(points))
    heat_divergence = np.zeros(len(points))
    inviscid_divergence = np.zeros((len(points), 4))
    values = primitive(points, time)
    local_mu = viscosity(values, mu, gas_constant, sutherland)
    frozen_stress_divergence = np.zeros_like(stress_divergence)
    frozen_work_divergence = np.zeros_like(work_divergence)
    frozen_heat_divergence = np.zeros_like(heat_divergence)
    step = 2e-4
    for direction in range(2):
        for offset, weight in zip((-2, -1, 1, 2), (1., -8., 8., -1.)):
            shifted = points.copy()
            shifted[:, direction] += offset*step
            stress, work, heat = viscous_parts(shifted, time, mu, gas_constant, prandtl, sutherland)
            factor = weight/(12.*step)
            stress_divergence += factor*stress[:, :, direction]
            work_divergence += factor*work[:, direction]
            heat_divergence += factor*heat[:, direction]
            q = exact_fields(shifted, time)
            rho = q[:, 0]
            velocity = q[:, 1:3]/rho[:, None]
            pressure = .4*(q[:, 3] - .5*rho*np.sum(velocity**2, axis=1))
            flux = q*velocity[:, direction, None]
            flux[:, 1 + direction] += pressure
            flux[:, 3] += pressure*velocity[:, direction]
            inviscid_divergence += factor*flux
            if sutherland:
                # Hold each sampled point's coefficients fixed while differencing
                # its surrounding fluxes. The difference isolates product-rule terms.
                frozen_stress, frozen_work, frozen_heat = viscous_parts(
                    shifted, time, local_mu, gas_constant, prandtl)
                frozen_stress_divergence += factor*frozen_stress[:, :, direction]
                frozen_work_divergence += factor*frozen_work[:, direction]
                frozen_heat_divergence += factor*frozen_heat[:, direction]
    time_derivative = np.imag(exact_fields(points, time + 1e-30j))/1e-30
    source = time_derivative + inviscid_divergence
    source[:, 1:3] -= stress_divergence
    source[:, 3] -= work_divergence + heat_divergence
    def rms(values):
        if values.ndim == 1:
            return float(np.sqrt(np.sum(measure*values**2)))
        return np.sqrt(np.sum(measure[:, None]*values**2, axis=0)).tolist()
    report = dict(mu=mu, R_gas=gas_constant, Pr=prandtl, time=time,
                measure='20x20 Gauss-sampled physical RMS over [-1,1]^2',
                time_derivative=rms(time_derivative), inviscid_divergence=rms(inviscid_divergence),
                stress_divergence=rms(stress_divergence), viscous_work_divergence=rms(work_divergence),
                conduction_divergence=rms(heat_divergence), full_source=rms(source))
    if sutherland:
        temperature = values[:, 3]/(values[:, 0]*gas_constant)
        report.update(transport='sutherland', mu0=mu,
                      temperature_range=[float(temperature.min()), float(temperature.max())],
                      viscosity_range=[float(local_mu.min()), float(local_mu.max())],
                      coefficient_gradient_stress=rms(stress_divergence - frozen_stress_divergence),
                      coefficient_gradient_work=rms(work_divergence - frozen_work_divergence),
                      coefficient_gradient_conduction=rms(heat_divergence - frozen_heat_divergence))
        del report['mu']
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sutherland', action='store_true',
                        help='Audit mu0=.02, R=1/600 with temperature-dependent transport')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(gas_constant=1./600. if args.sutherland else 1., sutherland=args.sutherland)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
