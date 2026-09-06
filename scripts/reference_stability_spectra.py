#!/usr/bin/env python3
"""Compute reference one-dimensional DGSEM stability spectra.

This offline calibration utility requires NumPy.  It prints the unmodified
spectral radii; safety margins are applied in StabilityEstimate.hpp.
"""

import argparse

import numpy as np


def gll_operators(order):
    coefficient = np.zeros(order + 1)
    coefficient[-1] = 1.0
    if order == 1:
        reference_nodes = np.array([-1.0, 1.0])
    else:
        legendre_derivative = np.polynomial.legendre.legder(coefficient)
        interior = np.polynomial.legendre.legroots(legendre_derivative)
        reference_nodes = np.concatenate(([-1.0], interior, [1.0]))

    nodes = 0.5 * (reference_nodes + 1.0)
    points = order + 1
    barycentric = np.empty(points)
    for row in range(points):
        barycentric[row] = 1.0 / np.prod(nodes[row] - np.delete(nodes, row))

    derivative = np.zeros((points, points))
    for row in range(points):
        for column in range(points):
            if row != column:
                derivative[row, column] = (
                    barycentric[column]
                    / barycentric[row]
                    / (nodes[row] - nodes[column])
                )
        derivative[row, row] = -np.sum(derivative[row, :])

    polynomial = np.polynomial.legendre.legval(reference_nodes, coefficient)
    weights = 1.0 / (order * (order + 1) * polynomial**2)
    return derivative, weights


def reference_operators(order, elements):
    derivative, weights = gll_operators(order)
    points = order + 1
    degrees = elements * points
    advection = np.zeros((degrees, degrees))
    gradient = np.zeros((degrees, degrees))

    for element in range(elements):
        block = slice(element * points, (element + 1) * points)
        left = element * points
        right = left + points - 1
        previous_right = ((element - 1) % elements) * points + points - 1
        next_left = ((element + 1) % elements) * points

        # Positive-speed, strong-form upwind advection on unit cells.
        advection[block, block] -= derivative
        advection[left, left] -= 1.0 / weights[0]
        advection[left, previous_right] += 1.0 / weights[0]

        # BR1 auxiliary gradient with centered traces.  The scalar diffusion
        # operator then applies the same DG derivative a second time.
        gradient[block, block] += derivative
        gradient[left, left] += 0.5 / weights[0]
        gradient[left, previous_right] -= 0.5 / weights[0]
        gradient[right, right] -= 0.5 / weights[-1]
        gradient[right, next_left] += 0.5 / weights[-1]

    return advection, gradient @ gradient


def spectral_radius(operator):
    return np.max(np.abs(np.linalg.eigvals(operator)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-order", type=int, default=12)
    parser.add_argument("--elements", type=int, default=24)
    args = parser.parse_args()

    print("order,advection_radius,br1_diffusion_radius")
    for order in range(1, args.max_order + 1):
        advection, diffusion = reference_operators(order, args.elements)
        print(
            f"{order},{spectral_radius(advection):.15g},"
            f"{spectral_radius(diffusion):.15g}"
        )


if __name__ == "__main__":
    main()
