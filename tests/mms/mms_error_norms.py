"""Physical error norms for GLL tensor polynomials on uniform Cartesian meshes."""
from mms_paths import REPOSITORY_ROOT
import numpy as np


def tensor_volume_errors(keys, state, dimension, elements_per_axis, origin, width,
                         time, exact_fields, polynomial_order, quadrature_points):
    """Integrate reconstructed DG polynomials, preserving separate element traces.

    keys contain three cell-center coordinates followed by three node coordinates.
    This routine deliberately supports only affine, uniform 2D/3D Gauss-Lobatto case meshes.
    The maximum is sampled at quadrature points; it is not a continuum supremum.
    """
    if dimension not in (2, 3) or elements_per_axis < 1 or width <= 0.:
        raise ValueError('invalid uniform Cartesian mesh')
    if polynomial_order < 1 or quadrature_points < polynomial_order + 1:
        raise ValueError('quadrature must resolve the squared polynomial')
    if keys.shape != (len(state), 6) or state.shape[1] != dimension + 2:
        raise ValueError('invalid conservative field/coordinate shape')
    if not np.all(np.isfinite(keys)) or not np.all(np.isfinite(state)):
        raise ValueError('nonfinite field/coordinates')
    centers = keys[:, :dimension]
    coordinates = keys[:, 3:3 + dimension]
    elements = np.floor((centers - origin)/width + 1e-9).astype(int)
    local_positions = 2.*((coordinates - origin)/width - elements) - 1.
    legendre = np.polynomial.legendre.Legendre.basis(polynomial_order)
    reference_nodes = np.r_[-1., np.real_if_close(legendre.deriv().roots()).astype(float), 1.]
    nodes = np.argmin(np.abs(local_positions[..., None] - reference_nodes), axis=-1)
    if (np.any(elements < 0) or np.any(elements >= elements_per_axis)
            or not np.allclose(local_positions, reference_nodes[nodes], rtol=0., atol=1e-7)):
        raise ValueError('output is not on the declared uniform GLL mesh')
    element_count = elements_per_axis**dimension
    element_ids = sum(elements[:, d]*elements_per_axis**d for d in range(dimension))
    nodal_state = np.full((element_count,) + (polynomial_order + 1,)*dimension + (dimension + 2,), np.nan)
    nodal_state[(element_ids,) + tuple(nodes[:, d] for d in range(dimension))] = state
    if not np.all(np.isfinite(nodal_state)):
        raise ValueError('missing element GLL nodes')
    # Repeated visualization subcell vertices must represent the same DG trace.
    reconstructed = nodal_state[(element_ids,) + tuple(nodes[:, d] for d in range(dimension))]
    if not np.allclose(reconstructed, state, rtol=0., atol=1e-12):
        raise ValueError('inconsistent repeated DG nodes')

    points, weights = np.polynomial.legendre.leggauss(quadrature_points)
    basis = np.ones((quadrature_points, polynomial_order + 1))
    for node in range(polynomial_order + 1):
        for other in range(polynomial_order + 1):
            if other != node:
                basis[:, node] *= (points - reference_nodes[other])/(reference_nodes[node] - reference_nodes[other])
    reference_grid = np.stack(np.meshgrid(*([points]*dimension), indexing='ij'), axis=-1)
    physical_weights = np.ones((quadrature_points,)*dimension)
    for d in range(dimension):
        shape = [1]*dimension
        shape[d] = quadrature_points
        physical_weights *= weights.reshape(shape)
    physical_weights *= (width/2.)**dimension
    l1 = np.zeros(dimension + 2)
    l2_squared = np.zeros(dimension + 2)
    sampled_maximum = np.zeros(dimension + 2)
    momentum_l1 = 0.
    momentum_maximum = 0.
    for start in range(0, element_count, 256):
        ids = np.arange(start, min(start + 256, element_count))
        indices = np.column_stack([(ids//elements_per_axis**d) % elements_per_axis
                                   for d in range(dimension)])
        positions = (origin + width*indices.reshape((-1,) + (1,)*dimension + (dimension,))
                     + .5*width*(reference_grid + 1.))
        exact = exact_fields(positions.reshape(-1, dimension), time).reshape(
            (-1,) + (quadrature_points,)*dimension + (dimension + 2,))
        if dimension == 3:
            numerical = np.einsum('ai,bj,ck,eijkq->eabcq', basis, basis, basis,
                                  nodal_state[ids], optimize=True)
        else:
            numerical = np.einsum('ai,bj,eijq->eabq', basis, basis, nodal_state[ids], optimize=True)
        error = numerical - exact
        if not np.all(np.isfinite(error)):
            raise ValueError('nonfinite interpolated/exact field')
        momentum_error = np.linalg.norm(error[..., 1:dimension + 1], axis=-1)
        momentum_l1 += float(np.sum(momentum_error*physical_weights))
        momentum_maximum = max(momentum_maximum, float(np.max(momentum_error)))
        sum_axes = tuple(range(dimension + 1))
        l1 += np.sum(np.abs(error)*physical_weights[..., None], axis=sum_axes)
        l2_squared += np.sum(error**2*physical_weights[..., None], axis=sum_axes)
        sampled_maximum = np.maximum(sampled_maximum, np.max(np.abs(error), axis=sum_axes))
    return dict(l1=l1.tolist(), l2=np.sqrt(l2_squared).tolist(),
                sampled_max=sampled_maximum.tolist(), quadrature_points=quadrature_points,
                momentum=dict(l1=momentum_l1,
                              l2=float(np.sqrt(np.sum(l2_squared[1:dimension + 1]))),
                              sampled_max=momentum_maximum))


def p2_volume_errors(keys, state, dimension, elements_per_axis, origin, width,
                     time, exact_fields, quadrature_points=8):
    return tensor_volume_errors(keys, state, dimension, elements_per_axis, origin, width,
                                time, exact_fields, 2, quadrature_points)


def headline_l2(record):
    """Density, Euclidean vector-momentum error, and total-energy L2 norms."""
    component = record['component_physical']['l2']
    return [component[0], float(np.linalg.norm(component[1:-1])), component[-1]]
