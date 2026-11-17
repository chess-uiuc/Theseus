"""Mach-regime definitions and conservative all-space/all-time bounds."""
import numpy as np

REGIMES = ('subsonic', 'transonic', 'supersonic')
MEAN_VELOCITY_SCALES = dict(subsonic=.25, transonic=1., supersonic=5.)


def mean_velocity_scale(regime):
    if regime not in REGIMES:
        raise ValueError('MMS regime must be subsonic, transonic or supersonic')
    return MEAN_VELOCITY_SCALES[regime]


def regime_bounds(regime, dimension=2):
    if dimension not in (2, 3):
        raise ValueError('MMS regime bounds require dimension 2 or 3')
    scale = mean_velocity_scale(regime)
    # Triangle bounds on independent sine/cosine products, valid at every time.
    radius = np.array([.24, .13, .14, .09, .17])
    if dimension == 3:
        radius += np.array([.11, .06, .07, .10, .08])
    mean = np.array([.7, .3, .2])[:dimension]*scale
    lower_velocity = np.maximum(np.abs(mean) - radius[1:dimension + 1], 0.)
    upper_velocity = np.abs(mean) + radius[1:dimension + 1]
    rho_lower, rho_upper = 2. - radius[0], 2. + radius[0]
    p_lower, p_upper = 1. - radius[4], 1. + radius[4]
    sound_lower = np.sqrt(1.4*p_lower/rho_upper)
    sound_upper = np.sqrt(1.4*p_upper/rho_lower)
    return dict(regime=regime, mean_velocity_scale=scale, dimension=dimension,
                density_bounds=[rho_lower, rho_upper], pressure_bounds=[p_lower, p_upper],
                mach_bounds=[float(np.linalg.norm(lower_velocity)/sound_upper),
                             float(np.linalg.norm(upper_velocity)/sound_lower)],
                bound_scope='conservative bounds valid at all positions and times')


def regime_evidence(regime):
    from trigonometric_euler_study import exact_fields
    evidence = regime_bounds(regime)
    x = np.linspace(-1., 1., 65)
    coordinates = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1).reshape(-1, 2)
    state = exact_fields(coordinates, 0., regime)
    rho = state[:, 0]
    speed_squared = np.sum((state[:, 1:3]/rho[:, None])**2, axis=1)
    pressure = .4*(state[:, 3] - .5*rho*speed_squared)
    mach = np.sqrt(speed_squared*rho/(1.4*pressure))
    evidence['sampled_initial_mach_range'] = [float(mach.min()), float(mach.max())]
    low, high = evidence['mach_bounds']
    if (regime == 'subsonic' and high >= 1.
            or regime == 'supersonic' and low < 3.
            or regime == 'transonic' and not mach.min() < 1. < mach.max()):
        raise ValueError('Manufactured field does not establish the requested Mach regime')
    return evidence
