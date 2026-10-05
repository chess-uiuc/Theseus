# Boundary conditions

Boundary handlers and kernels are implemented in [BC kernels](../include/bc_kernels.hpp),
using descriptors and payloads defined in [BC utilities](../include/bc_cache_utilities.hpp).

For constant exterior states and stationary radial profiles converted by the selected
CPG or LTE gas model, see [physical boundary configuration](physical-state-conversion.md#physical-boundary-configuration).
The [PX chamber](px-chamber.md) documents the legacy CPG radial-profile case.
