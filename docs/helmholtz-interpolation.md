# Bicubic Helmholtz interpolation helper

`LTEHelmholtzInterpolation.hpp` evaluates a bicubic Hermite cell and its physical
coordinate derivatives directly from the four corner jets. It does not construct
a local 4x4 coefficient array. The caller supplies H, H_r, H_T and H_rT as
mass-specific potential and physical density/temperature derivatives.

`Theseus::HelmholtzInterpolation::Cell` borrows four property-plane pointers,
each positioned at the cell's lower-density/lower-temperature node. Density
samples are contiguous. `temperature_stride` is the number of density samples
between consecutive temperature rows. Both cell widths must be positive, all
four corners must be readable, and the table owner must outlive evaluation.

```cpp
using namespace Theseus::HelmholtzInterpolation;
Cell cell{H_lower, Hr_lower, HT_lower, HrT_lower, density_count, drho, dT};
const auto H = Evaluate<0, 0>(cell, xi, eta);
const auto H_r = Evaluate<1, 0>(cell, xi, eta);
const auto H_T = Evaluate<0, 1>(cell, xi, eta);
const auto H_rr = Evaluate<2, 0>(cell, xi, eta);
const auto H_rT = Evaluate<1, 1>(cell, xi, eta);
const auto H_TT = Evaluate<0, 2>(cell, xi, eta);
```

`xi=(rho-rho_lower)/drho` and `eta=(T-T_lower)/dT` are normalized coordinates.
The returned derivatives are with respect to physical rho and T; widths are
applied internally. Derivative orders0,1,2 in each coordinate are compile-time
arguments. No lookup, allocation, synchronization or coordinate validation is
performed here. Host/device callers must select a valid cell and rebind pointers
to the correct memory space before using this view in a kernel.

Corner contributions are read and consumed as scalars. The helper evaluates the
same polynomial as an expanded coefficient-matrix formulation; summation order
and rounding can differ. This is not a centered or shape-preserving interpolant,
and no positivity/concavity guarantee follows from the interpolation alone.

The helper is available independently of PLATO. It does not itself change the
solver's gas model, table generation or flux selection.

CTest `HelmholtzInterpolation_CPU` checks all16 bicubic monomials, nonuniform
cells, endpoints, nine derivative combinations, input immutability, arbitrary
corner jets against an exact-rational reference, and actual forall evaluation.
CUDA builds additionally register `HelmholtzInterpolation_CUDA`, requiring a
CUDA backend. The interpolation tests use forward roundoff bounds based on the
input coefficient scale; these are not CPU/GPU evolution acceptance tolerances.

```sh
cmake --build /absolute/build --target helmholtz_interpolation_tests -j 2
ctest --test-dir /absolute/build -R '^HelmholtzInterpolation_' --output-on-failure
```

## Stable physical divided differences

`DividedDifference<Axis, D_rho, D_T>(cell, xi1, eta1, xi2, eta2)` evaluates the
physical secant of the requested derivative inside one cell. `Axis=0` varies
rho and keeps eta1 fixed; `Axis=1` varies T and keeps xi1 fixed. Both endpoints
must belong to that cell. Coincident coordinates give the derivative limit.

The cubic basis secants are factored polynomials: for example the secant of x^3
is x1^2+x1*x2+x2^2. Corner contributions are consumed directly, avoiding a
subtraction of two completed large interpolants and division by their small
coordinate separation. The additional physical cell width is included in scaling.
No full coefficient array or kernel allocation is introduced.
