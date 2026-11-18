# Runtime manufactured solutions

This directory contains exact manufactured states, derivatives and device-ready
forcing evaluators. `sources/` contains their adapters to the general source
registry; `Registration.hpp` groups source and exact-state registration. The
general source factory delegates to it. Shared source interfaces and physical research sources remain in the
usual source-infrastructure directories.

Use project-relative includes such as
`verification/mms/TrigonometricCNSMMS.hpp`. The runtime source identifiers and
initial/exact-state function names are independent of file layout and remain
unchanged. These capabilities are part of the normal Theseus executable; no
verification-only build or executable is required.

Independent reference calculations, convergence studies and result plotting live
in `tests/mms/`. Runtime headers must not include anything from that test directory.
See [the verification guide](../../../docs/mms-verification.md) for equations,
parameters, observed accuracy and limitations.
