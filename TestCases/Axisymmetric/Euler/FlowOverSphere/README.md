# Axisymmetric Euler Flow Over a Sphere

This case models Mach 2 inviscid flow over a unit sphere. The two-dimensional
mesh is the meridional half-plane `(z, r)`; revolving the semicircular slip wall
about `r = 0` produces the sphere.

Run from the repository root with an axisymmetric build:

```sh
scripts/run_theseus.sh -b /path/to/axisymmetric-build \
  -c TestCases/Axisymmetric/Euler/FlowOverSphere/config.json -p 2
```

The checked-in Gmsh 2.2 mesh is deterministic. Regenerate it without external
dependencies by running `python3 generate_mesh.py` in this directory.
