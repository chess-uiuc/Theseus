# ChannelFlow reference data

The compressed NumPy files contain cell-local fields at steps 0, 25 and 50 and
50 printed integral-change records for the unforced ChannelFlow case. Separate
references cover tabulated and Sutherland viscosity. Both use tabulated thermal
conductivity and zero runtime blending.

The references were generated with a single CPU rank. Each JSON records the
input configuration, fixed schedule, executable and build-cache hashes, case
source hashes and timing. The reference fields define regression expectations;
they are not independent mathematical or experimental accuracy measurements.

The regression reads these files and never updates them. The development-only
`record` action in `tests/channel_flow_regression.py` requires a single-rank CPU
run and refuses to overwrite either reference file. Review intentional changes
to inputs or numerical behavior before replacing reference data.

See the [case guide](../../LTE/NavierStokes/ChannelFlow/README.md) for execution,
field tolerances and runtime limits.
