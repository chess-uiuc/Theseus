# Shared gas-property evaluation

`GasModelInterface::EvaluateProperties` is a host-only API for requesting several
properties of one conservative state. The caller selects the properties explicitly:

```cpp
GasPropertyRequest request;
request.pressure = true;
request.temperature = true;
request.sound_speed = true;

const auto properties = gas.EvaluateProperties(state, request);
```

The interface accepts a `DofStateView`. Concrete gas models also accept the other
state views through their templated overload. Available requests are pressure,
temperature, sound speed, dynamic viscosity, and thermal conductivity. The returned
`GasProperties` contains named values; unrequested values remain zero and must not
be interpreted as evaluated physical properties. An empty request performs no EOS
or transport evaluation.

CPG delegates to its existing EOS and transport methods. It retains constant or
Sutherland viscosity and conductivity `mu * cp / Pr`.

LTE recovers temperature once for a nonempty request. When table properties are
needed, `LTEPropertySample` locates the density/temperature cell once and reuses
its interpolation weights for all requested properties. Its interpolation uses the
same four-corner expression as the runtime EOS. Viscosity comes from the LTE table
unless Sutherland is enabled; conductivity always comes from the LTE conductivity
table. It is not reconstructed from a fixed Prandtl number. Requested property
indices absent from the table layout are rejected.

The sample is a temporary view of the gas model's existing host tables. It neither
owns nor copies them. The caller must keep the model and its tables alive during
evaluation. The API adds no state to the gas model and does not change device-cache
construction, runtime property queries, or timestep kernels.

Visualization combines selected field dependencies into one request per point.
See [visualization output](visualization.md) for field selection and ownership.

`GasPropertyTests_standard` and `GasPropertyTests_sutherland` run in both CTest
configurations. They check CPG and analytic LTE values, agreement with individual
queries, table endpoints and interpolation, selective requests, unavailable
properties, and exactly one LTE temperature recovery (zero for an empty request).
