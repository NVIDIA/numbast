Argument intents
================

Argument intents control how C++ parameters are exposed in generated Numba
bindings. You can configure them statically with ``Function Argument Intents``
in binding config files or programmatically with ``arg_intent`` when calling
the template/API.

C++ source of truth
-------------------

.. code-block:: c++

  struct RunningStats {
    int count;
    float sum;
    float sum_sq;
  };

  __device__ void stats_update(RunningStats &state, float x);

  __device__ void stats_get_mean(const RunningStats &state, float &mean_out);

  __device__ bool stats_update_and_get_zscore(
      RunningStats &state, float x, float &zscore_out);

  __device__ uint4 texture_footprint(
      cudaTextureObject_t texture, float x, float y,
      unsigned int *single_mip_level);

Example config
--------------

.. code-block:: yaml

  Function Argument Intents:
    stats_update:
      state: inout_ptr
    stats_get_mean:
      state: in
      mean_out: out_ptr
    stats_update_and_get_zscore:
      state: inout_ptr
      zscore_out: out_return
    texture_footprint:
      single_mip_level: out_return

Intent semantics
----------------

``in``
^^^^^^

- Default mode.
- Parameter stays visible in the Python call signature.
- Parameter is typed as the base value type ``T``.
- For C++ references, updates done by C++ are not surfaced as Python-visible outputs.

``inout_ptr``
^^^^^^^^^^^^^

- Parameter stays visible in the Python call signature.
- Parameter is typed as ``CPointer(T)``.
- Use this when C++ mutates referenced state and the caller should pass an addressable pointer.

``out_ptr``
^^^^^^^^^^^

- Parameter stays visible in the Python call signature.
- Parameter is typed as ``CPointer(T)``.
- Use this for explicit output buffers where the caller owns storage and passes a pointer.

``out_return``
^^^^^^^^^^^^^^

- Parameter is removed from the visible Python call arguments.
- Numbast allocates temporary storage, passes it to C++, then returns the value to Python.
- For scalar pointer outputs such as ``unsigned int*``, the returned value is
  the pointee scalar, not ``CPointer(T)``.
- If C++ also returns a non-``void`` value, generated return type is packed as a tuple.

Generated Python signatures
---------------------------

Representative signatures for the example API:

.. code-block:: python

  # in (default):
  signature(void, _type_RunningStats, float32)

  # inout_ptr:
  signature(void, CPointer(_type_RunningStats), float32)

  # out_ptr:
  signature(void, _type_RunningStats, CPointer(float32))

  # out_return with existing non-void C++ return (bool):
  signature(
      types.Tuple((bool_, float32)),
      CPointer(_type_RunningStats),
      float32,
  )

  # out_return on a scalar pointer output:
  signature(types.Tuple((uint32x4, uint32)), cudaTextureObject_t, float32, float32)

Notes
-----

- ``inout_ptr`` and ``out_ptr`` are only supported on C++ reference parameters
  (``T&`` / ``T&&``). ``out_return`` also supports scalar pointer output
  parameters (``T*``).
- In ``Function Argument Intents``, parameter overrides can be keyed by
  parameter name or 0-based parameter index.

  .. code-block:: yaml

    0: inout_ptr  # demonstrates 0-based parameter indexing for overrides
- ``out_return`` removes that parameter from visible arguments.
- If C++ has a non-``void`` return and one or more ``out_return`` parameters,
  Numbast returns ``types.Tuple((cxx_return, out1, ...))``.
- If C++ returns ``void`` and there is exactly one ``out_return``, Numbast
  returns that value directly (not a tuple).
- If C++ returns ``void`` and there are multiple ``out_return`` parameters,
  Numbast returns ``types.Tuple((out1, out2, ...))``.
