Template calls
==============

This page complements :doc:`dynamic` with class-template and function-template
usage patterns.

Class template calls
--------------------

Class templates use a one-step constructor form:

.. code-block:: python

  obj = TemplateClass(*ctor_args, **ctor_kwargs, **template_kwargs)

Class template example
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  # Given C++:
  # template <typename T, int BLOCK_DIM_X> struct BlockScan { __device__ BlockScan(); ... };
  # template <int N, typename T> class Foo { __device__ Foo(T t); ... };
  apis = bind_cxx_class_templates(
      decls.class_templates,
      header_path=source,
      shim_writer=shim_writer,
  )
  BlockScan, Foo = apis

  @cuda.jit(link=shim_writer.links())
  def kernel(inp, out):
      block_scan = BlockScan(T=np.int32, BLOCK_DIM_X=128)
      foo = Foo(t=inp[0], N=128)  # T is deduced from constructor args.
      out[0] = foo.get_t()

Function template calls
-----------------------

Function templates are bound as normal callable Python handles:

.. code-block:: python

  # Given C++:
  # template <typename T> __device__ T add(T a, T b);
  # template <typename T> __device__ T add(T a, T b, T c);
  funcs = bind_cxx_function_templates(
      function_templates=decls.function_templates,
      shim_writer=shim_writer,
  )
  add = next(f for f in funcs if f.__name__ == "add")

Function template example
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  @cuda.jit(link=shim_writer.links())
  def kernel(x, y, z, out):
      out[0] = add(x[0], y[0])        # picks 2-arg overload
      out[1] = add(x[0], y[0], z[0])  # picks 3-arg overload

Argument deduction
------------------

- Class template type parameters can be deduced from constructor arguments.
- Explicit template kwargs are validated against deduced types, and conflicts
  raise typing errors.
- Constructor keyword arguments must appear before template-parameter keywords.
- Function-template overload and template-parameter deduction are performed from
  call argument types at typing time.
- Argument-intent overrides can change visible call arguments and therefore
  affect deduction behavior; see :doc:`/argument_intents`.
