Class template calls
====================

This page complements :doc:`dynamic` with class template constructor usage and examples.

Usage
-----

Class templates use a one-step constructor call form:

.. code-block:: python

  obj = TemplateClass(*ctor_args, **ctor_kwargs, **template_kwargs)

Example
-------

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

Notes
-----

- Template parameters are provided by keyword name.
- Constructor keyword arguments must appear before template-parameter keywords.
- Explicit template kwargs are validated against deduced template types.
