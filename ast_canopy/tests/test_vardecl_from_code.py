import os

import pytest

from ast_canopy import parse_declarations_from_source, value_from_constexpr_vardecl


@pytest.fixture(scope="module")
def sample_constexpr_function_template():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        current_directory, "data/", "sample_constexpr_function_template.cu"
    )


@pytest.fixture(scope="module")
def decls(sample_constexpr_function_template):
    return parse_declarations_from_source(
        sample_constexpr_function_template,
        [sample_constexpr_function_template],
        "sm_80",
    )


@pytest.fixture(scope="module")
def foo_t(decls):
    assert len(decls.class_templates) == 1
    return decls.class_templates[0]


@pytest.fixture(scope="module")
def smem(decls):
    assert len(decls.function_templates) == 1
    return decls.function_templates[0]


def test_value_from_vardecl(sample_constexpr_function_template, foo_t, smem):
    foo_t_one = foo_t.instantiate(N=1)
    foo_t_two = foo_t.instantiate(N=2)

    smem_three = smem.instantiate(TA=foo_t_one, TB=foo_t_two)

    assert foo_t_one.get_instantiated_c_stmt() == "foo_t<1>"
    assert foo_t_two.get_instantiated_c_stmt() == "foo_t<2>"
    assert smem_three.get_instantiated_c_stmt() == "smem<foo_t<1>, foo_t<2>>"

    assembled_code_template = """
#include <{header}>
{argument_decls}
__device__ constexpr auto ast_canopy_var_value__ = {tfunc_instantiation}
    """

    argument_decls = ""
    for i, foo_t_it in enumerate([foo_t_one, foo_t_two]):
        argument_decls += f"__device__ {foo_t_it.get_instantiated_c_stmt()} arg_{i};\n"

    fml_arglist = ",".join([f"arg_{i}" for i in range(2)])
    tfunc_instantiation = smem_three.get_instantiated_c_stmt() + f"({fml_arglist})"

    assembled_code = assembled_code_template.format(
        header=sample_constexpr_function_template,
        argument_decls=argument_decls,
        tfunc_instantiation=tfunc_instantiation,
    )

    res = value_from_constexpr_vardecl(
        assembled_code, "ast_canopy_var_value__", "sm_80", verbose=True
    )

    assert res is not None
    assert res.value == "3"
    assert res.type_.name == "const unsigned int"
    assert res.name == "ast_canopy_var_value__"

    # res = smem_three.evaluate_constexpr_value(
    #     foo_t_one, foo_t_two, header=sample_constexpr_function_template)

    # assert res is not None
    # assert res.value == "3"
    # assert res.type_.name == "const unsigned int"
    # assert res.name == "ast_canopy_var_value__"
