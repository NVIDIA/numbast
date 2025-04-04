import os

import pytest

from ast_canopy import parse_declarations_from_source


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

    # Python API to allow instantiation
    assert foo_t_one.get_instantiated_c_stmt() == "foo_t<1>"
    assert foo_t_two.get_instantiated_c_stmt() == "foo_t<2>"
    assert smem_three.get_instantiated_c_stmt() == "smem<foo_t<1>, foo_t<2>>"

    # Constexpr evaluation of arguments that are compile time evaluable.
    res = smem_three.evaluate_constexpr_value(
        foo_t_one, foo_t_two, header=sample_constexpr_function_template
    )

    assert res is not None
    assert res.value == "3"
    assert res.type_.name == "const unsigned int"
