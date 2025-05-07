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


def test_evaluate_constexpr(sample_constexpr_function_template, foo_t, smem):
    # Python API to allow tracking instantiation
    foo_t_one = foo_t.instantiate(N=1)
    foo_t_two = foo_t.instantiate(N=2)

    smem_three = smem.instantiate(TA=foo_t_one, TB=foo_t_two)

    # Constexpr evaluation of arguments that are compile time evaluable.
    res = smem_three.evaluate_constexpr_value(foo_t_one, foo_t_two)

    assert res == 3
