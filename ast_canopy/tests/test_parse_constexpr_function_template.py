import os

import pytest

from ast_canopy import parse_declarations_from_source
from pylibastcanopy import template_param_kind


@pytest.fixture(scope="module")
def sample_constexpr_function_template():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        current_directory, "data/", "sample_constexpr_function_template.cu"
    )


def test_parse_constexpr_function_template(sample_constexpr_function_template):
    decls = parse_declarations_from_source(
        sample_constexpr_function_template,
        [sample_constexpr_function_template],
        "sm_80",
    )

    func_temp = decls.function_templates
    class_temp = decls.class_templates

    assert len(func_temp) == 1
    assert len(class_temp) == 1

    assert func_temp[0].function.name == "smem"
    assert len(func_temp[0].template_parameters) == 2
    assert func_temp[0].template_parameters[0].name == "TA"
    assert func_temp[0].template_parameters[0].kind == template_param_kind.type_
    assert func_temp[0].template_parameters[1].name == "TB"
    assert func_temp[0].template_parameters[1].kind == template_param_kind.type_
    assert func_temp[0].function.is_constexpr

    assert class_temp[0].record.name == "foo_t"
    assert len(class_temp[0].template_parameters) == 1
    assert class_temp[0].template_parameters[0].name == "N"
    assert (
        class_temp[0].template_parameters[0].kind
        == template_param_kind.non_type
    )
