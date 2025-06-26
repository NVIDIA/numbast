import os

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def decls(data_folder):
    path = os.path.join(data_folder, "sample_itanium_mangled_names.cu")

    decls = parse_declarations_from_source(
        path,
        [path],
        "sm_80",
    )

    return decls


def test_itanium_mangled_name(decls):
    structs = decls.structs
    functions = decls.functions

    assert len(structs) == 2
    assert len(functions) == 4

    assert structs[0].name == "Foo"
    assert structs[1].name == "Bar"

    assert structs[0].methods[0].name == "Foo"
    assert structs[1].methods[0].name == "Bar"

    assert structs[0].methods[0].mangled_name == "_ZN3FooC1Ev"
    assert structs[1].methods[0].mangled_name == "_ZN3BarC1Ev"

    assert functions[0].name == "operator+"
    assert functions[1].name == "operator+"
    assert functions[2].name == "inner_func"
    assert functions[3].name == "inner_func"

    assert functions[0].mangled_name == "_ZplRK3FooS1_"
    assert functions[1].mangled_name == "_ZplRK3BarS1_"
    assert functions[2].mangled_name == "_ZN3ns110inner_funcE3Foo3Bar"
    assert functions[3].mangled_name == "_ZN3ns210inner_funcE3Foo3Bar"
