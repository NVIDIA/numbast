import os
import pickle

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def decls(data_folder):
    path = os.path.join(data_folder, "sample_itanium_mangled_names.cu")

    decls = parse_declarations_from_source(
        path,
        [path],
        "sm_80",
        defines=["__device__=__attribute__((device))"],
        cuda_header_mode=True,
    )

    return decls


def test_itanium_mangled_name(decls):
    structs = decls.structs
    functions = decls.functions

    assert len(structs) == 2
    assert len(functions) == 6

    assert structs[0].name == "Foo"
    assert structs[1].name == "Bar"

    assert structs[0].methods[0].name == "Foo"
    assert structs[0].methods[1].name == "Foo"
    assert structs[1].methods[0].name == "Bar"
    assert structs[1].methods[1].name == "Bar"

    assert structs[0].methods[0].mangled_name == "_ZN3FooC1Ev"
    assert structs[0].methods[1].mangled_name == "_ZN3FooC1Ei"
    assert structs[1].methods[0].mangled_name == "_ZN3BarC1Ev"
    assert structs[1].methods[1].mangled_name == "_ZN3BarC1Ei"

    assert functions[0].name == "operator+"
    assert functions[1].name == "operator+"
    assert functions[2].name == "inner_func"
    assert functions[3].name == "inner_func"

    assert functions[0].mangled_name == "_ZplRK3FooS1_"
    assert functions[1].mangled_name == "_ZplRK3BarS1_"
    assert functions[2].mangled_name == "_ZN3ns110inner_funcE3Foo3Bar"
    assert functions[3].mangled_name == "_ZN3ns210inner_funcE3Foo3Bar"


def test_c_linkage_and_variadic_metadata(decls):
    functions = {function.name: function for function in decls.functions}

    c_device = functions["c_device_func"]
    assert c_device.is_c_linkage is True
    assert c_device.is_variadic is False
    assert c_device.mangled_name == "c_device_func"

    c_variadic = functions["c_variadic_func"]
    assert c_variadic.is_c_linkage is True
    assert c_variadic.is_variadic is True
    assert c_variadic.mangled_name == "c_variadic_func"

    assert functions["inner_func"].is_c_linkage is False


def test_c_linkage_metadata_survives_pickle(decls):
    functions = {
        function.name: pickle.loads(pickle.dumps(function))
        for function in decls.functions
    }

    assert functions["c_device_func"].is_c_linkage is True
    assert functions["c_device_func"].is_variadic is False
    assert functions["c_variadic_func"].is_c_linkage is True
    assert functions["c_variadic_func"].is_variadic is True


def test_variadic_member_function_metadata_survives_pickle(decls):
    foo = next(struct for struct in decls.structs if struct.name == "Foo")
    method = next(
        method for method in foo.methods if method.name == "variadic_member"
    )

    assert method.is_c_linkage is False
    assert method.is_variadic is True
    restored = pickle.loads(pickle.dumps(method))
    assert restored.is_c_linkage is False
    assert restored.is_variadic is True
