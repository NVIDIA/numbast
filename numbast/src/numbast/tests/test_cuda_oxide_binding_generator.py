# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from numbast.binding_model import (
    BindingModelError,
    build_c_device_model,
    modern_nvvm_required_symbols,
    parse_c_abi_type,
    rust_type,
)
from numbast.tools.binding_config import CudaOxideConfig
from numbast.tools.cuda_oxide_binding_generator import (
    generate_cuda_oxide_bindings,
    render_cuda_oxide_bindings,
    verify_symbol_inventory,
)


class FakeType:
    def __init__(self, name, left_reference=False, right_reference=False):
        self.name = name
        self._left_reference = left_reference
        self._right_reference = right_reference

    def is_left_reference(self):
        return self._left_reference

    def is_right_reference(self):
        return self._right_reference


def function(
    name,
    return_type,
    params=(),
    execution_space="device",
    c_linkage=True,
    variadic=False,
):
    return SimpleNamespace(
        name=name,
        return_type=FakeType(return_type),
        params=[
            SimpleNamespace(name=param_name, type_=FakeType(param_type))
            for param_name, param_type in params
        ],
        exec_space=SimpleNamespace(name=execution_space),
        is_c_linkage=c_linkage,
        is_variadic=variadic,
        mangled_name=name if c_linkage else f"_Z{len(name)}{name}",
    )


def declarations(**overrides):
    values = {
        "functions": [],
        "function_templates": [],
        "class_templates": [],
        "typedefs": [],
        "enums": [],
        "structs": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def config(tmp_path):
    header = tmp_path / "device_api.h"
    header.write_text("// fixture\n", encoding="utf-8")
    return CudaOxideConfig(
        {
            "Name": "Round 1 fixture",
            "Version": 1,
            "Entry Point": str(header),
            "GPU Arch": ["sm_90"],
            "File List": [str(header)],
            "Backend": "cuda-oxide",
            "Output Name": "bindings.rs",
            "API Prefix Removal": {"Function": ["library_"]},
            "CUDA Oxide": {
                "LTOIR Inputs": ["libdevice_api.ltoir"],
                "Manifest Name": "bindings.manifest.json",
                "Type Aliases": {"team_t": "int"},
                "Constants": {"TEAM_WORLD": {"Type": "team_t", "Value": 0}},
            },
        }
    )


def test_qualified_pointer_depth_and_array_order():
    type_ = parse_c_abi_type(FakeType("const int *const *volatile"))
    assert type_.pointer_kinds == ("const", "const")
    assert rust_type(
        type_, SimpleNamespace(enums=[], records=[], typedefs=[])
    ) == ("*const *const i32")

    array = parse_c_abi_type(FakeType("unsigned int[2][3]"))
    assert (
        rust_type(array, SimpleNamespace(enums=[], records=[], typedefs=[]))
        == "[[u32; 3]; 2]"
    )

    pointer_to_array = parse_c_abi_type(FakeType("const int (*)[2][3]"))
    assert pointer_to_array.pointer_kinds == ("const",)
    assert pointer_to_array.array_dimensions == (2, 3)
    assert (
        rust_type(
            pointer_to_array,
            SimpleNamespace(cuda_aliases={}, enums=[], records=[], typedefs=[]),
        )
        == "*const [[i32; 3]; 2]"
    )


def test_cuda_vector_storage_preserves_alignment(config):
    model = build_c_device_model(
        declarations(
            functions=[
                function(
                    "library_vector",
                    "void",
                    (("values", "const double2 *"),),
                )
            ]
        ),
        config,
    )
    rendered, _ = render_cuda_oxide_bindings(model, config)
    assert "pub type double2 = [u128; 1];" in rendered
    assert "core::mem::align_of::<double2>()" in rendered


def test_small_value_abi_records_legacy_limit_and_uses_modern_half(
    config, tmp_path
):
    parsed = declarations(
        functions=[
            function("library_half", "__half", (("value", "__half"),)),
            function(
                "library_bfloat",
                "__nv_bfloat16",
                (("value", "__nv_bfloat16"),),
            ),
        ]
    )
    legacy_model = build_c_device_model(parsed, config)
    assert legacy_model.cuda_aliases["__half"] == ("u16", 2, 2)
    assert modern_nvvm_required_symbols(legacy_model) == [
        "library_bfloat",
        "library_half",
    ]
    rust_path, manifest_path = generate_cuda_oxide_bindings(
        config, tmp_path / "legacy", declarations=parsed
    )
    assert "CUDA-Oxide can call those declarations only on sm_100+" in Path(
        rust_path
    ).read_text(encoding="utf-8")
    compatibility = json.loads(Path(manifest_path).read_text(encoding="utf-8"))[
        "compatibility"
    ]
    assert compatibility["cuda_oxide_nvvm_dialect"] == "legacy-llvm-7"
    assert compatibility["selected_arch_supports_all_symbols"] is False

    modern_raw = copy.deepcopy(config.raw_config)
    modern_raw["GPU Arch"] = ["sm_100"]
    modern = CudaOxideConfig(modern_raw)
    modern_model = build_c_device_model(parsed, modern)
    assert modern_model.cuda_aliases["__half"] == ("f16", 2, 2)
    rendered, _ = render_cuda_oxide_bindings(modern_model, modern)
    assert "pub type __half = f16;" in rendered


def test_generates_golden_rust_and_manifest_without_numba(config, tmp_path):
    template = SimpleNamespace(function=SimpleNamespace(name="cxx_template"))
    parsed = declarations(
        functions=[
            function(
                "library_copy",
                "int",
                (("in", "const float *"), ("output", "int *const *")),
            ),
            function("host_only", "void", execution_space="host"),
        ],
        function_templates=[template],
    )

    rust_path, manifest_path = generate_cuda_oxide_bindings(
        config, tmp_path, declarations=parsed
    )
    rendered = Path(rust_path).read_text(encoding="utf-8")
    golden_path = (
        Path(__file__).with_name("data") / "c_device_round1.expected.rs"
    )
    assert rendered == golden_path.read_text(encoding="utf-8")
    assert "shim" not in rendered.lower()

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["target"] == {"abi": "nvptx64-c", "gpu_arch": "sm_90"}
    assert "cuda_toolkit_version" in manifest["generator"]
    assert manifest["generator"]["generation_command"] is None
    assert manifest["artifacts"]["ltoir_inputs"] == ["libdevice_api.ltoir"]
    assert [item["native_name"] for item in manifest["symbols"]] == [
        "library_copy"
    ]
    assert manifest["symbols"][0]["public_name"] == "copy"
    assert manifest["symbols"][0]["rust_name"] == "copy"
    assert manifest["symbols"][0]["parameters"][0]["rust_type"] == (
        "*const f32"
    )
    assert {item["reason"] for item in manifest["excluded_declarations"]} == {
        "execution-space:host",
        "round-1-c-api-only",
    }


@pytest.mark.parametrize(
    ("bad_function", "message"),
    [
        (function("cpp", "int", c_linkage=False), "does not have C linkage"),
        (function("variadic", "int", variadic=True), "variadic"),
        (function("unknown", "mystery_t"), "unsupported C ABI type"),
        (
            function("vector", "void", (("value", "double2"),)),
            "only supports it behind a pointer",
        ),
    ],
)
def test_strictly_rejects_out_of_contract_device_declarations(
    config, bad_function, message
):
    with pytest.raises(BindingModelError, match=message):
        build_c_device_model(declarations(functions=[bad_function]), config)


def test_requires_new_ast_canopy_linkage_metadata(config):
    candidate = function("old_ast", "int")
    del candidate.is_c_linkage
    with pytest.raises(BindingModelError, match="Function.is_c_linkage"):
        build_c_device_model(declarations(functions=[candidate]), config)


def test_symbol_inventory_reports_missing_symbols(config, tmp_path):
    model = build_c_device_model(
        declarations(
            functions=[
                function("library_one", "int"),
                function("library_two", "int"),
            ]
        ),
        config,
    )
    inventory = tmp_path / "symbols.txt"
    inventory.write_text("000 T library_one\n", encoding="utf-8")
    with pytest.raises(BindingModelError, match="library_two"):
        verify_symbol_inventory(model, str(inventory))


def test_symbol_inventory_requires_exact_selected_surface(config, tmp_path):
    model = build_c_device_model(
        declarations(functions=[function("library_one", "int")]), config
    )
    inventory = tmp_path / "symbols.txt"
    inventory.write_text("library_one\nunselected_helper\n", encoding="utf-8")
    with pytest.raises(BindingModelError, match="ungenerated"):
        verify_symbol_inventory(model, str(inventory))

    inventory.write_text("library_one\n", encoding="utf-8")
    assert (
        verify_symbol_inventory(model, str(inventory))["status"] == "verified"
    )


def test_symbol_inventory_rejects_malformed_entries(config, tmp_path):
    model = build_c_device_model(
        declarations(functions=[function("library_one", "int")]), config
    )
    inventory = tmp_path / "symbols.txt"
    inventory.write_text("library_one\nnot-a-c-symbol\n", encoding="utf-8")
    with pytest.raises(BindingModelError, match="line 2"):
        verify_symbol_inventory(model, str(inventory))


def test_supplemental_type_alias_cycles_are_rejected(config):
    config.type_aliases = {"alias_a": "alias_b", "alias_b": "alias_a"}
    model = build_c_device_model(declarations(), config)
    with pytest.raises(BindingModelError, match="cyclic type aliases"):
        render_cuda_oxide_bindings(model, config)


def test_constants_cannot_collide_after_rust_name_mapping(config):
    enum = SimpleNamespace(
        name="",
        underlying_type=FakeType("int"),
        enumerators=["self"],
        enumerator_values=[0],
    )
    config.constants = {"self_": {"Type": "int", "Value": 1}}
    model = build_c_device_model(declarations(enums=[enum]), config)
    with pytest.raises(BindingModelError, match="conflicts with parsed"):
        render_cuda_oxide_bindings(model, config)


def test_supplemental_cuda_alias_adds_its_storage_definition(config):
    config.type_aliases = {**config.type_aliases, "library_half_t": "__half"}
    model = build_c_device_model(declarations(), config)
    rendered, _ = render_cuda_oxide_bindings(model, config)
    assert "pub type __half = u16;" in rendered
    assert "pub type library_half_t = __half;" in rendered


def test_identity_record_typedef_is_emitted_once(config):
    record = SimpleNamespace(
        name="record_t",
        sizeof_=4,
        alignof_=4,
        fields=[SimpleNamespace(name="value", type_=FakeType("int"))],
    )
    typedef = SimpleNamespace(
        name="record_t", underlying_name="struct record_t"
    )
    model = build_c_device_model(
        declarations(
            functions=[
                function("library_record", "void", (("record", "record_t *"),))
            ],
            structs=[record],
            typedefs=[typedef],
        ),
        config,
    )
    assert [item.name for item in model.records] == ["record_t"]
    assert model.typedefs == []
    rendered, _ = render_cuda_oxide_bindings(model, config)
    assert rendered.count("pub type record_t =") == 1


def test_public_alias_cannot_shadow_another_native_symbol(config):
    with pytest.raises(BindingModelError, match="conflicts with native symbol"):
        build_c_device_model(
            declarations(
                functions=[
                    function("library_foo", "int"),
                    function("library_library_foo", "int"),
                ]
            ),
            config,
        )


def test_rust_keyword_function_names_use_raw_identifiers(config):
    model = build_c_device_model(
        declarations(
            functions=[
                function("match", "int"),
                function("union", "int"),
                function("library_type", "int"),
            ]
        ),
        config,
    )
    rendered, _ = render_cuda_oxide_bindings(model, config)

    assert "pub fn r#match() -> i32;" in rendered
    assert "pub fn r#union() -> i32;" in rendered
    assert "pub use self::library_type as r#type;" in rendered

    with pytest.raises(BindingModelError, match="exact CUDA-Oxide"):
        build_c_device_model(
            declarations(functions=[function("self", "int")]), config
        )


def test_cuda_arch_macro_is_derived_and_mismatch_is_rejected(config):
    assert config.parser_defines == ["__CUDA_ARCH__=900"]
    raw = dict(config.raw_config)
    raw["Predefined Macros"] = ["__CUDA_ARCH__=800"]
    with pytest.raises(ValueError, match="does not match"):
        CudaOxideConfig(raw)

    raw = copy.deepcopy(config.raw_config)
    raw["GPU Arch"] = ["sm_90a"]
    assert CudaOxideConfig(raw).parser_defines == ["__CUDA_ARCH__=900"]


def test_unknown_cuda_oxide_configuration_is_rejected(config):
    raw = copy.deepcopy(config.raw_config)
    raw["CUDA Oxide"]["CUDA Header Mode"] = True
    with pytest.raises(ValueError, match="CUDA Header Mode"):
        CudaOxideConfig(raw)


def test_round_one_rejects_argument_intents(config):
    raw = copy.deepcopy(config.raw_config)
    raw["Function Argument Intents"] = {"library_copy": {"output": "out"}}
    with pytest.raises(ValueError, match="Round 2"):
        CudaOxideConfig(raw)


def test_rust_output_and_manifest_names_must_differ(config):
    raw = copy.deepcopy(config.raw_config)
    raw["CUDA Oxide"]["Manifest Name"] = raw["Output Name"]
    with pytest.raises(ValueError, match="must differ"):
        CudaOxideConfig(raw)


def test_real_ast_canopy_c_device_parse(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    header = repo_root / "examples" / "cuda_oxide_c_device" / "device_api.h"
    parsed_config = CudaOxideConfig(
        {
            "Entry Point": str(header),
            "File List": [str(header)],
            "GPU Arch": ["sm_80"],
            "Backend": "cuda-oxide",
            "Output Name": "bindings.rs",
            "CUDA Oxide": {
                "LTOIR Inputs": ["device_api.ltoir"],
                "Bypass Parse Errors": True,
            },
        }
    )
    rust_path, manifest_path = generate_cuda_oxide_bindings(
        parsed_config, tmp_path
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert [symbol["native_name"] for symbol in manifest["symbols"]] == [
        "round1_accumulate",
        "round1_add",
        "round1_apply",
    ]
    assert 'unsafe extern "C"' in Path(rust_path).read_text(encoding="utf-8")


def test_lightweight_generator_import_does_not_import_numba():
    source_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import numbast.tools.cuda_oxide_binding_generator; "
                "assert 'numba' not in sys.modules"
            ),
        ],
        check=True,
        env=environment,
    )
