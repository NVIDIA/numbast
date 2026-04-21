# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that ast_canopy handles complex template hierarchies without crashing.

Each test targets a specific crash root cause that was found when parsing
deeply nested C++ template code (CRTP hierarchies, dependent types,
template template parameters, etc.).

Root causes fixed:
1. function.cpp: mangleName() segfaults on dependent/uninstantiated template types
2. type.cpp: getCanonicalType() on dependent types crashes
3. record.cpp: getTypeSize()/getTypeAlign() on dependent/incomplete types crashes
4. typedef.cpp: record_id_to_name->at() throws for template instantiations not in map
5. template_param.cpp: TemplateTemplateParmDecl was unhandled
6. class_template_specialization.cpp: unsupported template argument kinds caused throws
7. All matchers: no try-catch — one bad declaration crashed the entire parse
"""

import os
import pytest
from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def source_path():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        current_directory, "data", "sample_complex_templates.cu"
    )


class TestCRTPDependentTypes:
    """Case 1 & 5: CRTP hierarchies with dependent method types.

    The Itanium name mangler segfaults when called on methods with
    dependent (uninstantiated) parameter or return types inside class
    templates. The fix skips mangling for dependent signatures.
    """

    def test_parse_crtp_base_does_not_crash(self, source_path):
        """Parsing a header with CRTP base classes should not segfault."""
        decls = parse_declarations_from_source(
            source_path,
            [source_path],
            "sm_80",
            bypass_parse_error=True,
        )
        assert decls is not None

    def test_crtp_class_templates_found(self, source_path):
        """CRTP base templates should be reported as class templates."""
        decls = parse_declarations_from_source(
            source_path,
            [source_path],
            "sm_80",
            bypass_parse_error=True,
        )
        ct_names = [ct.qual_name for ct in decls.class_templates]
        assert any("CRTPBase" in n for n in ct_names)
        assert any("Vec3" in n for n in ct_names)

    def test_deep_crtp_hierarchy(self, source_path):
        """Multi-level CRTP (Level1 -> Level2 -> Concrete) should parse."""
        decls = parse_declarations_from_source(
            source_path,
            [source_path],
            "sm_80",
            bypass_parse_error=True,
        )
        ct_names = [ct.qual_name for ct in decls.class_templates]
        assert any("Level1" in n for n in ct_names)
        assert any("Level2" in n for n in ct_names)
        assert any("Concrete" in n for n in ct_names)

    def test_concrete_function_still_parsed(self, source_path):
        """Non-template functions using CRTP types should still parse."""
        decls = parse_declarations_from_source(
            source_path,
            [source_path],
            "sm_80",
            bypass_parse_error=True,
        )
        func_names = [f.name for f in decls.functions]
        assert "vec3_dot" in func_names


class TestTypedefTemplateInstantiation:
    """Case 2: Typedefs to class template instantiations.

    record_id_to_name->at() threw when the underlying record was a
    template instantiation not in the map. Fixed with safe find() lookup.
    """

    def test_typedef_to_template_instantiation(self, source_path):
        """Typedefs like 'using Vec3fStorage = Storage<float, 3>' should parse."""
        decls = parse_declarations_from_source(
            source_path,
            [source_path],
            "sm_80",
            bypass_parse_error=True,
        )
        td_names = [td.name for td in decls.typedefs]
        assert "Vec3fStorage" in td_names
        assert "Vec4dStorage" in td_names


class TestTemplateTemplateParameter:
    """Case 3: Template template parameters.

    TemplateTemplateParmDecl was unhandled and threw. Now handled with
    kind=template_ and the parameter name is preserved.
    """

    def test_template_template_param_does_not_crash(self, source_path):
        """A class template with a template template parameter should parse."""
        decls = parse_declarations_from_source(
            source_path,
            [source_path],
            "sm_80",
            bypass_parse_error=True,
        )
        ct_names = [ct.qual_name for ct in decls.class_templates]
        assert any("Adapter" in n for n in ct_names)


class TestNonTypeTemplateExpressions:
    """Case 6: Non-type template parameters with default expressions.

    Template args that are expressions (e.g., `(N > 4)`) could cause
    throws in class_template_specialization.cpp. Fixed with fallback
    to '<unsupported>' placeholder.
    """

    def test_expression_template_args_do_not_crash(self, source_path):
        """Template specializations with expression args should parse."""
        decls = parse_declarations_from_source(
            source_path,
            [source_path],
            "sm_80",
            bypass_parse_error=True,
        )
        # TestInst forces instantiation of AlignedStorage with expression args
        struct_names = [s.name for s in decls.structs]
        assert "TestInst" in struct_names
