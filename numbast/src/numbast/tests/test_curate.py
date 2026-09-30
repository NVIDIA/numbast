# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the backend-neutral curation pass.

These assert the curation *rules* directly, rather than inferring them from
generated bindings. The rules are now shared by every backend, so a change in
any one of them changes the surface of all of them at once.
"""

import os

import pytest

from ast_canopy import parse_declarations_from_source

from numbast.curate import (
    function_skip_reason,
    matches_any_regex_pattern,
    plan_functions,
)

HEADER = os.path.join(os.path.dirname(__file__), "data", "curate_input.cuh")


@pytest.fixture(scope="module")
def decls():
    """Function declarations from the fixture header, keyed by name.

    A real parse rather than stub objects: curation reads ``exec_space``, and
    the point of the non-device rule is that it agrees with what ``ast_canopy``
    actually reports for an unannotated function.
    """
    # Curation is independent of compute capability, so this needs no device.
    parsed = parse_declarations_from_source(HEADER, [HEADER], "sm_80")
    return {f.name: f for f in parsed.functions}


def test_the_fixture_header_covers_both_execution_spaces(decls):
    """Guard against the fixture silently ceasing to test anything."""
    assert set(decls) == {
        "acmeCompute",
        "acmeExcluded",
        "acmeCooperativeReduce",
        "internalHelper",
        "acmeHostOnly",
        "internalHostOnly",
    }


def test_regex_patterns_are_unanchored():
    """``re.search``, not ``re.match``.

    Worth pinning down: a user writing ``Reduce`` in a config expects it to
    match ``acmeCooperativeReduce``, and a user writing ``^acme`` is relying on
    the anchor being theirs to add.
    """
    assert matches_any_regex_pattern("acmeCooperativeReduce", ["Reduce"])
    assert matches_any_regex_pattern("acmeCooperativeReduce", ["^acme"])
    assert not matches_any_regex_pattern("acmeCompute", ["^Compute"])


def test_no_patterns_matches_nothing():
    assert not matches_any_regex_pattern("acmeCompute", [])


def test_in_scope_declarations_have_no_skip_reason(decls):
    assert (
        function_skip_reason(
            decls["acmeCompute"],
            excludes=[],
            skip_prefix=None,
            skip_non_device=True,
        )
        is None
    )


@pytest.mark.parametrize(
    "name,excludes,skip_prefix,expected",
    [
        ("acmeExcluded", ["acmeExcluded"], None, "excluded"),
        ("internalHelper", [], "internal", "skip_prefix"),
        ("acmeHostOnly", [], None, "non_device"),
        # Excludes are checked before the prefix rule.
        ("internalHelper", ["internalHelper"], "internal", "excluded"),
        # The prefix rule is checked before the execution-space rule. This
        # ordering is user-visible: only "non_device" warns, and a function the
        # user explicitly skipped by prefix should not also be warned about.
        ("internalHostOnly", [], "internal", "skip_prefix"),
    ],
)
def test_skip_reasons_and_their_precedence(
    decls, name, excludes, skip_prefix, expected
):
    assert (
        function_skip_reason(
            decls[name],
            excludes=excludes,
            skip_prefix=skip_prefix,
            skip_non_device=True,
        )
        == expected
    )


def test_exclusion_is_by_exact_name_not_by_pattern(decls):
    """``Exclude Functions`` is a membership test, unlike the regex options."""
    assert (
        function_skip_reason(
            decls["acmeExcluded"],
            excludes=["Excluded"],
            skip_prefix=None,
            skip_non_device=True,
        )
        is None
    )


def test_host_functions_are_kept_when_the_rule_is_off(decls):
    reason = function_skip_reason(
        decls["acmeHostOnly"],
        excludes=[],
        skip_prefix=None,
        skip_non_device=False,
    )
    assert reason is None


def _plan_names(plans):
    return {p.decl.name for p in plans}


def test_planning_drops_every_out_of_scope_declaration(decls):
    with pytest.warns(UserWarning):
        plans = plan_functions(
            decls.values(),
            header_path=HEADER,
            excludes=["acmeExcluded"],
            skip_prefix="internal",
        )

    assert _plan_names(plans) == {"acmeCompute", "acmeCooperativeReduce"}


def test_planning_keeps_host_functions_when_asked(decls):
    plans = plan_functions(
        decls.values(), header_path=HEADER, skip_non_device=False
    )
    assert "acmeHostOnly" in _plan_names(plans)


def test_only_non_device_skips_warn(decls, recwarn):
    """The other two skips are what the user asked for, so they are silent."""
    plan_functions(
        [decls["acmeExcluded"], decls["internalHelper"]],
        header_path=HEADER,
        excludes=["acmeExcluded"],
        skip_prefix="internal",
    )
    assert [w for w in recwarn.list if "Skipping" in str(w.message)] == []


def test_the_non_device_warning_names_the_declaration_and_header(decls):
    with pytest.warns(UserWarning, match="acmeHostOnly") as record:
        plan_functions([decls["acmeHostOnly"]], header_path=HEADER)

    messages = [str(w.message) for w in record]
    assert any(HEADER in m for m in messages), messages


def test_prefix_removal_sets_the_exposed_name_and_keeps_the_c_symbol(decls):
    """Both names are needed, and they are not interchangeable.

    ``exposed_name`` is what callers type; ``c_symbol`` is what has to appear in
    the emitted call. A backend that conflated them would emit a call to a
    symbol that does not exist.
    """
    (plan,) = plan_functions(
        [decls["acmeCompute"]], header_path=HEADER, prefix_removal=["acme"]
    )
    assert plan.exposed_name == "Compute"
    assert plan.c_symbol == "acmeCompute"


def test_the_exposed_name_defaults_to_the_declared_name(decls):
    (plan,) = plan_functions([decls["acmeCompute"]], header_path=HEADER)
    assert plan.exposed_name == "acmeCompute" == plan.c_symbol


def test_cooperative_launch_is_resolved_per_declaration(decls):
    plans = plan_functions(
        [decls["acmeCompute"], decls["acmeCooperativeReduce"]],
        header_path=HEADER,
        cooperative_launch_required=["Cooperative"],
    )
    assert {p.decl.name: p.use_cooperative for p in plans} == {
        "acmeCompute": False,
        "acmeCooperativeReduce": True,
    }


def test_cooperative_launch_is_matched_on_the_declared_name(decls):
    """Not on the exposed name -- the config predates prefix removal.

    Pinning this down because the two names differ exactly when prefix removal
    is configured, and silently switching which one the regex sees would change
    behaviour for existing configs.
    """
    (plan,) = plan_functions(
        [decls["acmeCompute"]],
        header_path=HEADER,
        prefix_removal=["acme"],
        cooperative_launch_required=["^acmeCompute$"],
    )
    assert plan.use_cooperative


def test_argument_intents_reach_the_plan_whole(decls):
    """The renderers index the mapping by declaration name themselves.

    So curation passes the whole mapping through rather than the entry for this
    declaration; narrowing it here would break every renderer that looks up a
    method as ``"<Struct>.<method>"``.
    """
    intents = {"acmeCompute": {"x": "out"}, "somethingElse": {0: "in"}}
    (plan,) = plan_functions(
        [decls["acmeCompute"]], header_path=HEADER, argument_intents=intents
    )
    assert plan.argument_intents == intents


def test_plans_default_to_an_empty_intent_mapping(decls):
    (plan,) = plan_functions([decls["acmeCompute"]], header_path=HEADER)
    assert plan.argument_intents == {}


def test_planning_preserves_declaration_order(decls):
    """Generated output is diffed and cached, so ordering cannot be incidental."""
    ordered = [
        decls["acmeCooperativeReduce"],
        decls["acmeCompute"],
        decls["acmeExcluded"],
    ]
    plans = plan_functions(ordered, header_path=HEADER)
    assert [p.decl.name for p in plans] == [d.name for d in ordered]
