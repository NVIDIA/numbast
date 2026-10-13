# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The options an invocation passed, for the generated binding's provenance stamp."""

import click
from click.core import ParameterSource

__all__ = ["params_the_user_set"]


def _source_of(ctx: click.Context, name: str) -> ParameterSource | None:
    """Where ``name`` got its value, searching outward through parent contexts.

    ``Context.forward`` builds the callee's context by hand rather than by
    parsing, so it records no sources at all and ``get_parameter_source``
    answers ``None`` for every option. The dispatcher in :mod:`numbast.cli`
    forwards, and it is the only installed entry point, so consulting a single
    context would mean the real command line never reports a source. The
    context that did the parsing is up the parent chain, and it declares these
    options under the same names.
    """
    while ctx is not None:
        source = ctx.get_parameter_source(name)
        if source is not None:
            return source
        ctx = ctx.parent
    return None


def params_the_user_set(ctx: click.Context) -> dict:
    """The options this invocation actually passed, for the provenance stamp.

    Stamping ``ctx.params`` wholesale records the CLI's *signature* rather than
    the invocation: every option appears, including ones nobody passed, so
    adding an option rewrites the provenance comment in every generated file
    without any binding changing. That makes a regeneration impossible to
    review as a diff and byte-level comparison of generated output useless.

    Options left at their default also carry no information a reader does not
    already have -- the default is discoverable from ``--help`` -- while the
    ones that were passed are exactly what is needed to reproduce the run.

    An option whose source is unknown even after walking the parent chain is
    omitted, on the grounds that a value nobody can attribute to a command line
    is the same kind of noise as a default.
    """
    return {
        name: value
        for name, value in ctx.params.items()
        if _source_of(ctx, name) not in (None, ParameterSource.DEFAULT)
    }
