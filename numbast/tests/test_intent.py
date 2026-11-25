# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from numbast.intent import compute_intent_plan


class _ReferenceType:
    def is_left_reference(self):
        return True


def test_boolean_override_key_is_not_an_integer_parameter_index():
    params = [SimpleNamespace(name="src"), SimpleNamespace(name="dst")]
    param_types = [_ReferenceType(), _ReferenceType()]

    with pytest.raises(TypeError, match="arg_intent keys.*bool"):
        compute_intent_plan(
            params=params,
            param_types=param_types,
            overrides={True: "out_ptr"},
        )
