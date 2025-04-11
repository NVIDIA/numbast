# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
import typing

import pylibastcanopy as bindings

CXX_OP_TO_PYTHON_OP = {
    "+": [operator.pos, operator.add],
    "-": [operator.neg, operator.sub],
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
    "!": operator.not_,
    "~": operator.inv,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "+=": operator.iadd,
    "-=": operator.isub,
    "*=": operator.imul,
    "/=": operator.itruediv,
    "%=": operator.imod,
    "&=": operator.iand,
    "|=": operator.ior,
    "^=": operator.ixor,
    "<<=": operator.ilshift,
    ">>=": operator.irshift,
    "=": None,  # assignment in python can be indexed assignment or sliced assignment, but copy assignment doesn't exist.
}


class Function:
    """
    Represents a C++ function.

    For C++ operators types:
    https://en.cppreference.com/w/cpp/language/operators.
    """

    def __init__(
        self,
        name: str,
        return_type: bindings.Type,
        params: list[bindings.ParamVar],
        exec_space: bindings.execution_space,
    ):
        self.name = name
        self.return_type = return_type
        self.params = params
        self.is_operator = self.name.startswith("operator")
        self._op_str = self.name[8:] if self.is_operator else None
        self.exec_space = exec_space

    def __str__(self):
        return f"{self.name}({', '.join(str(p) for p in self.params)}) -> {self.return_type}"

    def __repr__(self):
        old = super().__repr__()
        return f"{old[:-1]} {self.__str__()}>"

    @property
    def mangled_name(self):
        name = self.name
        if self.is_overloaded_operator():
            py_op = self.overloaded_operator_to_python_operator
            name = "operator" + "_" + py_op.__name__
        name = name.replace(" ", "_")
        return name

    @property
    def param_types(self) -> list[bindings.Type]:
        return [a.type_ for a in self.params]

    def is_allocation_operator(self) -> bool:
        if self._op_str is None:
            return False
        return self._op_str.startswith("new")

    def is_deallocation_operator(self) -> bool:
        if self._op_str is None:
            return False
        return self._op_str.startswith("delete")

    def is_user_defined_literal(self) -> bool:
        if self._op_str is None:
            return False
        return self._op_str.startswith('""')

    def is_cowait_operator(self) -> bool:
        if self._op_str is None:
            return False
        return self._op_str.startswith("co_await")

    def is_overloaded_operator(self) -> bool:
        if self._op_str is None:
            return False
        return self._op_str in CXX_OP_TO_PYTHON_OP

    def is_copy_assignment_operator(self) -> bool:
        return self._op_str == "="

    def is_conversion_operator(self) -> bool:
        # This standalone function doesn't keep track of all types in C++
        # system, so we find out if it's a conversion operator by eliminating
        # other possibilities.
        return (
            self.is_operator
            and not self.is_overloaded_operator()
            and not self.is_allocation_operator()
            and not self.is_deallocation_operator()
            and not self.is_user_defined_literal()
            and not self.is_cowait_operator()
        )

    @property
    def overloaded_operator_to_python_operator(self):
        if not self.is_overloaded_operator():
            raise ValueError(f"{self.name} is not an overloaded operator")
        self._op_str = typing.cast(str, self._op_str)

        if self._op_str in ("+", "-"):
            # Note this is standalone function
            # T T::operator-(const T2 &lh, const T2 &rh) const;
            # T T::operator-(const T2 &x) const;
            return CXX_OP_TO_PYTHON_OP[self._op_str][len(self.params) - 1]
        else:
            return CXX_OP_TO_PYTHON_OP[self._op_str]

    @classmethod
    def from_c_obj(cls, c_obj: bindings.Function):
        return cls(
            c_obj.name, c_obj.return_type, c_obj.params, c_obj.exec_space
        )


class FunctionTemplate:
    def __init__(
        self,
        template_parameters: list[bindings.TemplateParam],
        function: Function,
        num_min_required_args: int,
    ):
        self.template_parameters = template_parameters
        self.function = function
        self.num_min_required_args = num_min_required_args

    @classmethod
    def from_c_obj(cls, c_obj: bindings.FunctionTemplate):
        return cls(
            c_obj.template_parameters,
            Function.from_c_obj(c_obj.function),
            c_obj.num_min_required_args,
        )


class StructMethod(Function):
    def __init__(
        self,
        name: str,
        return_type: bindings.Type,
        params: list[bindings.ParamVar],
        kind: bindings.method_kind,
        exec_space: bindings.execution_space,
        is_move_constructor: bool = False,
    ):
        super().__init__(name, return_type, params, exec_space)
        self.kind = kind
        self.is_move_constructor = is_move_constructor

    @property
    def overloaded_operator_to_python_operator(self):
        if not self.is_overloaded_operator():
            raise ValueError(f"{self.name} is not an overloaded operator")
        self._op_str = typing.cast(str, self._op_str)

        if self._op_str in ("-", "+"):
            # Note that this is inside class definition, not standalone
            # T T::operator-() const;
            # T T::operator-(const T2 &b) const;
            return CXX_OP_TO_PYTHON_OP[self._op_str][len(self.params)]
        else:
            return CXX_OP_TO_PYTHON_OP[self._op_str]

    @classmethod
    def from_c_obj(cls, c_obj: bindings.Method):
        return cls(
            c_obj.name,
            c_obj.return_type,
            c_obj.params,
            c_obj.kind,
            c_obj.exec_space,
            c_obj.is_move_constructor(),
        )


class TemplatedStructMethod(StructMethod):
    @property
    def decl_name(self):
        """For a templated struct method, if the name contains template parameters,
        Decl name is the name without the template parameters. e.g.

        template<typename T, int n>
        struct Foo { Foo() {} };

        The name of the constructor is Foo<T, n>. Decl name is Foo.
        """

        if "<" in self.name:
            return self.name.split("<")[0]
        else:
            return self.name


class Struct:
    def __init__(
        self,
        name: str,
        fields: list[bindings.Field],
        methods: list[StructMethod],
        templated_methods: list[FunctionTemplate],
        nested_records: list[bindings.Record],
        nested_class_templates: list[bindings.ClassTemplate],
        sizeof_: int,
        alignof_: int,
    ):
        self.name = name
        self.fields = fields
        self.methods = methods
        self.templated_methods = templated_methods
        self.nested_records = nested_records
        self.nested_class_templates = nested_class_templates
        self.sizeof_ = sizeof_
        self.alignof_ = alignof_

    def constructors(self):
        for m in self.methods:
            if m.name == self.name:
                yield m

    def overloaded_operators(self):
        for m in self.methods:
            if m.is_overloaded_operator():
                yield m

    def conversion_operators(self):
        for m in self.methods:
            if m.is_conversion_operator():
                yield m

    @classmethod
    def from_c_obj(cls, c_obj: bindings.Record):
        methods = [StructMethod.from_c_obj(m) for m in c_obj.methods]
        return cls(
            c_obj.name,
            c_obj.fields,
            methods,
            [FunctionTemplate.from_c_obj(tm) for tm in c_obj.templated_methods],
            c_obj.nested_records,
            c_obj.nested_class_templates,
            c_obj.sizeof_,
            c_obj.alignof_,
        )


class TemplatedStruct(Struct):
    def __init__(
        self,
        name: str,
        fields: list[bindings.Field],
        methods: list[TemplatedStructMethod],
        templated_methods: list[FunctionTemplate],
        nested_records: list[bindings.Record],
        nested_class_templates: list[bindings.ClassTemplate],
        sizeof_: int,
        alignof_: int,
    ):
        super().__init__(
            name,
            fields,
            [],
            templated_methods,
            nested_records,
            nested_class_templates,
            sizeof_,
            alignof_,
        )
        self.methods = methods

    @classmethod
    def from_c_obj(cls, c_obj: bindings.Record):
        methods = [TemplatedStructMethod.from_c_obj(m) for m in c_obj.methods]

        return cls(
            c_obj.name,
            c_obj.fields,
            methods,
            [FunctionTemplate.from_c_obj(tm) for tm in c_obj.templated_methods],
            c_obj.nested_records,
            c_obj.nested_class_templates,
            c_obj.sizeof_,
            c_obj.alignof_,
        )


class ClassTemplate:
    def __init__(
        self,
        record: bindings.Record,
        template_parameters: list[bindings.TemplateParam],
        num_min_required_args: int,
    ):
        self.record = TemplatedStruct.from_c_obj(record)
        self.template_parameters = template_parameters
        self.num_min_required_args = num_min_required_args

    @classmethod
    def from_c_obj(cls, c_obj: bindings.ClassTemplate):
        return cls(
            c_obj.record, c_obj.template_parameters, c_obj.num_min_required_args
        )
