# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
import typing

import pylibastcanopy as bindings

from ast_canopy.instantiations import FunctionInstantiation, ClassInstantiation

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

CXX_TYPE_TO_PYTHON_TYPE = {
    "int": int,
    "unsigned int": int,
    "long": int,
    "unsigned long": int,
    "long long": int,
    "unsigned long long": int,
    "int16_t": int,
    "int32_t": int,
    "int64_t": int,
    "uint16_t": int,
    "uint32_t": int,
    "uint64_t": int,
    "float": float,
    "double": float,
}


class Declaration:
    """Base class for all declarations that can appear in a namespace."""

    namespace_stack: list[str]
    # Stores nested namespace names in bottom-up order (deepest to top-level)
    # For example, for a declaration in "outer::middle::inner", the vector
    # would contain ["inner", "middle", "outer"]

    def __init__(self, namespace_stack: list[str]):
        self.namespace_stack = namespace_stack

    @property
    def concatenated_namespace(
        self,
        delimiter: str = "_",
        anonymous_namespace_placeholder: str = "anonymous",
    ) -> str:
        return delimiter.join(
            [
                x if x else anonymous_namespace_placeholder
                for x in reversed(self.namespace_stack)
            ]
        )


class Function(Declaration):
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
        is_constexpr: bool,
        parse_entry_point: str,
        namespace_stack: list[str],
    ):
        super().__init__(namespace_stack)
        self.name = name
        self.return_type = return_type
        self.params = params
        self.is_operator = self.name.startswith("operator")
        self._op_str = self.name[8:] if self.is_operator else None
        self.exec_space = exec_space
        self.is_constexpr = is_constexpr

        self.parse_entry_point = parse_entry_point

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
    def fully_qualified_name(self):
        namespace_prefix = (
            f"{self.concatenated_namespace}_"
            if self.concatenated_namespace
            else ""
        )
        return f"{namespace_prefix}{self.mangled_name}"

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
    def from_c_obj(cls, c_obj: bindings.Function, parse_entry_point: str):
        return cls(
            c_obj.name,
            c_obj.return_type,
            c_obj.params,
            c_obj.exec_space,
            c_obj.is_constexpr,
            parse_entry_point,
            c_obj.namespace_stack,
        )


class Template:
    def __init__(self, template_parameters, num_min_required_args):
        self.template_parameters = template_parameters
        self.num_min_required_args = num_min_required_args


class FunctionTemplate(Template, Declaration):
    def __init__(
        self,
        template_parameters: list[bindings.TemplateParam],
        num_min_required_args: int,
        function: Function,
        parse_entry_point: str,
        namespace_stack: list[str],
    ):
        Template.__init__(self, template_parameters, num_min_required_args)
        Declaration.__init__(self, namespace_stack)
        self.function = function

        self.parse_entry_point = parse_entry_point

    @classmethod
    def from_c_obj(
        cls, c_obj: bindings.FunctionTemplate, parse_entry_point: str
    ):
        return cls(
            c_obj.template_parameters,
            c_obj.num_min_required_args,
            Function.from_c_obj(c_obj.function, parse_entry_point),
            parse_entry_point,
            c_obj.namespace_stack,
        )

    def instantiate(self, **kwargs):
        tfunc = FunctionInstantiation(self)
        return tfunc.instantiate(**kwargs)


class StructMethod(Function):
    def __init__(
        self,
        name: str,
        return_type: bindings.Type,
        params: list[bindings.ParamVar],
        kind: bindings.method_kind,
        exec_space: bindings.execution_space,
        is_constexpr: bool,
        is_move_constructor: bool,
        parse_entry_point: str,
        namespace_stack: list[str],
        parent_name_prefix: str,
    ):
        super().__init__(
            name,
            return_type,
            params,
            exec_space,
            is_constexpr,
            parse_entry_point,
            namespace_stack,
        )
        self.kind = kind
        self.is_move_constructor = is_move_constructor
        self.parent_name_prefix = parent_name_prefix

    @property
    def fully_qualified_name(self):
        namespace_prefix = (
            f"{self.concatenated_namespace}_"
            if self.concatenated_namespace
            else ""
        )
        parent_name_all_level = (
            f"{self.parent_name_prefix}_" if self.parent_name_prefix else ""
        )
        print(
            f"namespace_prefix: {namespace_prefix}, parent_name_all_level: {parent_name_all_level}, mangled_name: {self.mangled_name}"
        )
        return f"{namespace_prefix}{parent_name_all_level}{self.mangled_name}"

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
    def from_c_obj(cls, c_obj: bindings.Method, parse_entry_point: str):
        return cls(
            c_obj.name,
            c_obj.return_type,
            c_obj.params,
            c_obj.kind,
            c_obj.exec_space,
            c_obj.is_constexpr,
            c_obj.is_move_constructor(),
            parse_entry_point,
            c_obj.namespace_stack,
            c_obj.parent_name_prefix(),
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


class Struct(Declaration):
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
        parse_entry_point: str,
        namespace_stack: list[str],
    ):
        super().__init__(namespace_stack)
        self.name = name
        self.fields = fields
        self.methods = methods
        self.templated_methods = templated_methods
        self.nested_records = nested_records
        self.nested_class_templates = nested_class_templates
        self.sizeof_ = sizeof_
        self.alignof_ = alignof_

        self.parse_entry_point = parse_entry_point

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
    def from_c_obj(cls, c_obj: bindings.Record, parse_entry_point: str):
        return cls(
            c_obj.name,
            c_obj.fields,
            [
                StructMethod.from_c_obj(m, parse_entry_point)
                for m in c_obj.methods
            ],
            [
                FunctionTemplate.from_c_obj(tm, parse_entry_point)
                for tm in c_obj.templated_methods
            ],
            c_obj.nested_records,
            c_obj.nested_class_templates,
            c_obj.sizeof_,
            c_obj.alignof_,
            parse_entry_point,
            c_obj.namespace_stack,
        )


class TemplatedStruct(Struct):
    templated_methods: list[TemplatedStructMethod]

    @classmethod
    def from_c_obj(cls, c_obj: bindings.Record, parse_entry_point: str):
        return cls(
            c_obj.name,
            c_obj.fields,
            [
                TemplatedStructMethod.from_c_obj(m, parse_entry_point)
                for m in c_obj.methods
            ],
            [
                FunctionTemplate.from_c_obj(tm, parse_entry_point)
                for tm in c_obj.templated_methods
            ],
            c_obj.nested_records,
            c_obj.nested_class_templates,
            c_obj.sizeof_,
            c_obj.alignof_,
            parse_entry_point,
            c_obj.namespace_stack,
        )


class ClassTemplate(Template, Declaration):
    def __init__(
        self,
        record: TemplatedStruct,
        template_parameters: list[bindings.TemplateParam],
        num_min_required_args: int,
        parse_entry_point: str,
        namespace_stack: list[str],
    ):
        Template.__init__(self, template_parameters, num_min_required_args)
        Declaration.__init__(self, namespace_stack)
        self.record = record

        self.parse_entry_point = parse_entry_point

    @classmethod
    def from_c_obj(cls, c_obj: bindings.ClassTemplate, parse_entry_point: str):
        return cls(
            TemplatedStruct.from_c_obj(c_obj.record, parse_entry_point),
            c_obj.template_parameters,
            c_obj.num_min_required_args,
            parse_entry_point,
            c_obj.namespace_stack,
        )

    def instantiate(self, **kwargs):
        tstruct = ClassInstantiation(self)
        return tstruct.instantiate(**kwargs)


class ConstExprVar:
    def __init__(self, name: str, type_: bindings.Type, value_serialized: str):
        self.name = name
        self.type_ = type_
        self.value_serialized = value_serialized

    @classmethod
    def from_c_obj(cls, c_obj: bindings.ConstExprVar):
        return cls(c_obj.name, c_obj.type_, c_obj.value)

    @property
    def value(self):
        cxx_type_name = self.type_.unqualified_non_ref_type_name
        return CXX_TYPE_TO_PYTHON_TYPE[cxx_type_name](self.value_serialized)
