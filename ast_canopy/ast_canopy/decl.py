# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
import typing
from functools import cached_property

from ast_canopy import pylibastcanopy as bindings

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


class Function:
    """Represents a C++ function.

    For C++ operator types, see
    https://en.cppreference.com/w/cpp/language/operators.
    """

    def __init__(
        self,
        name: str,
        return_type: bindings.Type,
        params: list[bindings.ParamVar],
        exec_space: bindings.execution_space,
        is_constexpr: bool,
        mangled_name: str,
        attributes: str,
        parse_entry_point: str,
    ):
        self.name = name
        self.return_type = return_type
        self.params = params
        self.is_operator = self.name.startswith("operator")
        self._op_str = self.name[8:] if self.is_operator else None
        self.exec_space = exec_space
        self.is_constexpr = is_constexpr
        self.mangled_name = mangled_name
        self.attributes = attributes

        self.parse_entry_point = parse_entry_point

    def __str__(self):
        return f"{self.name}({', '.join(str(p) for p in self.params)}) -> {self.return_type}"

    def __repr__(self):
        old = super().__repr__()
        return f"{old[:-1]} {self.__str__()}>"

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
            c_obj.mangled_name,
            c_obj.attributes,
            parse_entry_point,
        )


class Template:
    """Base class for C++ template declarations.

    Stores the list of template parameters and the minimum number of
    required template arguments.
    """

    def __init__(
        self,
        template_parameters: list[bindings.TemplateParam],
        num_min_required_args: int,
    ):
        self.template_parameters = template_parameters
        self.num_min_required_args = num_min_required_args

    @cached_property
    def tparam_dict(self):
        res = {}
        for tparam in self.template_parameters:
            res[tparam.name] = tparam
        return res


class FunctionTemplate(Template):
    """Represents a C++ function template declaration.

    Wraps a parsed function template and provides ``instantiate`` for
    building a concrete instantiation.
    """

    def __init__(
        self,
        template_parameters: list[bindings.TemplateParam],
        num_min_required_args: int,
        function: Function,
        parse_entry_point: str,
    ):
        super().__init__(template_parameters, num_min_required_args)
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
        )

    def instantiate(self, **kwargs):
        tfunc = FunctionInstantiation(self)
        return tfunc.instantiate(**kwargs)


class StructMethod(Function):
    """Represents a method of a C++ struct/class.

    Includes constructors, conversion operators, and overloaded operators.
    Extends ``Function`` with method-specific metadata.
    """

    def __init__(
        self,
        name: str,
        return_type: bindings.Type,
        params: list[bindings.ParamVar],
        kind: bindings.method_kind,
        exec_space: bindings.execution_space,
        is_constexpr: bool,
        is_move_constructor: bool,
        mangled_name: str,
        attributes: str,
        parse_entry_point: str,
    ):
        super().__init__(
            name,
            return_type,
            params,
            exec_space,
            is_constexpr,
            mangled_name,
            attributes,
            parse_entry_point,
        )
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
    def from_c_obj(cls, c_obj: bindings.Method, parse_entry_point: str):
        return cls(
            c_obj.name,
            c_obj.return_type,
            c_obj.params,
            c_obj.kind,
            c_obj.exec_space,
            c_obj.is_constexpr,
            c_obj.is_move_constructor(),
            c_obj.mangled_name,
            c_obj.attributes,
            parse_entry_point,
        )


class TemplatedStructMethod(StructMethod):
    """Struct/class method whose name may include template parameters.

    Provides utilities for working with the declaration name without
    template arguments.
    """

    @property
    def decl_name(self):
        """Return the declaration name without template parameters.

        For templated struct methods, if the name contains template parameters,
        the declaration name is the base name without parameters.

        Example:

        .. code-block:: cpp

            template<typename T, int n>
            struct Foo { Foo() {} };

        The constructor name is ``Foo<T, n>``; the declaration name is ``Foo``.
        """

        if "<" in self.name:
            return self.name.split("<")[0]
        else:
            return self.name


class Struct:
    """Represents a C++ record (struct/class) and its metadata.

    Contains fields, methods, templated methods, nested records and class
    templates, as well as size/align information and the parse entry point.
    """

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
    ):
        self._name = name
        self.fields = fields
        self.methods = methods
        self.templated_methods = templated_methods
        self.nested_records = nested_records
        self.nested_class_templates = nested_class_templates
        self.sizeof_ = sizeof_
        self.alignof_ = alignof_

        self.parse_entry_point = parse_entry_point

    def public_fields(self) -> list[bindings.Field]:
        return [
            f for f in self.fields if f.access == bindings.access_kind.public_
        ]

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

    def regular_member_functions(self):
        """Generator for methods that are not constructors, overload operators and conversion operators."""
        for m in self.methods:
            if (
                m.name != self.name
                and (not m.is_overloaded_operator())
                and (not m.is_conversion_operator())
            ):
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
        )

    @property
    def name(self):
        return self._name


class TemplatedStruct(Struct):
    """A ``Struct`` whose methods include templated methods.

    Specializes method handling to use ``TemplatedStructMethod``.
    """

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
        )


class ClassTemplate(Template):
    """Represents a C++ class template declaration.

    Holds the underlying ``TemplatedStruct`` and provides ``instantiate`` for
    building a concrete class instantiation.
    """

    def __init__(
        self,
        record: TemplatedStruct,
        template_parameters: list[bindings.TemplateParam],
        num_min_required_args: int,
        parse_entry_point: str,
    ):
        super().__init__(template_parameters, num_min_required_args)
        self.record = record

        self.parse_entry_point = parse_entry_point

    @classmethod
    def from_c_obj(cls, c_obj: bindings.ClassTemplate, parse_entry_point: str):
        return cls(
            TemplatedStruct.from_c_obj(c_obj.record, parse_entry_point),
            c_obj.template_parameters,
            c_obj.num_min_required_args,
            parse_entry_point,
        )

    def instantiate(self, **kwargs):
        tstruct = ClassInstantiation(self)
        return tstruct.instantiate(**kwargs)


class ConstExprVar:
    """Represents a constexpr variable extracted from C++.

    Stores the C++ type and serialized value; ``value`` converts it to the
    corresponding Python value based on the type mapping.
    """

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


class ClassTemplateSpecialization(Struct, ClassInstantiation):
    """Represents a C++ class template specialization declaration.

    Holds the underlying ``TemplatedStruct`` and provides ``instantiate`` for
    building a concrete class instantiation.
    """

    def __init__(
        self,
        record: Struct,
        class_template: ClassTemplate,
        actual_template_arguments: list[str],
    ):
        Struct.__init__(
            self,
            record.name,
            record.fields,
            record.methods,
            record.templated_methods,
            record.nested_records,
            record.nested_class_templates,
            record.sizeof_,
            record.alignof_,
            record.parse_entry_point,
        )
        ClassInstantiation.__init__(self, class_template)

        targ_names: list[str] = [
            tp.name for tp in class_template.template_parameters
        ]
        targ_values: list[str] = actual_template_arguments

        kwargs = dict(zip(targ_names, targ_values))

        self.instantiate(**kwargs)

        self.actual_template_arguments = actual_template_arguments

    @classmethod
    def from_c_obj(
        cls, c_obj: bindings.ClassTemplateSpecialization, parse_entry_point: str
    ):
        return cls(
            Struct.from_c_obj(c_obj, parse_entry_point),
            ClassTemplate.from_c_obj(c_obj.class_template, parse_entry_point),
            c_obj.actual_template_arguments,
        )

    @property
    def name(self):
        return self.get_instantiated_c_stmt()

    @property
    def base_name(self):
        return self.record.name

    def constructors(self):
        for m in self.methods:
            if m.name == self.base_name:
                yield m
