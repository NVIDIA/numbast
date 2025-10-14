from __future__ import annotations
import os

import ast_canopy


# TODO: POC only, ideally this logic should be implemented in libastcanopy
# The python binding provides python type - CXX type conversion
class BaseInstantiation:
    """Represent an instantiation of a template."""

    def __init__(self, template: ast_canopy.decl.Template):
        self.template = template
        self.template_parameters = template.template_parameters
        self._template_param_names = [
            tparam.name for tparam in self.template_parameters
        ]

    def instantiate(self, **kwargs):
        self.instantiated_args = kwargs
        self.validate()
        return self

    def validate(self):
        if any(
            key not in self._template_param_names
            for key in self.instantiated_args
        ):
            raise ValueError(
                f"Invalid template parameter names: {self.instantiated_args.keys()}, allowed template parameter names: {self._template_param_names}"
            )

    @property
    def param_list(self):
        return [
            self.instantiated_args[tparam.name]
            for tparam in self.template_parameters
        ]

    def get_instantiated_c_stmt(self) -> str:
        name = self.base_name
        param_list = self.param_list

        flatten = []
        for param in param_list:
            if isinstance(param, BaseInstantiation):
                flatten.append(param.get_instantiated_c_stmt())
            else:
                flatten.append(str(param))

        param_list = ", ".join(flatten)
        return f"{name}<{param_list}>"

    def base_name(self):
        raise NotImplementedError(
            "BaseInstantiation.base_name is not implemented"
        )


class FunctionInstantiation(BaseInstantiation):
    """Represent an instantiation of a function template."""

    def __init__(self, function_template: ast_canopy.decl.FunctionTemplate):
        super().__init__(function_template)
        self.function = function_template.function

    @property
    def base_name(self):
        return self.function.name

    def evaluate_constexpr_value(self, *args, header=None):
        if not self.function.is_constexpr:
            raise ValueError("Function is not constexpr")

        if header is None:
            header = self.function.parse_entry_point

        if not os.path.exists(header):
            raise ValueError(f"{header} does not exist.")

        header = os.path.abspath(header)

        assembled_code_template = """
#include <{header}>
{argument_decls}
__device__ constexpr auto ast_canopy_var_value__ = {tfunc_instantiation}
"""

        # Construct default-initialized arguments.
        argument_decls = ""
        for i, arg in enumerate(args):
            argument_decls += (
                f"__device__ {arg.get_instantiated_c_stmt()} arg_{i};\n"
            )

        fml_arglist = ",".join([f"arg_{i}" for i in range(len(args))])
        tfunc_instantiation = (
            self.get_instantiated_c_stmt() + f"({fml_arglist});"
        )

        assembled_code = assembled_code_template.format(
            header=header,
            argument_decls=argument_decls,
            tfunc_instantiation=tfunc_instantiation,
        )

        res = ast_canopy.value_from_constexpr_vardecl(
            assembled_code,
            "ast_canopy_var_value__",
            "sm_80",
            verbose=True,
        )

        return res.value


class ClassInstantiation(BaseInstantiation):
    """Represent an instantiation of a class template."""

    def __init__(self, class_template: ast_canopy.decl.ClassTemplate):
        super().__init__(class_template)
        self.record = class_template.record

    @property
    def base_name(self):
        return self.record.name
