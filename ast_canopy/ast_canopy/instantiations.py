from __future__ import annotations

import ast_canopy


# TODO: POC only, ideally this logic should be implemented in libastcanopy
# The python binding provides python type - CXX type conversion
class BaseInstantiation:
    """Indicates a instantiation of a template"""

    def __init__(self, template: "ast_canopy.decl.Template"):
        self.template_parameters = template.template_parameters
        self._template_param_names = [
            tparam.name for tparam in self.template_parameters
        ]

    def instantiate(self, **kwargs):
        self.instantiated_args = kwargs
        self.validate()
        return self

    def validate(self):
        if any(key not in self._template_param_names for key in self.instantiated_args):
            raise ValueError(
                f"Invalid template parameter names: {self.instantiated_args.keys()}, allowed template parameter names: {self._template_param_names}"
            )
        pass

    @property
    def param_list(self):
        return [
            self.instantiated_args[tparam.name] for tparam in self.template_parameters
        ]

    def get_instantiated_c_stmt(self) -> str:
        name = self.name
        param_list = self.param_list

        flatten = []
        for param in param_list:
            if isinstance(param, BaseInstantiation):
                flatten.append(param.get_instantiated_c_stmt())
            else:
                flatten.append(str(param))

        param_list = ", ".join(flatten)
        return f"{name}<{param_list}>"


class FunctionInstantiation(BaseInstantiation):
    """Indicates an instantiation of a function template"""

    def __init__(self, function_template: "ast_canopy.decl.FunctionTemplate"):
        super().__init__(function_template)
        self.function = function_template.function

    @property
    def name(self):
        return self.function.name


class ClassInstantiation(BaseInstantiation):
    """Indicates an instantiation of a class template"""

    def __init__(self, class_template: "ast_canopy.decl.ClassTemplate"):
        super().__init__(class_template)
        self.record = class_template.record

    @property
    def name(self):
        return self.record.name
