from numbast.static.renderer import BaseRenderer, get_rendered_imports


class StaticClassTemplateRenderer(BaseRenderer):
    _python_api_rendered: str

    _python_api_template = """
def {class_template_name}():
    pass
"""

    def __init__(self, decl):
        self.decl = decl

    def _render_python_api(self):
        self._python_api_rendered = self._python_api_template.format(
            class_template_name=self.decl.name
        )

    def render(self):
        self._render_python_api()


class StaticClassTemplatesRenderer(BaseRenderer):
    def __init__(self, decls):
        self.decls = decls

    def _render(self, with_imports):
        self._python_rendered = []

        for decl in self.decls:
            SCTR = StaticClassTemplateRenderer(decl)
            SCTR.render()
            self._python_rendered.append(SCTR._python_api_rendered)

        self._python_str = ""

        if with_imports:
            self._python_str += "\n" + get_rendered_imports()

        self._python_str += "\n" + "\n".join(self._python_rendered)

    def render_as_str(self, with_imports: bool, with_shim_stream: bool) -> str:
        self._render(with_imports)
        return self._python_str
