import click
import os
import json
from collections import defaultdict

import numba.types
import numba.core.datamodel.models

from ast_canopy import parse_declarations_from_source

from numbast.static.struct import StaticStructRenderer


class NumbaTypeDictType(click.ParamType):
    name = "numba_type_dict"

    def convert(self, value, param, ctx):
        try:
            d = json.loads(value)
        except Exception:
            self.fail(f"{self.name} parameter must be valid JSON string. Got {value}")

        try:
            keys = [*d.keys()]
            for k in keys:
                d[k] = getattr(numba.types, d[k])
        except Exception:
            self.fail(
                f"Unable to convert input type dictionary string into dict of numba types. Got {d}."
            )

        return d


numba_type_dict = NumbaTypeDictType()


class NumbaDataModelDictType(click.ParamType):
    name = "numba_datamodel_type"

    def convert(self, value, param, ctx):
        try:
            d = json.loads(value)
        except Exception:
            self.fail(f"{self.name} parameter must be valid JSON string. Got {value}")

        try:
            keys = [*d.keys()]
            for k in keys:
                d[k] = getattr(numba.core.datamodel.models, d[k])
        except Exception:
            self.fail(
                f"Unable to convert input data model dictionary string into dict of numba data models. Got {d}."
            )

        return d


numba_datamodel_dict = NumbaDataModelDictType()


def _generate_structs(struct_decls, aliases, header_path, types, data_models):
    struct_bindings = ""

    for struct_decl in struct_decls:
        struct_name = struct_decl.name
        this_aliases = aliases[struct_name]
        this_type = types[struct_name]
        this_data_model = data_models[struct_name]

        SSR = StaticStructRenderer(
            struct_decl,
            struct_name,
            this_type,
            this_data_model,
            header_path,
            this_aliases,
        )

        struct_bindings += SSR.render_as_str()

    return struct_bindings


@click.command()
@click.argument(
    "input_header", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option("--types", type=numba_type_dict)
@click.option("--datamodels", type=numba_datamodel_dict)
@click.option(
    "--output_dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        writable=True,
    ),
)
def static_binding_generator(input_header, output_dir, types, datamodels):
    """
    A CLI tool to generate CUDA static bindings for CUDA C++ headers.

    INPUT_HEADER: Path to the input CUDA header file.
    OUTPUT_DIR: Path to the output directory where the processed files will be saved.
    """

    try:
        basename = os.path.basename(input_header)
        basename = basename.split(".")[0]
    except Exception:
        click.echo(f"Unable to extract base name from {input_header}.")
        return

    structs, functions, function_templates, class_templates, typedefs, enums = (
        parse_declarations_from_source(
            input_header, [input_header], compute_capability="sm_50"
        )
    )

    aliases = defaultdict(list)
    for typedef in typedefs:
        aliases[typedef.underlying_name].append(typedef.name)

    struct_bindings = _generate_structs(
        structs, aliases, input_header, types, datamodels
    )

    # Example: Save the processed output to the output directory
    output_file = os.path.join(output_dir, f"{basename}.py")
    with open(output_file, "w") as file:
        file.write(struct_bindings)
        click.echo(f"Bindings for {input_header} generated in {output_file}")
