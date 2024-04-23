import click
from jinja2 import Environment, FileSystemLoader
import os


@click.command()
@click.argument("template_path")
@click.argument("output_path")
@click.argument("cuda_version")
@click.argument("python_version")
def cli(template_path, output_path, cuda_version, python_version):
    click.echo(f"Path provided: {template_path}")
    click.echo(f"cuda_version provided: {cuda_version}")
    click.echo("Updating environment template.")
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
    template = env.get_template(os.path.basename(template_path))
    output = template.render(cuda_version=cuda_version, python_version=python_version)
    print(output)
    with open(output_path, "w") as f:
        f.write(output)


if __name__ == "__main__":
    cli()
