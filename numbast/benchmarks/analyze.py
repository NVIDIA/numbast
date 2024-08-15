import click
import json
import pandas as pd


@click.command()
@click.argument(
    "gold_name", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.argument("py_name", type=click.Path(exists=True, dir_okay=False, file_okay=True))
def compare_gpu_kern(gold_name, py_name):
    """Read profile results from gold run result and Numba kernel, compare them.

    GOLD_NAME: JSON profile result of the gold kernel.
    PY_NAME: JSON profile result of the Numba kernel.
    """
    with open(gold_name, "r") as goldf:
        gold_kerns = json.load(goldf)
    with open(py_name, "r") as pyf:
        py_kerns = json.load(pyf)

    gold_kern = gold_kerns[0]
    py_kern = py_kerns[0]

    columns = ["GOLD: " + gold_kern["Name"], "PY: " + py_kern["Name"]]
    index = [k for k in gold_kern.keys() if k != "Name"]
    values = [(gold_kern[k], py_kern[k]) for k in gold_kern.keys() if k != "Name"]

    df = pd.DataFrame(data=values, index=index, columns=columns)

    print(df)

    print("Perf Ratio (PY / GOLD, %): ")
    diff = df.iloc[:, 1] / df.iloc[:, 0] * 100
    print(diff[["Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)", "StdDev (ns)"]])


if __name__ == "__main__":
    compare_gpu_kern()
