import click
import json
import pandas as pd


@click.command()
@click.argument(
    "gold_name", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.argument(
    "py_lto_off_name", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.argument(
    "py_lto_on_name", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
def compare_gpu_kern(gold_name, py_lto_off_name, py_lto_on_name):
    """Read profile results from gold run result and Numba kernel, compare them.

    GOLD_NAME: JSON profile result of the gold kernel.
    NUMBA_NAME: JSON profile result of the Numba kernel.
    """
    with open(gold_name, "r") as goldf:
        gold_kerns = json.load(goldf)
    with open(py_lto_off_name, "r") as pyf:
        lto_off_kerns = json.load(pyf)
    with open(py_lto_on_name, "r") as pyf:
        lto_on_kerns = json.load(pyf)

    gold_kern = gold_kerns[0]
    lto_off_kern = lto_off_kerns[0]
    lto_on_kern = lto_on_kerns[0]

    columns = [
        "GOLD: " + gold_kern["Name"],
        "NUMBA LTO OFF: " + lto_off_kern["Name"],
        "NUMBA LTO ON: " + lto_on_kern["Name"],
    ]
    index = [k for k in gold_kern.keys() if k != "Name"]
    values = [
        (gold_kern[k], lto_off_kern[k], lto_on_kern[k])
        for k in gold_kern.keys()
        if k != "Name"
    ]

    df = pd.DataFrame(data=values, index=index, columns=columns)

    print(df)

    print("Perf Ratio (NUMBA LTO OFF / GOLD, %): ")
    diff = df.iloc[:, 1] / df.iloc[:, 0] * 100
    diff.index = diff.index.str.strip("%ns)").str.strip("( ")
    print(diff[["Avg", "Med", "Min", "Max", "StdDev"]])

    print("---------")

    print("Perf Ratio (NUMBA LTO ON / GOLD, %): ")
    diff = df.iloc[:, 2] / df.iloc[:, 0] * 100
    diff.index = diff.index.str.strip("%ns)").str.strip("( ")
    print(diff[["Avg", "Med", "Min", "Max", "StdDev"]])


if __name__ == "__main__":
    compare_gpu_kern()
