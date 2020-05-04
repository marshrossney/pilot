import pandas as pd
import matplotlib.pyplot as plt

import pypandoc
from pathlib import Path

import pilot.params as p
from pilot.utils import NotDefinedForField

# Steal basic html and css templates
HTML_TEMPLATE = r"https://raw.githubusercontent.com/tonyblundell/pandoc-bootstrap-template/master/template.html"
CSS_TEMPLATE = r"https://raw.githubusercontent.com/tonyblundell/pandoc-bootstrap-template/master/template.css"


def make_summary_table(obj):
    labels, values = [], []
    for prop in obj.summary_properties:
        labels.append(prop.replace("_", " "))
        values.append(str(getattr(obj, prop)))

    df = pd.DataFrame(values, index=labels, columns=["",])
    df.index.name = type(obj).__name__
    return df


def make_report(lattice, field, algorithm, observables, output, mode):

    # Set dirs
    base_dir = (output + "/").replace("//", "/")
    if mode == "therm":
        base_dir += "prelims/therm/"
    elif mode == "autocorr":
        base_dir += "prelims/autocorr/"
    input_dir = base_dir + "input/"
    tables_dir = base_dir + "tables/"
    figures_dir = base_dir + "figures/"

    # Don't bother will all observables if just looking at thermalisation
    # or autocorrelation
    if mode == "therm":
        filtered = lambda l: [i for i in l if "series" in i]
    elif mode == "autocorr":
        filtered = lambda l: [i for i in l if "series" in i or "autocorrelation" in i]
    else:
        filtered = lambda l: l

    report_str = "# Inputs\n"
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    for name, obj in zip(
        ["lattice", "field", "algorithm"], [lattice, field, algorithm]
    ):
        df = make_summary_table(obj)
        with open(input_dir + name + ".csv", "w") as f:
            f.write(df.to_csv())

        report_str += f"\n## {name}\n"
        report_str += f"\n{df.to_markdown()}\n"

    report_str += "\n# Tables\n"
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    for table in filtered(observables.tables):

        try:
            df = getattr(observables, "table_" + table)
            outfile = tables_dir + table + ".csv"
            with open(outfile, "w") as f:
                f.write(df.to_csv())

            report_str += f"\n## {table.replace('_', ' ')}\n"
            report_str += f"\n{df.to_markdown()}\n"
        except NotDefinedForField:
            pass

    report_str += "\n# Figures\n"
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    for figure in filtered(observables.figures):
        try:
            fig = getattr(observables, "plot_" + figure)
            outfile = figures_dir + figure + ".png"
            plt.savefig(outfile)
            plt.clf()

            report_str += f"\n## {figure.replace('_', ' ')}\n"
            report_str += f"\n![]({outfile})\n"

        except NotDefinedForField:
            pass

    report_file = base_dir + "report.html"
    output = pypandoc.convert_text(
        report_str,
        "html",
        format="md",
        outputfile=report_file,
        extra_args=[
            f"--template={HTML_TEMPLATE}",
            f"--css={CSS_TEMPLATE}",
            "--self-contained",
            "--toc",
            "--toc-depth=2",
        ],
    )

    return
