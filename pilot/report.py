import pandas as pd
import matplotlib.pyplot as plt

import pypandoc
from pathlib import Path

SAVE_DIR = "output/"
INPUT_DIR = SAVE_DIR + "input/"
TABLES_DIR = SAVE_DIR + "tables/"
FIGURES_DIR = SAVE_DIR + "figures/"

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


def make_report(lattice, field, algorithm, observables):

    report_str = "# Inputs\n"
    Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
    for name, obj in zip(
        ["lattice", "field", "algorithm"], [lattice, field, algorithm]
    ):
        df = make_summary_table(obj)
        with open(INPUT_DIR + name + ".csv", "w") as f:
            f.write(df.to_csv())

        report_str += f"\n## {name}\n"
        report_str += f"\n{df.to_markdown()}\n"

    report_str += "\n# Tables\n"
    Path(TABLES_DIR).mkdir(parents=True, exist_ok=True)
    for table in observables.tables:
        df = getattr(observables, "table_" + table)
        outfile = TABLES_DIR + table + ".csv"
        with open(outfile, "w") as f:
            f.write(df.to_csv())

        report_str += f"\n## {table.replace('_', ' ')}\n"
        report_str += f"\n{df.to_markdown()}\n"

    report_str += "\n# Figures\n"
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    for figure in observables.figures:
        fig = getattr(observables, "plot_" + figure)()
        outfile = FIGURES_DIR + figure + ".png"
        plt.savefig(outfile)
        plt.clf()

        report_str += f"\n## {figure.replace('_', ' ')}\n"
        report_str += f"\n![]({outfile})\n"

    try:
        report_file = SAVE_DIR + "report.html"
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
    except:
        # TODO: this sucks because observables are re-computed
        print("Failed to create report .html file. Writing to .txt instead")
        outfile = SAVE_DIR + "report.txt"
        output = "\n".join(
            [obj.__str__() for obj in [lattice, field, algorithm, observables]]
        )
        with open(outfile, "w") as f:
            f.write(output)

    return
