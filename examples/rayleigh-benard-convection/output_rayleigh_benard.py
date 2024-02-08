#!/usr/bin/env python3

import os

import pandas as pd


def get_latex_table(content, caption, label):
    header = [
        "\\begin{table}[h]",
        f"\\caption{{{caption}}}",
        "\\vspace{2mm}",
        "\\begin{center}",
    ]
    footer = [
        "\\end{center}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ]
    lines = []
    for line in header + content + footer:
        if not line.endswith("\n"):
            line += "\n"
        lines.append(line)

    return lines


def _prepare_table_content(data):
    # Add header
    content = [
        r"\begin{tabular}{c|c|c|c|c|c|c}",
        r"DOF & MPI & Nonlinear & Linear & Navier-Stokes & Temperature & Time to\\",
        r"($\times 10^6$) & processes & iterations & iterations & iterations & iterations"
        r" & solution (s)\\",
        r"\hline",
    ]

    # Add data
    num_rows = len(data.index)
    columns = [
        "num_dofs",
        "num_procs",
        "its_snes",
        "its_ksp",
        "its_fs0",
        "its_fs0_avg",
        "its_fs1",
        "its_fs1_avg",
        "SNESSolve",
    ]
    lformat = r"{num_dofs:.4g} & {num_procs:.0f} & {its_snes:.0f} & {its_ksp:.0f}"
    lformat += r" & {its_fs0:.0f} ({its_fs0_avg:.3g}) & {its_fs1:.0f} ({its_fs1_avg:.2g})"
    lformat += r" & {SNESSolve:.3g}"
    values = {}
    for index, row in data[columns].iterrows():
        values.clear()
        for k, v in zip(columns, row.tolist()):
            values[k] = v
        values["num_dofs"] *= 1e-06
        content.append(lformat.format(**values))
        if index < num_rows - 1:
            content[-1] += r"\\"

    # Add footer
    content.append(r"\end{tabular}")

    return content


def _generate_tables(outdir, results_file):
    data_0 = pd.read_csv(results_file)
    data_0 = data_0.sort_values("num_procs")
    basename = os.path.splitext(os.path.basename(results_file))[0]

    saved_tables = []
    pc_approaches = data_0["pc_approach"].unique()
    for pc_approach in pc_approaches:
        data_1 = data_0.loc[data_0["pc_approach"] == pc_approach].reset_index(drop=True)
        content = _prepare_table_content(data_1)

        caption = r"Nonlinear iteration counts, total linear iterations, total iterations for"
        caption += r" Navier-Stokes and temperature solves (with average iterations"
        caption += r" per outer linear solve in brackets), and time to solution"
        caption += rf" for Rayleigh-Bénard convection with \emph{{{pc_approach}}} preconditioning."
        label = f"boussinesq_steady_rayleigh_benard_tab_{pc_approach}"
        table = get_latex_table(content, caption, label)

        tabfile = os.path.join(outdir, f"tab_{basename}_{pc_approach}.tex")
        with open(tabfile, "w") as f:
            f.writelines(table)
        saved_tables.append(tabfile)

    return saved_tables


def _run_pvscript(pvscript_args):
    import subprocess
    from shutil import which

    pvbatch = which("pvbatch")
    if pvbatch is not None:
        thisfiledir = os.path.dirname(os.path.realpath(__file__))
        pvscript = os.path.realpath(pvscript_args.pop(0))  # convention: script on the 1st position

        TEXMFDIST_DIR = subprocess.check_output("kpsewhich -var-value=TEXMFDIST", shell=True)
        TEXMFDIST_DIR = TEXMFDIST_DIR.decode("utf-8")[:-1]
        os.environ.setdefault("TEXMFDIST_DIR", TEXMFDIST_DIR)
        cmd = [pvbatch, pvscript, *pvscript_args]
        print("Calling '{}'".format(" ".join(cmd)))
        subprocess.check_call(cmd, cwd=thisfiledir)
        os.environ.pop("TEXMFDIST_DIR")


def main(results_file, pvscript_args=None):
    print(f"Generating output from {results_file}...")
    outdir = os.path.dirname(results_file)

    saved_tables = _generate_tables(outdir, results_file)
    print("Generated tables:")
    for table in saved_tables:
        print(f"  + {table}")

    if pvscript_args:
        _run_pvscript(pvscript_args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script generating tables for the Rayleigh-Bénard convection problem."
    )
    parser.add_argument("results_file", metavar="FILENAME", type=str, help="CSV file with results")
    parser.add_argument("--pvscript", nargs="*", help="arguments for paraview script")
    args = parser.parse_args()

    from mpi4py import MPI

    assert MPI.COMM_WORLD.size == 1, "This script should not be executed in parallel"

    # TODO: Store results using YAML, so pvscript arguments can be contained therein.
    main(args.results_file, args.pvscript)
