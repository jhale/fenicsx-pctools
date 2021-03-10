#!/usr/bin/env python3

import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure as MPLFigure
from matplotlib import rcParams
from shutil import which


rcParams.update({"figure.autolayout": True, "axes.titlesize": "medium", "legend.fontsize": "small"})
if which("latex") is not None:
    rcParams.update(
        {
            "text.usetex": True,
            "font.family": "times",
            "text.latex.preamble": [r"\usepackage{amsmath}"],
        }
    )


def _get_empty_figure(grid=False, subplots=(1, 1), **kwargs):
    n, m = subplots
    kwargs.setdefault("figsize", (m * 5.0, n * 5.0 / 1.618))
    kwargs.setdefault("dpi", 150)

    fig = plt.figure(**kwargs)
    gs = fig.add_gridspec(n, m)
    axes = [fig.add_subplot(gs[i, j]) for j in range(m) for i in range(n)]
    for ax in axes:
        ax.tick_params(which="both", direction="in", right=True, top=True)
        if grid:
            ax.grid(axis="both", which="both", linestyle=":", linewidth=0.8)
    return fig


def _get_label(data):
    models = data["model"].unique()
    assert len(models) == 1
    model_name = models[0]

    label = {"OldroydB": "Oldroyd-B"}[model_name]

    return label


def _get_reference_result():
    r"""Reference data taken from Knechtges et al. [1]_ (p. 84, Table 2, Column M5).

    .. [1] \ P. Knechtges, M. Behr, and S. Elgeti, “Fully-implicit log-conformation formulation
             of constitutive laws,” Journal of Non-Newtonian Fluid Mechanics, vol. 214, pp. 78–87,
             Dec. 2014, doi: 10.1016/j.jnnfm.2014.09.018.
    """
    ref_data = np.array(
        [
            [0.10, 130.3626],
            [0.20, 126.6252],
            [0.30, 123.1912],
            [0.40, 120.5912],
            [0.50, 118.8260],
            [0.60, 117.7752],
            [0.70, 117.3157],
            [0.75, 117.2752],
            [0.80, 117.3454],
            [0.85, 117.5138],
            [0.88, 117.6567],
            [0.89, 117.7107],
            [0.90, 117.7678],
        ]
    )
    return ref_data[:, 0], ref_data[:, 1]


def _plot_dragforce(axes, data, label_suffix=""):
    if isinstance(axes, MPLFigure):
        axes = axes.get_axes()

    model_label = _get_label(data)

    data = data.sort_values("Wi")
    Wi = data["Wi"].to_numpy()
    drag = data["F_drag"].to_numpy()
    ref_Wi, ref_drag = _get_reference_result()

    plot_opts = dict(linestyle="-", marker="^", markerfacecolor="none", color="#93b3bf")
    labels = [
        rf"{model_label + label_suffix} -- reference result",
        rf"{model_label + label_suffix} -- simulation result",
    ]

    ax = axes[0]
    ax.set_xlabel(r"Weissenberg number $\mathrm{Wi}$")
    ax.set_ylabel(r"Drag force $\ F_d$")
    ax.plot(ref_Wi, ref_drag, label=labels[0], **plot_opts)
    plot_opts.update(linestyle="", marker="o", color="#1a6887")
    ax.plot(Wi, drag, label=labels[-1], **plot_opts)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0))


def _generate_plots(results_file, single_file=True):
    data_0 = pandas.read_csv(results_file)

    drawn_figs = []
    if single_file:
        drawn_figs.append(("all-plots", _get_empty_figure(grid=True, subplots=(1, 1))))
    # else:
    #     drawn_figs.append(("drag-force", _get_empty_figure(grid=True, subplots=(1, 1))))
    #     drawn_figs.append(("energy", _get_empty_figure(grid=True, subplots=(1, 1))))

    if single_file:
        _plot_dragforce(drawn_figs[-1][1].get_axes(), data_0)
    # else:
    #     _plot_dragforce(drawn_figs[0][1], data_1)
    #     _plot_energy(drawn_figs[1][1], data_1)

    return drawn_figs


def main(results_file):
    print(f"Generating output from {results_file}...")
    outdir = os.path.dirname(results_file)
    basename = os.path.splitext(os.path.basename(results_file))[0]

    drawn_figs = _generate_plots(results_file)

    print("Generated output:")
    for name, fig in drawn_figs:
        plotfile = os.path.join(outdir, f"fig_{basename}_{name}.png")
        fig.savefig(plotfile)
        plt.close(fig)
        print(f"  + {plotfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script generating output for data from confined cylinder benchmark."
    )
    parser.add_argument("results_file", metavar="FILENAME", type=str, help="CSV file with results")
    args = parser.parse_args()

    main(args.results_file)
