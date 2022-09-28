#!/usr/bin/env python3

import os
from shutil import which

import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib import rcParams
from matplotlib.figure import Figure as MPLFigure

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

    Re_max = data["Re"].unique().max()
    Wi_max = data["Wi"].unique().max()

    label = ""
    if model_name == "NavierStokes":
        label += rf"Navier-Stokes, $\mathrm{{max\;Re}}$ = {Re_max:.2g}"
    elif model_name == "PowerLaw":
        label += rf"Power-law, $\mathrm{{max\;Re}}$ = {Re_max:.2g}"
    elif model_name == "OldroydB":
        mu0 = data["mu_0"].unique()
        assert len(mu0) == 1
        # FIXME: Remove the following entry from results as soon as the Leonov model is implemented!
        _model_type = data["_model_type"].unique()
        assert len(_model_type) == 1
        label += "Leonov" if _model_type[0] == "nonlinear" else "Oldroyd-B"
        label += r" ($\mu_0 = 0$)" if mu0[0] == 0.0 else ""
        label += rf", $\mathrm{{max\;Re}}$ = {Re_max:.2g}"
        label += rf", $\mathrm{{max\;Wi}}$ = {Wi_max:.2g}"

    return label


def _get_analytic_formulas(data):
    models = data["model"].unique()
    assert len(models) == 1
    model_name = models[0]

    shstress_formula = None
    N1_formula, N2_formula = None, None
    if model_name == "NavierStokes":
        mu = data["mu"].unique()[0]
        shstress_formula = lambda g: mu * g
        N1_formula = lambda g: 0.0 * g
        N2_formula = lambda g: 0.0 * g

    elif model_name == "PowerLaw":
        from problem_PowerLaw import CarreauYasudaViscosity

        mu_0 = data["mu_0"].unique()[0]
        mu_8 = data["mu_8"].unique()[0]
        alpha = data["alpha"].unique()[0]
        n = data["n"].unique()[0]

        mu = CarreauYasudaViscosity(mu_0, mu_8, alpha, n)
        shstress_formula = lambda g: mu(g ** 2.0) * g
        N1_formula = lambda g: 0.0 * g
        N2_formula = lambda g: 0.0 * g

    elif model_name == "OldroydB":
        # FIXME: Missing sanity checks!
        mu0 = data["mu_0"].unique()[0]
        mu1 = data["mu_1"].unique()[0]
        G1 = data["G_1"].unique()[0]
        tau1 = mu1 / G1

        # FIXME: Remove the following entry from results as soon as the Leonov model is implemented!
        _model_type = data["_model_type"].unique()[0]
        A1 = lambda g: np.ones_like(g)
        if _model_type == "nonlinear":
            A1 = lambda g: (np.sqrt(1.0 + 4.0 * (tau1 * g) ** 2) - 1.0) / (2.0 * (tau1 * g) ** 2)

        shstress_formula = lambda g: (mu0 + mu1 * A1(g)) * g
        N1_formula = lambda g: 2.0 * mu1 * tau1 * (A1(g) ** 1.5) * (g ** 2)
        N2_formula = lambda g: 0.0 * g
        if _model_type == "nonlinear":
            N2_formula = lambda g: -2.0 * mu1 * tau1 * (A1(g) ** 1.5) * (g ** 2)

    return shstress_formula, N1_formula, N2_formula


def _plot_flowcurves(axes, data, label_suffix="", color="#1a6887", show_raw_data=False):
    if isinstance(axes, MPLFigure):
        axes = axes.get_axes()

    model_label = _get_label(data)
    shstress_formula = _get_analytic_formulas(data)[0]

    data = data.sort_values("dgamma")
    dgamma = data["dgamma"].to_numpy()
    stress = data["shstress"].to_numpy()
    eta = stress / dgamma
    e_min = 0.5 * (dgamma.max() - dgamma.min()) / dgamma.max()
    e_max = 0.5 * (dgamma.max() - dgamma.min()) / dgamma.min()
    dgamma_ex = 10 ** np.linspace(
        np.log10(dgamma.min() - e_min), np.log10(dgamma.max() + e_max), num=100
    )
    stress_ex = shstress_formula(dgamma_ex)
    eta_ex = stress_ex / dgamma_ex
    dgamma_app = data["dgamma_app"].to_numpy()
    stress_app = data["shstress_app"].to_numpy()
    eta_app = stress_app / dgamma_app

    plot_opts = dict(markerfacecolor="none", color=color)
    labels = [
        rf"{model_label + label_suffix} -- analytic formula",
        rf"{model_label + label_suffix} -- simulation result",
        rf"{model_label + label_suffix} -- raw data",
    ]

    ax = axes[0]
    # ax.set_xlabel(r"shear rate $[\mathrm{s}^{-1}]$")
    ax.set_ylabel(r"shear stress $[\mathrm{Pa}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(dgamma_ex, stress_ex, label=labels[0], linestyle="-", **plot_opts)
    ax.plot(dgamma, stress, label=labels[1], marker="o", linestyle="", **plot_opts)
    if show_raw_data:
        ax.plot(dgamma_app, stress_app, label=labels[2], marker="x", linestyle="", **plot_opts)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0))

    ax = axes[1]
    ax.set_xlabel(r"shear rate $[\mathrm{s}^{-1}]$")
    ax.set_ylabel(r"viscosity $[\mathrm{Pa \cdot s}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Fix y scale for almost constant functions
    if not show_raw_data:
        _y_min, _y_max = eta.min(), eta.max()
    else:
        _y_min, _y_max = min(eta.min(), eta_app.min()), max(eta.max(), eta_app.max())
    _y_rdiff = (_y_max - _y_min) / _y_max
    if _y_rdiff < 0.01:
        ax.set_ylim(10 ** (0.75 * np.log10(_y_min)), 10 ** (1.25 * np.log10(_y_max)))

    ax.plot(dgamma_ex, eta_ex, label=labels[0], linestyle="-", **plot_opts)
    ax.plot(dgamma, eta, label=labels[1], marker="o", linestyle="", **plot_opts)
    if show_raw_data:
        ax.plot(dgamma_app, eta_app, label=labels[2], marker="x", linestyle="", **plot_opts)
    # ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0))


def _plot_nstressdiffs(axes, data, label_suffix="", color="#1a6887", adjust_yscale=True):
    if isinstance(axes, MPLFigure):
        axes = axes.get_axes()

    model_label = _get_label(data)
    N1_formula = _get_analytic_formulas(data)[1]
    N2_formula = _get_analytic_formulas(data)[2]

    data = data.sort_values("dgamma")
    dgamma = data["dgamma"].to_numpy()
    Trr = data["nstress_rr"].to_numpy()
    Tpp = data["nstress_pp"].to_numpy()
    Tzz = data["nstress_zz"].to_numpy()
    N1 = Tzz - Trr
    N2 = Trr - Tpp
    nstress_max = np.max(np.abs([Trr, Tpp, Tzz]), axis=0)  # maximum values of normal stress
    e_min = 0.5 * (dgamma.max() - dgamma.min()) / dgamma.max()
    e_max = 0.5 * (dgamma.max() - dgamma.min()) / dgamma.min()
    dgamma_ex = 10 ** np.linspace(
        np.log10(dgamma.min() - e_min), np.log10(dgamma.max() + e_max), num=100
    )
    N1_ex = N1_formula(dgamma_ex)
    N2_ex = N2_formula(dgamma_ex)

    plot_opts = dict(markerfacecolor="none", color=color)
    labels = [
        rf"{model_label + label_suffix} -- analytic formula",
        rf"{model_label + label_suffix} -- simulation result",
    ]

    ax = axes[0]
    # ax.set_xlabel(r"shear rate $[\mathrm{s}^{-1}]$")
    ax.set_ylabel(r"$1^{\mathrm{st}}$ normal stress difference $[\mathrm{Pa}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if np.max(N1 / nstress_max) < 1e-03:  # almost zero
        ax.set_yscale("linear")
        if adjust_yscale:
            ax.set_ylim(-0.1 * nstress_max.min(), nstress_max.min())
    ax.plot(dgamma_ex, N1_ex, label=labels[0], linestyle="-", **plot_opts)
    ax.plot(dgamma, N1, label=labels[1], marker="o", linestyle="", **plot_opts)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0))

    ax = axes[1]
    ax.set_xlabel(r"shear rate $[\mathrm{s}^{-1}]$")
    ax.set_ylabel(r"negative $2^{\mathrm{nd}}$ normal stress difference $[\mathrm{Pa}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if np.max(np.abs(N2) / nstress_max) < 1e-03:  # almost zero
        ax.set_yscale("linear")
        if adjust_yscale:
            ax.set_ylim(-0.1 * nstress_max.min(), nstress_max.min())
    ax.plot(dgamma_ex, -N2_ex, label=labels[0], linestyle="-", **plot_opts)
    ax.plot(dgamma, -N2, label=labels[1], marker="o", linestyle="", **plot_opts)
    # ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0))


def _generate_plots(results_file, groupby, single_file=True):
    data_0 = pandas.read_csv(results_file)

    drawn_figs = []
    if single_file:
        drawn_figs.append(("all-plots", _get_empty_figure(grid=True, subplots=(2, 2))))
    else:
        drawn_figs.append(("flow-curve", _get_empty_figure(grid=True, subplots=(2, 1))))
        drawn_figs.append(("nstress-diffs", _get_empty_figure(grid=True, subplots=(2, 1))))

    groupIDs = data_0[groupby].unique()
    for groupID in groupIDs:
        data_1 = data_0.loc[data_0[groupby] == groupID].reset_index(drop=True)
        if single_file:
            _plot_flowcurves(drawn_figs[-1][1].get_axes()[0:2], data_1)
            _plot_nstressdiffs(drawn_figs[-1][1].get_axes()[2:4], data_1)
        else:
            _plot_flowcurves(drawn_figs[0][1], data_1)
            _plot_nstressdiffs(drawn_figs[1][1], data_1)

    return drawn_figs


def main(results_file, groupby="model"):
    print(f"Generating output from {results_file}...")
    outdir = os.path.dirname(results_file)
    basename = os.path.splitext(os.path.basename(results_file))[0]

    #drawn_figs = _generate_plots(results_file, groupby)

    #print("Generated output:")
    #for name, fig in drawn_figs:
    #    plotfile = os.path.join(outdir, f"fig_{basename}_{name}.png")
    #    fig.savefig(plotfile)
    #    plt.close(fig)
    #    print(f"  + {plotfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script generating output for data from digital capillary rheometer."
    )
    parser.add_argument("results_file", metavar="FILENAME", type=str, help="CSV file with results")
    parser.add_argument(
        "-g", dest="groupby", metavar="PARAM", default="model", help="group by PARAM"
    )
    args = parser.parse_args()

    main(args.results_file, args.groupby)
