import numpy as np
from benchopt import BasePlot


def _short_name(name):
    """Strip parameters from a benchopt component name."""
    return name.split("[")[0]


class Plot(BasePlot):
    """Scatter plot comparing two solvers head-to-head across all datasets.

    Each point represents one dataset. Points above the diagonal indicate that
    the solver on the y-axis outperforms the one on the x-axis.
    """

    name = "Head-to-head"
    type = "scatter"
    options = {
        "metric": ["objective_auc_pr", "objective_auc_roc"],
    }

    def plot(self, df, metric):
        solvers = sorted(df["solver_name"].unique())
        if len(solvers) < 2:
            return []

        solver_a, solver_b = solvers[0], solvers[1]
        pivot = (
            df.pivot_table(
                index="dataset_name",
                columns="solver_name",
                values=metric,
                aggfunc="mean",
            )[[solver_a, solver_b]]
            .dropna()
        )

        traces = []
        for dataset in pivot.index:
            x = float(pivot.loc[dataset, solver_a])
            y = float(pivot.loc[dataset, solver_b])
            label = _short_name(dataset)
            traces.append(
                {"x": [x], "y": [y], "label": label, **self.get_style(label)}
            )

        vals = pivot.values.flatten()
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        pad = (hi - lo) * 0.05
        traces.append(
            {
                "x": [lo - pad, hi + pad],
                "y": [lo - pad, hi + pad],
                "label": "Equal performance",
                "color": "gray",
                "marker": "",
            }
        )
        return traces

    def get_metadata(self, df, metric):
        solvers = sorted(df["solver_name"].unique())
        a = _short_name(solvers[0]) if solvers else "Solver A"
        b = _short_name(solvers[1]) if len(solvers) > 1 else "Solver B"
        m = metric.replace("objective_", "").upper().replace("_", "-")
        return {
            "title": f"Head-to-head: {m}",
            "xlabel": f"{a} ({m})",
            "ylabel": f"{b} ({m})",
            "scale": "linear",
        }
