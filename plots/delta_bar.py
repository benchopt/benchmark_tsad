import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from benchopt import BasePlot


def _short_name(name):
    """Strip parameters from a benchopt component name."""
    return name.split("[")[0]


class Plot(BasePlot):
    """Bar chart showing per-dataset score difference between two solvers.

    Bars represent ``score(solver_b) − score(solver_a)`` for each dataset,
    sorted from most negative to most positive. Green bars indicate that
    solver B wins on that dataset; red bars indicate that solver A wins.
    """

    name = "Per-dataset delta"
    type = "bar_chart"
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
                aggfunc="median",
            )[[solver_a, solver_b]]
            .dropna()
        )

        delta = (pivot[solver_b] - pivot[solver_a])
        delta.index = delta.index.map(_short_name)
        grouped = delta.groupby(level=0).apply(list)
        grouped = grouped.reindex(
            grouped.map(lambda v: sum(v) / len(v)).sort_values().index
        )

        medians = {ds: sorted(v)[len(v) // 2] for ds, v in grouped.items()}
        all_med = list(medians.values())
        vmin = min(min(all_med), -1e-9)
        vmax = max(max(all_med), 1e-9)
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap("RdYlGn")

        bars = []
        for short_ds, vals in grouped.items():
            median = medians[short_ds]
            color = mcolors.to_hex(cmap(norm(median)))
            bars.append(
                {
                    "y": [float(v) for v in vals],
                    "label": short_ds,
                    "color": color,
                }
            )
        return bars

    def get_metadata(self, df, metric):
        solvers = sorted(df["solver_name"].unique())
        a = _short_name(solvers[0]) if solvers else "Solver A"
        b = _short_name(solvers[1]) if len(solvers) > 1 else "Solver B"
        m = metric.replace("objective_", "").upper().replace("_", "-")
        return {
            "title": f"Per-dataset delta: {b} − {a} ({m})",
            "ylabel": f"Δ {m}  ({b} − {a})",
        }
