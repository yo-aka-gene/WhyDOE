import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, accuracy_score
import statsmodels.api as sm
from tqdm.notebook import tqdm
import warnings

from doe_modules.design import CLOO, PlackettBurman, DOCLOO 
from doe_modules.design import d_criterion
from doe_modules.preferences import kwarg_err, outputdir, kwarg_savefig, fmt_suffix
from ._mlr import MLR
from ._theoretical_effects import TheoreticalEffects
from ._dunnett import Dunnett
from ._dunnett_power import dunnett_power
from ._anova_power import anova_power
from ._abstract import AbstractSimulator


def aptitude_score(res, gt):
    if gt.unique().size == 1:
        f = lambda res, gt: accuracy_score(res, gt)
    else:
        f = lambda res, gt: cohen_kappa_score(res, gt, weights="linear")
    return np.nan if res.isna().all() else f(res, gt)


def kappa(res, gt):
    return np.nan if res.isna().all() else cohen_kappa_score(res, gt, weights="linear")


class NoiseConfigurator:
    def __init__(
        self,
        noise_arr: list
    ) -> None:
        self.arr = noise_arr
        self.conf = [
            dict(kwarg_err=dict(loc=0, scale=sigma)) for sigma in noise_arr
        ]
        self.names = [
            "$\sigma=" + f"{sigma}$" for sigma in noise_arr
        ]
        self.size = len(noise_arr)


    def __call__(self) -> dict:
        return {
            "arr": self.arr,
            "conf": self.conf,
            "names": self.names,
            "size": self.size
        }


class MetricConfigurator:
    def __init__(
        self,
        evaluation_metric,
        metric_name: str,
    ) -> None:
        self.f = evaluation_metric
        self.name = metric_name


    def __call__(self) -> dict:
        return {
            "f": self.f,
            "name": self.name
        }


class MetaDataConfigurator:
    def __init__(
        self, 
        simulator: AbstractSimulator,
        n_range: list,
        N: int,
        n_add: list,
        n_rep: int,
        designs: dict, 
        edge_assignment: np.ndarray,
        random_state: int,
        seeds: np.ndarray,
        dunnett: bool
    ) -> None:
        self.simulator = simulator
        self.n_range = n_range
        self.N = N
        self.n_add = n_add
        self.n_rep = n_rep
        self.designs = designs
        self.edge_assignment = edge_assignment
        self.random_state = random_state
        self.seeds = seeds
        self.dunnett = dunnett


    def __call__(self) -> dict:
        return {
            "simulator": self.simulator,
            "n_range": self.n_range,
            "N": self.N,
            "n_add": self.n_add,
            "n_rep": self.n_rep,
            "designs": self.designs,
            "edge_assignment": self.edge_assignment,
            "random_state": self.random_state,
            "seeds": self.seeds,
            "dunnett": self.dunnett
        }


class Benchmarker:
    def __init__(
        self,
        simulator: AbstractSimulator,
        noise_arr: list,
        n_range: list,
        n_rep: int,
        designs: dict = {
            "pb": PlackettBurman,
            "cloo": CLOO,
        }, 
        edge_assignment: np.ndarray = None,
        random_state: int = 0,
        dunnett: bool = False,
        evaluation_metric = aptitude_score,
        metric_name: str = "Aptitude scores",
        suffix: str = ""
    ):
        self.noise = NoiseConfigurator(noise_arr=noise_arr)
        self.suffix = suffix
        self.metric = MetricConfigurator(
            evaluation_metric=evaluation_metric, 
            metric_name=metric_name
        )
        
        np.random.seed(random_state)
        seeds = np.random.randint(0, 2**32, n_rep)
        
        n_range = np.array(n_range)
        
        self.metadata = MetaDataConfigurator(
            simulator=simulator,
            n_range=n_range,
            N=None,
            n_add=np.array([]),
            n_rep=n_rep,
            designs=designs,
            edge_assignment=edge_assignment,
            random_state=random_state,
            seeds=seeds,
            dunnett=dunnett
        )
        
        caller = (lambda m: m()) if edge_assignment is None else (lambda m: m(edge_assignment=edge_assignment))
        self.conditions = {
            k: list(
                map(caller, [simulator] * self.noise.size * n_range.size * seeds.size)
            ) for k in designs
        }
        self.cmap = list(self.conditions.values())[0][0].cmap
        
        for k, models in tqdm(
            self.conditions.items(), total=len(self.conditions),
            desc="generating simulated results with experimental designs"
        ):
            [
                m.simulate(
                    design=designs[k],
                    n_rep=n_range[(i // n_rep) % n_range.size],
                    random_state=seeds[i % n_rep],
                    model_kwargs=self.noise.conf[i // (n_rep * n_range.size)]
                ) for i, m in tqdm(
                    enumerate(models), total=len(models),
                    desc=f"{k}-based simulators"
                )
            ]
        
        self.theoretical = [
            TheoreticalEffects(
                simulation=simulator() if edge_assignment is None else simulator(
                    edge_assignment=edge_assignment
                ), random_state=random_state, model_kwargs=nc
            ) for nc in tqdm(
                self.noise.conf, total=len(self.noise.conf),
                desc="computing theoretical main effects"
            )
        ]

        self.ground_truth = [
            self.theoretical[i].summary(dtype=int) for i in tqdm(
                np.tile(
                    np.arange(self.noise.size), 
                    n_range.size * seeds.size
                ).reshape(-1, self.noise.size).T.ravel(),
                total=self.noise.size * n_range.size * seeds.size,
                desc="generating ground truth labels for each condition"
            )
        ]

        warnings.simplefilter('ignore')
        
        self.results = {
            k: [
                MLR(m).summary(anova=True, dtype=int) for m in tqdm(
                    cond, total=len(cond),
                    desc=f"{k}-based simulators"
                )
            ] for k, cond in tqdm(
                self.conditions.items(), total=len(self.conditions),
                desc="encoding simulated results with MLR/AVOVA"
            )
        }
        
        if dunnett and ("cloo" in self.conditions):
            self.results = {
                **self.results,
                "pair": [
                    Dunnett(m).summary(dtype=int) for m in tqdm(
                        self.conditions["cloo"], total=len(self.conditions["cloo"]),
                        desc="encoding C+LOO-based simulated results with Dunnett's test"
                    )
                ]
            }
        
        self.scores = pd.DataFrame({
            "n": np.tile(
                np.tile(n_range, seeds.size).reshape(-1, n_range.size).T.ravel(), 
                self.noise.size
            ),
            "err": np.ravel([[v] * n_range.size * seeds.size for v in self.noise.arr]),
            **{
                f"{k}_metric": [
                    self.metric.f(res, gt) for res, gt in tqdm(
                        zip(results, self.ground_truth), total=len(self.ground_truth),
                        desc=f"{k}-based simulators" if k != "pair" else "dunnett's test"
                    )
                ] for k, results in tqdm(
                    self.results.items(), total=len(self.results),
                    desc="evaluating experimental design performance"
                )
            }
        })
        
        power = lambda key: anova_power if key != "pair" else dunnett_power
        condition_key = lambda key: "cloo" if key == "pair" else key
        
        power_summary = []
        for k in tqdm(self.results, total=len(self.results), desc="power analysis"):
            power_summary += [
                power(k)(m).assign(
                    noise=pd.Series([self.noise.names[i // (n_rep * n_range.size)] for _ in m.metadata["factor_list"]])
                ) for i, m in tqdm(
                    enumerate(self.conditions[condition_key(k)]), total=len(self.conditions[condition_key(k)]),
                    desc=f"{k}-based simulators" if k != "pair" else "dunnett's test"
                )
            ]
        
        self.power = pd.concat(power_summary)



    def plot_groundtruth(
        self,
        xscales: list = [1.6, 1.3],
        size: float = .5,
        unit_x_length: float = 3,
        unit_y_length: float = 2,
        wspace: float = .05
    ) -> tuple:
        fig, ax = plt.subplots(
            1, self.noise.size, 
            figsize=(unit_x_length * self.noise.size, unit_y_length), 
            sharey=True
        )
        plt.subplots_adjust(wspace=wspace)

        for t, sigma, a in zip(self.theoretical, self.noise.conf, ax.ravel()):
            t.plot(ax=a, jitter_ratio=.04, xscales=np.array(xscales), size=size, **kwarg_err)
            a.set(ylabel="", title="$\sigma=" + f"{sigma['kwarg_err']['scale']}" + "$")
        
        return fig, ax


    def plot_benchmarking(
        self,
        items: list = ["cloo", "pair", "pb"],
        cmap: dict = {"cloo": "C0", "pair": "C2", "pb": "C1"},
        unit_x_length: float = 3,
        unit_y_length: float = 2,
        wspace: float = .05,
        noise: float = None,
        show_dunnet: bool = True
    ):
        items = [
                item for item in items if (item != "pair") or (self.metadata.dunnett and show_dunnet)
            ]
        
        if noise is None:
            fig, ax = plt.subplots(
                1, self.noise.size, 
                figsize=(unit_x_length * self.noise.size, unit_y_length), 
                sharey=True
            )
            plt.subplots_adjust(wspace=wspace)

            for i, a in enumerate(ax.ravel()):
                e = self.scores.err.unique()[i]
                for k in items:
                    sns.lineplot(
                        data=self.scores[self.scores.err == e], x="n", y=f"{k}_metric", marker="s", 
                        ax=a, 
                        label=self.conditions[k][0].metadata["design"] if k != "pair" else "Dunnett", 
                        color=cmap[k],
                        markeredgewidth=0, errorbar=("ci", 95), n_boot=1000, seed=self.metadata.random_state
                    )
                a.set_ylim(-0.05, 1.05)
                a.set_xticks(self.metadata.n_range.tolist())
                a.set_yticks(np.linspace(0, 1, 6).tolist())
                a.set(
                    title=self.noise.names[i], xlabel="N: Num. of replication", 
                    ylabel=self.metric.name
                )
                a.legend(loc="best", fontsize="x-small", ncol=2)

        else:
            fig, ax = plt.subplots(figsize=(unit_x_length, unit_y_length))
            
            for k in items:
                sns.lineplot(
                    data=self.scores[self.scores.err == noise], x="n", y=f"{k}_metric", marker="s", 
                    ax=ax, 
                    label=self.conditions[k][0].metadata["design"] if k != "pair" else "Dunnett", 
                    color=cmap[k],
                    markeredgewidth=0, errorbar=("ci", 95), n_boot=1000, seed=self.metadata.random_state
                )
            ax.set_ylim(-0.05, 1.05)
            ax.set(title=self.metadata.simulator().name, xlabel="N: Num. of replication", ylabel=self.metric.name)
            ax.legend(loc="upper left", fontsize="x-small")
            ax.set_xticks(self.metadata.n_range.tolist())
            ax.set_yticks(np.linspace(0, 1, 6).tolist())
        
        return fig, ax

    
    def plot_power(
        self,
        items: list = ["cloo", "pair", "pb"],
        labels: dict = {"cloo": "C+LOO", "pair": "Dunnett", "pb": "PB"},
        unit_x_length: float = 3,
        unit_y_length: float = 2,
        wspace: float = .05,
        hspace: float = .05
    ):
        items = [
            labels[k] for k in [
                item for item in items if (item != "pair") or self.metadata.dunnett
            ]
        ]
        fig, ax = plt.subplots(len(items), self.noise.size, figsize=(3 * self.noise.size, 2 * len(items)), sharex=True, sharey=True)
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        
        bbox_to_anchor = (1, .5) if self.metadata.dunnett else (1, 1)

        for i, a in enumerate(ax.ravel()):
            sns.lineplot(
                data=self.power[
                    (self.power.noise == self.noise.names[i % self.noise.size]) & (self.power.model == items[i // self.noise.size])
                ].reset_index(), 
                x="n_rep", y="power", hue="term",
                ax=a, palette=self.cmap, marker="o",
                markeredgewidth=0, n_boot=1000, seed=self.metadata.random_state
            )

            a.set_ylim(-0.05, 1.05)
            a.set_xticks(self.metadata.n_range.tolist())
            a.set_yticks(np.linspace(0, 1, 6).tolist())
            if i % self.noise.size == 0:
                a.set_ylabel(r"$1-\beta$" + f" ({items[i // self.noise.size]})")

            a.set_title(self.noise.names[i % self.noise.size]) if i // self.noise.size == 0 else a.set_xlabel("N: Num. of replication")
            a.legend(loc="center left", bbox_to_anchor=bbox_to_anchor ) if i == 2 * self.noise.size - 1 else a.legend().remove()
        
        return fig, ax


class BenchmarkingPipeline:
    def __init__(
        self,
        configuration: dict,
        simulator: AbstractSimulator = None,
        noise_arr: list = [.5, 1, 2, 4],
        n_range: list = np.arange(1, 11),
        n_rep: int = 10,
        designs: dict = {
            "pb": PlackettBurman,
            "cloo": CLOO,
        }, 
        edge_assignment: np.ndarray = None,
        random_state: int = 0,
        dunnett: bool = False,
        evaluation_metric = aptitude_score,
        metric_name: str = "Aptitude scores",
    ):
        config = {
            k: {
                "simulator": simulator,
                "noise_arr": noise_arr,
                "n_range": n_range,
                "n_rep": n_rep,
                "designs": designs,
                "edge_assignment": edge_assignment,
                "random_state": random_state,
                "dunnett": dunnett,
                "evaluation_metric": evaluation_metric,
                "metric_name": metric_name,
                **d
            } for k, d in configuration.items()
        }
        
        self.configuration = {
            k: Benchmarker(**d, suffix=fmt_suffix(k)) for k, d in tqdm(
                config.items(), total=len(configuration),
                desc="Running benchmarking pipeline"
            )
        }
        
        self.outputs = {k: {} for k in config}


    def plot(
        self,
        func: list = [
            "plot_groundtruth", "plot_benchmarking", "plot_power"
        ],
        kwargs: dict = {
            "plot_groundtruth": {}, 
            "plot_benchmarking": {}, 
            "plot_power": {}
        },
        savefig: bool = False,
        where: str = outputdir,
        titles: dict = {
            "plot_groundtruth": "groundtrue_results", 
            "plot_benchmarking": "benchmarks", 
            "plot_power": "power"
        },
        kwarg_save: dict = kwarg_savefig
    ):
        for f in tqdm(func, total=len(func), desc="Plotting"):
            for k, bm in tqdm(
                self.configuration.items(), total=len(self.configuration),
                desc=f
            ):
                fig, ax = eval(f"bm.{f}")(**kwargs[f])
                if savefig:
                    fig.savefig(f"{where}/{titles[f]}{bm.suffix}", **kwarg_save)
                self.outputs[k][f] = (fig, ax)


class DOptimizationBenchmarker(Benchmarker):
    def __init__(
        self,
        from_benchmarker: Benchmarker,
        N: int,
        n_add: list,
        designs: dict = {"docloo": DOCLOO}
    ):
        self.noise = from_benchmarker.noise
        self.suffix = from_benchmarker.suffix
        self.metric = from_benchmarker.metric
        self.metadata = from_benchmarker.metadata
        self.cmap = from_benchmarker.cmap
        
        n_add = np.array(n_add)
        edge_assignment = self.metadata.edge_assignment
        seeds = self.metadata.seeds
        
        
        self.metadata.N = N
        self.metadata.n_add = n_add
        n_iter = self.noise.size * n_add.size * seeds.size
        
        caller = (lambda m: m()) if edge_assignment is None else (lambda m: m(edge_assignment=edge_assignment))
        self.conditions = {
            k: list(map(caller, [self.metadata.simulator] * n_iter)) for k in designs
        }
        
        
        for k, models in tqdm(
            self.conditions.items(), total=len(self.conditions),
            desc="generating simulated results with experimental designs"
        ):
            [
                m.simulate(
                    design=designs[k],
                    n_rep=N,
                    n_add=n_add[(i // seeds.size) % n_add.size],
                    random_state=seeds[i % seeds.size],
                    model_kwargs=self.noise.conf[i // (seeds.size * n_add.size)]
                ) for i, m in tqdm(
                    enumerate(models), total=len(models),
                    desc=f"{k}-based simulators"
                )
            ]

        self.theoretical = from_benchmarker.theoretical
        self.ground_truth = [
            self.theoretical[i // (seeds.size * n_add.size)].summary(dtype=int) for i in tqdm(
                range(n_iter), total=n_iter,
                desc="generating ground truth labels for each condition"
            )
        ]

        warnings.simplefilter('ignore')

        self.results = {
            k: [
                MLR(m).summary(anova=True, dtype=int) for m in tqdm(
                    cond, total=len(cond),
                    desc=f"{k}-based simulators"
                )
            ] for k, cond in tqdm(
                self.conditions.items(), total=len(self.conditions),
                desc="encoding simulated results with MLR/AVOVA"
            )
        }
        
        self.scores = pd.DataFrame({
            "err": np.ravel([[v] * n_range.size * seeds.size for v in self.noise.arr]),
            **{
                f"{k}_nmax": [
                    len(m.exmatrix) for m in tqdm(
                        cond, total=n_iter,
                        desc=f"{k}-based simulators"
                    )
                ] for k, cond in  tqdm(
                    self.conditions.items(), total=len(self.conditions),
                    desc="calculating n_max values"
                )
            }
            **{
                f"{k}_metric": [
                    self.metric.f(res, gt) for res, gt in tqdm(
                        zip(results, self.ground_truth), total=len(self.ground_truth),
                        desc=f"{k}-based simulators"
                    )
                ] for k, results in tqdm(
                    self.results.items(), total=len(self.results),
                    desc="evaluating experimental design performance"
                )
            },
            **{
                f"{k}_d": np.fromiter(
                    map(d_criterion, map(sm.add_constant, [m.exmatrix for m in cond])), 
                    float
                ) for k, cond in tqdm(
                    self.conditions.items(), total=len(self.conditions), 
                    desc="evaluating D-criterion values"
                )
            }
        })
        
        n_factor = self.metadata.simulator().n_factor
        
        self.scores_without_doptimization = pd.concat(
            [
                from_benchmarker.scores[from_benchmarker.scores.n == N].reset_index(drop=True),
                pd.DataFrame({
                    f"{k}_d": d_criterion(cond[self.metadata.n_rep * (N - 1)].exmatrix) * np.ones(
                        self.noise.size * self.metadata.n_rep
                    ) for k, cond in self.conditions.items()
                }),
                pd.DataFrame({
                    f"{k}_nmax": len(des().get_exmatrix(n_factor=n_factor)()) * N * np.ones(
                        self.noise.size * self.metadata.n_rep
                    ).astype(int) for k, des in self.metadata.designs
                })
                
            ],
            axis=1
        )
        
        self.power = {
            k: pd.concat(
                [
                    pd.concat(
                        [
                            anova_power(m),
                            pd.DataFrame({
                                "d": self.scores[f"{k}_d"].values[i] * np.ones(n_factor),
                                "nmax": self.scores[f"{k}_nmax"].values[i] * np.ones(n_factor),
                                "noise": ["$\sigma=" + f"{self.scores.err.values[i]}$"] * n_factor
                            })
                        ]
                    ) for i, m in tqdm(
                        enumerate(cond), total=n_iter,
                        desc=f"{k}-based simulators"
                    )
                ],
                axis=1
            ) for k, cond in tqdm(
                self.conditions.items(), total=len(self.conditions), 
                desc="Power analysis"
            )
        }


    def plot_groundtruth(
        self,
        xscales: list = [1.6, 1.3],
        size: float = .5,
        unit_x_length: float = 3,
        unit_y_length: float = 2,
        wspace: float = .05
    ) -> tuple:
        fig, ax = plt.subplots(
            1, self.noise.size, 
            figsize=(unit_x_length * self.noise.size, unit_y_length), 
            sharey=True
        )
        plt.subplots_adjust(wspace=wspace)

        for t, sigma, a in zip(self.theoretical, self.noise.conf, ax.ravel()):
            t.plot(ax=a, jitter_ratio=.04, xscales=np.array(xscales), size=size, **kwarg_err)
            a.set(ylabel="", title="$\sigma=" + f"{sigma['kwarg_err']['scale']}" + "$")
        
        return fig, ax


    def plot_benchmarking(
        self,
        items: list = ["docloo"],
        cmap: dict = {"cloo": "C0", "docloo": "C0", "pb": "C1"},
        unit_x_length: float = 3,
        unit_y_length: float = 2,
        wspace: float = .05,
        noise: float = None
    ):
        df = self.scores_without_doptimization
        if noise is None:
            fig, ax = plt.subplots(
                1, self.noise.size, 
                figsize=(unit_x_length * self.noise.size, unit_y_length), 
                sharey=True
            )
            plt.subplots_adjust(wspace=wspace)

            for i, a in enumerate(ax.ravel()):
                e = self.scores.err.unique()[i]
                for k, des in self.metadata.designs.items():
                    sns.lineplot(
                        data=df[df.err == e], x=f"{k}_d", y=f"{k}_metric", 
                        ax=a, marker="s", markersize=6, 
                        color=cmap[k], err_style="bars", label=des().name,
                        errorbar=("ci", 95), err_kws={"capsize": 5}, 
                        n_boot=1000, seed=self.metadata.random_state
                    )
                for k in items:
                    sns.lineplot(
                        data=self.scores[self.scores.err == e], x=f"{k}_d", y=f"{k}_metric", marker="o", 
                        ax=a, label=self.conditions[k][0].metadata["design"], 
                        color=cmap[k], markersize=6, markeredgewidth=0, 
                        errorbar=("ci", 95), n_boot=1000, seed=self.metadata.random_state
                    )
                a.set_ylim(-0.05, 1.05)
                a.set_yticks(np.linspace(0, 1, 6).tolist())
                a.set(
                    title=self.noise.names[i], xlabel="D-criterion values", 
                    ylabel=f"{self.metric.name} ({self.metadata.simulator().name})"
                )
                a.set_xscale("log")
                a.legend(loc="upper right", fontsize="x-small")
                
                for k in items:
                    df_group = self.scores[self.scores.err == e].groupby(f"{k}_nmax").mean()
                    for x, y, t in zip(
                        list(df_group[f"{k}_d"]) + df[df.err == e].mean()[[f"{k2}_d" for k2 in self.metadata.designs]].tolist(),
                        list(df_group[f"{k}_metric"] - 0.1) + (df[df.err == e].mean()[[f"{k2}_metric" for k2 in self.metadata.designs]] - 0.1).tolist(),
                        list(df_group.index) + (df[df.err == e].mean()[[f"{k2}_nmax" for k2 in self.metadata.designs]] - 0.1).tolist()
                    ):
                        a.text(x, y, t, ha="center", va="center", size=8)

        else:
            fig, ax = plt.subplots(figsize=(unit_x_length, unit_y_length))
            
            for k, des in self.metadata.designs.items():
                sns.lineplot(
                    data=df[df.err == noise], x=f"{k}_d", y=f"{k}_metric", 
                    ax=ax, marker="s", markersize=6, 
                    color=cmap[k], err_style="bars", label=des().name,
                    errorbar=("ci", 95), err_kws={"capsize": 5}, 
                    n_boot=1000, seed=self.metadata.random_state
                )
            for k in items:
                sns.lineplot(
                    data=self.scores[self.scores.err == noise], x=f"{k}_d", y=f"{k}_metric", marker="o", 
                    ax=ax, label=self.conditions[k][0].metadata["design"], 
                    color=cmap[k], markersize=6, markeredgewidth=0, 
                    errorbar=("ci", 95), n_boot=1000, seed=self.metadata.random_state
                )
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks(np.linspace(0, 1, 6).tolist())
            ax.set(title=self.metadata.simulator().name, xlabel="D-criterion values", ylabel=self.metric.name)
            ax.set_xscale("log")
            ax.legend(loc="upper right", fontsize="x-small")

            for k in items:
                df_group = self.scores[self.scores.err == noise].groupby(f"{k}_nmax").mean()
                for x, y, t in zip(
                    list(df_group[f"{k}_d"]) + df[df.err == noise].mean()[[f"{k2}_d" for k2 in self.metadata.designs]].tolist(),
                    list(df_group[f"{k}_metric"] - 0.1) + (df[df.err == noise].mean()[[f"{k2}_metric" for k2 in self.metadata.designs]] - 0.1).tolist(),
                    list(df_group.index) + (df[df.err == noise].mean()[[f"{k2}_nmax" for k2 in self.metadata.designs]] - 0.1).tolist()
                ):
                    a.text(x, y, t, ha="center", va="center", size=8)
        
        return fig, ax

    
    def plot_power(
        self,
        items: list = ["docloo"],
        labels: dict = {"docloo": "DO-C+LOO"},
        unit_x_length: float = 3,
        unit_y_length: float = 2,
        wspace: float = .05,
        hspace: float = .05
    ):
        fig, ax = plt.subplots(len(items), self.noise.size, figsize=(3 * self.noise.size, 2 * len(items)), sharex=True, sharey=True)
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        
        bbox_to_anchor = (1, .5) if len(items) % 2 == 1 else (1, 1)
        center = np.floor(len(items) / 2).astype(int)

        for i, a in enumerate(ax.ravel()):
            sns.lineplot(
                data=self.power[
                    (self.power.noise == self.noise.names[i % self.noise.size]) & (self.power.model == labels[items[i // self.noise.size]])
                ].reset_index(), 
                x="n_max", y="power", hue="term",
                ax=a, palette=self.cmap, marker="o",
                markeredgewidth=0, n_boot=1000, seed=self.metadata.random_state,
            )

            a.set_ylim(-0.05, 1.05)
            a.set_xticks(self.power.n_max.unique().tolist())
            a.set_yticks(np.linspace(0, 1, 6).tolist())
            if i % self.noise.size == 0:
                a.set_ylabel(r"$1-\beta$" + f" ({labels[items[i // self.noise.size]]})")

            a.set_title(self.noise.names[i % self.noise.size]) if i // self.noise.size == 0 else a.set_xlabel("N: Num. of replication")
            a.legend(loc="center left", bbox_to_anchor=bbox_to_anchor ) if i == (1 + center) * self.noise.size - 1 else a.legend().remove()
        
        return fig, ax


class DOptimizationBenchmarkingPipeline(BenchmarkingPipeline):
    def __init__(
        self,
        from_pipeline: BenchmarkingPipeline,
        N: int,
        n_add: list,
        designs: dict = {"docloo": DOCLOO}
    ):
        self.configuration = {
            k: DOptimizationBenchmarker(
                from_benchmarker=bm, N=N, n_add=n_add, designs=designs
            ) for k, bm in tqdm(
                from_pipeline.configuration.items(), total=len(from_pipeline.configuration),
                desc="Running D-optimization pipeline"
            )
        }
        
        self.outputs = {k: {} for k in self.configuration}


    def plot(
        self,
        func: list = [
            "plot_groundtruth", "plot_benchmarking", "plot_power"
        ],
        kwargs: dict = {
            "plot_groundtruth": {}, 
            "plot_benchmarking": {}, 
            "plot_power": {}
        },
        savefig: bool = False,
        where: str = outputdir,
        titles: dict = {
            "plot_groundtruth": "groundtrue_results", 
            "plot_benchmarking": "benchmarks", 
            "plot_power": "power"
        },
        kwarg_save: dict = kwarg_savefig
    ):
        for f in tqdm(func, total=len(func), desc="Plotting"):
            for k, bm in tqdm(
                self.configuration.items(), total=len(self.configuration),
                desc=f
            ):
                fig, ax = eval(f"bm.{f}")(**kwargs[f])
                if savefig:
                    fig.savefig(f"{where}/{titles[f]}{bm.suffix}", **kwarg_save)
                self.outputs[k][f] = (fig, ax)