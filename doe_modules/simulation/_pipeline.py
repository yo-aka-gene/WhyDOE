import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm.notebook import tqdm
import warnings

from doe_modules.design import CLOO, PlackettBurman
from doe_modules.preferences import kwarg_err, outputdir, kwarg_savefig
from ._mlr import MLR
from ._theoretical_effects import TheoreticalEffects
from ._dunnett import Dunnett
from ._dunnett_power import dunnett_power
from ._anova_power import anova_power


def aptitude_score(res, gt):
    if gt.unique().size == 1:
        f = lambda res, gt: accuracy_score(res, gt)
    else:
        f = lambda res, gt: cohen_kappa_score(res, gt, weights="linear")
    return np.nan if res.isna().all() else f(res, gt)


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


class MetricConfigurator:
    def __init__(
        self,
        evaluation_metric,
        metric_name: str,
    ) -> None:
        self.f = evaluation_metric
        self.name = metric_name
        

class MetaDataConfigurator:
    def __init__(
        self, 
        n_range: list,
        n_rep: int,
        designs: dict, 
        edge_assignment: np.ndarray,
        random_state: int,
        seeds: np.ndarray,
        dunnett: bool
    ) -> None:
        self.n_range = n_range
        self.n_rep = n_rep
        self.designs = designs
        self.edge_assignment = edge_assignment
        self.random_state = random_state
        self.seeds = seeds
        self.dunnett = dunnett


class Benchmarker:
    def __init__(
        self,
        simulator,
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
            n_range=n_range,
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
                    np.arange(len(self.noise.conf)), 
                    n_range.size * seeds.size
                ).reshape(-1, len(self.noise.conf)).T.ravel(),
                total=self.noise.size * n_range.size * seeds.size,
                desc="generating ground truth labels for each condition"
            )
        ]
        
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
            
        
        warnings.simplefilter('ignore')
        
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
        unit_x_length: float = 3,
        unit_y_length: float = 2,
        wspace: float = .05
    ):
        fig, ax = plt.subplots(
            1, self.noise.size, 
            figsize=(unit_x_length * self.noise.size, unit_y_length), 
            sharey=True
        )
        plt.subplots_adjust(wspace=wspace)
        
        items = [
                item for item in items if (item != "pair") or self.metadata.dunnett
            ]
        cmap = [{"cloo": "C0", "pair": "C2", "pb": "C1"}[item] for item in items]
        
        for i, a in enumerate(ax.ravel()):
            e = self.scores.err.unique()[i]
            for k in items:
                sns.lineplot(
                    data=self.scores[self.scores.err == e], x="n", y=f"{k}_metric", marker="s", 
                    ax=a, 
                    label=self.conditions[k][0].metadata["design"] if k != "pair" else "Dunnett", 
                    markeredgewidth=0, errorbar=("ci", 95), n_boot=1000, seed=self.metadata.random_state,
                )
            a.set_ylim(-0.05, 1.05)
            a.set_xticks(self.metadata.n_range.tolist())
            a.set_yticks(np.linspace(0, 1, 6).tolist())
            a.set(
                title=self.noise.names[i], xlabel="N: Num. of replication", 
                ylabel=self.metric.name
            )
            a.legend(loc="best", fontsize="x-small", ncol=2)
        
        return fig, ax

    
    def plot_power(
        self,
        items: list = ["cloo", "pair", "pb"],
        unit_x_length: float = 3,
        unit_y_length: float = 2,
        wspace: float = .05,
        hspace: float = .05
    ):
        items = [
            {"cloo": "C+LOO", "pair": "Dunnett", "pb": "PB"}[k] for k in [
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
                markeredgewidth=0, n_boot=1000, seed=self.metadata.random_state,
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
        simulator,
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
            k: Benchmarker(**d, suffix=f"_{k}") for k, d in tqdm(
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
