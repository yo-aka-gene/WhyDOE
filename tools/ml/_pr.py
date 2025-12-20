import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


class MulticlassPR:
    def __init__(self, model, x, y):
        self.x = [
            precision_recall_curve(y[:, i], model.predict(x)[:, i])[1] for i in range(y.shape[1])
        ]
        self.y = [
            precision_recall_curve(y[:, i], model.predict(x)[:, i])[0] for i in range(y.shape[1])
        ]
        self.thresh = [
            precision_recall_curve(y[:, i], model.predict(x)[:, i])[2].tolist() for i in range(y.shape[1])
        ]
        self.ap = [
            average_precision_score(y[:, i], model.predict(x)[:, i]) for i in range(y.shape[1])
        ]
        self.base = [len(y[:, i][y[:, i] == 1]) / len(y) for i in range(y.shape[1])]


def plot_pr(model, x, y, ax, cmap, label_dict):
    assert len(ax.ravel()) >= 4, \
        f"At least 4 subplots are required; got {ax}"
    
    pr = MulticlassPR(model=model, x=x, y=y)
    ax = ax.ravel()
    
    for (i, x_), y_, ap, label, base in zip(enumerate(pr.x), pr.y, pr.ap, label_dict, pr.base):
        ax[i].plot(
            x_, y_, 
            label=f"{label} (AP:{ap.round(3)})", 
            c=eval(f"plt.cm.{cmap}")(i/(len(pr.ap) - 1)) if isinstance(cmap, str) else cmap[i]
        )
        ax[i].plot([0, 1], [base, base], c="gray", label="baseline", zorder=1, alpha=0.5)
    
    p_avg, r_avg, thr_avg = precision_recall_curve(y.ravel(), model.predict(x).ravel())
    
    ax[-1].plot(
        r_avg, p_avg, 
        c=".2", 
        label=f"micro (AP:{average_precision_score(y, model.predict(x)).round(2)})"
    )
    base = np.array(pr.base).mean()
    ax[-1].plot([0, 1], [base, base], c="gray", label="baseline", zorder=1, alpha=0.5)
    ax[-1].set_ylim(*ax[0].get_ylim())
    
    [sp.plot([0, 1], [1, 1], linestyle=(0, (1, 2)), c=".2", label="ideal", zorder=2) for sp in ax]
    [sp.set(xlabel="recall", ylabel="precision", title="PR curve (OvR)") for sp in ax] 
    ax[-1].set(title="PR curve")
    [sp.legend(fontsize="small") for sp in ax]


def plot_pr2(model, x, y, ax, cmap, label_dict, minimalist: bool = False):
    pr = MulticlassPR(model=model, x=x, y=y)
    
    for (i, x_), y_, ap, label, base in zip(enumerate(pr.x), pr.y, pr.ap, label_dict, pr.base):
        ax.plot(
            x_, y_, 
            label=f"{label} (AP:{ap.round(3)})", 
            c=eval(f"plt.cm.{cmap}")(i/(len(pr.ap) - 1)) if isinstance(cmap, str) else cmap[i]
        )
        ax.plot([0, 1], [base, base], c="gray", linestyle=(0, (1, 2)), zorder=1, alpha=0.5)
    
    p_avg, r_avg, thr_avg = precision_recall_curve(y.ravel(), model.predict(x).ravel())
    
    ax.plot(
        r_avg, p_avg, 
        c=".2", 
        label=f"micro (AP:{average_precision_score(y, model.predict(x)).round(2)})"
    )
    base = np.array(pr.base).mean()
    
    baseline_name = None
    if not minimalist:
        ax.plot([0, 1], [1, 1], linestyle=(0, (1, 2)), c=".2", label="ideal", zorder=2)
        baseline_name = "baselines"
    ax.plot([0, 1], [base, base], c="gray", linestyle=(0, (1, 2)), label=baseline_name, zorder=1, alpha=0.5)
    ax.set(xlabel="recall", ylabel="precision", title="PR curve (OvR)")
    ax.legend(fontsize="small")