import matplotlib.pyplot as plt


sim1 = [plt.cm.rainbow(i/9) for i in range(9)]

nonlinear = [
    plt.cm.gist_rainbow_r((i + 1) /9) for i in range(8)
] + [
    plt.cm.gist_rainbow_r(0)
]

circuit = [plt.cm.Spectral_r(i/9) for i in range(9)]

sparse = [plt.cm.turbo((i + 1)/10) for i in range(9)]


__all__ = [
    sim1,
    nonlinear,
    circuit,
    sparse,
]