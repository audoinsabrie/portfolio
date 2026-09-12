import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt


tickers = ["SPY", "EFA", "TLT", "GLD", "VNQ"]

data = yf.download(
    tickers,
    start="2018-01-01",
    end="2026-09-09",
    auto_adjust=True
)

prices = data["Close"].dropna()
returns = prices.pct_change().dropna()

weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

portfolio_returns = returns @ weights
losses = -portfolio_returns

window = 750
n_simulations = 10000

var_monte_carlo_95 = []
var_monte_carlo_99 = []
dates = []

rng = np.random.default_rng(42)

for i in range(window, len(returns)):
    historical_returns = returns.iloc[i - window:i]

    mean_vector = historical_returns.mean().to_numpy()
    covariance_matrix = historical_returns.cov().to_numpy()

    simulated_returns = rng.multivariate_normal(
        mean_vector,
        covariance_matrix,
        size=n_simulations
    )

    simulated_portfolio_returns = simulated_returns @ weights
    simulated_losses = -simulated_portfolio_returns

    var_monte_carlo_95.append(
        np.quantile(simulated_losses, 0.95)
    )

    var_monte_carlo_99.append(
        np.quantile(simulated_losses, 0.99)
    )

    dates.append(returns.index[i])


var_monte_carlo_95_series = pd.Series(
    var_monte_carlo_95,
    index=dates,
    name="VaR 95 %"
)

var_monte_carlo_99_series = pd.Series(
    var_monte_carlo_99,
    index=dates,
    name="VaR 99 %"
)


def aligned_data(losses, var):
    return pd.concat(
        [
            losses.rename("losses"),
            var.rename("var")
        ],
        axis=1
    ).dropna()


def kupiec_test(losses, var, confidence=0.95):
    data = aligned_data(losses, var)

    exceptions = (data["losses"] > data["var"]).astype(int)

    n = len(exceptions)
    x = int(exceptions.sum())

    expected_rate = 1 - confidence
    observed_rate = x / n

    if x == 0 or x == n:
        lr_uc = np.inf
        p_value = 0.0
    else:
        likelihood_null = (
            (1 - expected_rate) ** (n - x)
            * expected_rate ** x
        )

        likelihood_alternative = (
            (1 - observed_rate) ** (n - x)
            * observed_rate ** x
        )

        lr_uc = -2 * np.log(
            likelihood_null / likelihood_alternative
        )

        p_value = 1 - chi2.cdf(lr_uc, df=1)

    return {
        "exceptions": x,
        "observations": n,
        "exception_rate": observed_rate,
        "lr_uc": float(lr_uc),
        "p_value": float(p_value)
    }


def christoffersen_test(losses, var):
    data = aligned_data(losses, var)

    exceptions = (
        data["losses"] > data["var"]
    ).astype(int).to_numpy()

    n00 = 0
    n01 = 0
    n10 = 0
    n11 = 0

    for i in range(1, len(exceptions)):
        previous = exceptions[i - 1]
        current = exceptions[i]

        if previous == 0 and current == 0:
            n00 += 1
        elif previous == 0 and current == 1:
            n01 += 1
        elif previous == 1 and current == 0:
            n10 += 1
        elif previous == 1 and current == 1:
            n11 += 1

    total_0 = n00 + n01
    total_1 = n10 + n11

    pi_0 = n01 / total_0 if total_0 > 0 else 0
    pi_1 = n11 / total_1 if total_1 > 0 else 0
    pi = (n01 + n11) / (total_0 + total_1)

    if pi_0 in (0, 1) or pi_1 in (0, 1) or pi in (0, 1):
        lr_ind = 0.0
    else:
        likelihood_independent = (
            (1 - pi) ** (n00 + n10)
            * pi ** (n01 + n11)
        )

        likelihood_dependent = (
            (1 - pi_0) ** n00
            * pi_0 ** n01
            * (1 - pi_1) ** n10
            * pi_1 ** n11
        )

        lr_ind = -2 * np.log(
            likelihood_independent / likelihood_dependent
        )

    p_value = 1 - chi2.cdf(lr_ind, df=1)

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr_ind": float(lr_ind),
        "p_value": float(p_value)
    }


def conditional_coverage_test(losses, var, confidence=0.95):
    kupiec = kupiec_test(losses, var, confidence)
    christoffersen = christoffersen_test(losses, var)

    lr_cc = (
        kupiec["lr_uc"]
        + christoffersen["lr_ind"]
    )

    p_value = 1 - chi2.cdf(lr_cc, df=2)

    return {
        "lr_cc": float(lr_cc),
        "p_value": float(p_value)
    }


for confidence, var in [
    (0.95, var_monte_carlo_95_series),
    (0.99, var_monte_carlo_99_series)
]:
    kupiec = kupiec_test(losses, var, confidence)
    christoffersen = christoffersen_test(losses, var)
    conditional_coverage = conditional_coverage_test(
        losses,
        var,
        confidence
    )

    print(f"\nVaR Monte Carlo {confidence:.0%}")
    print(f"Exceptions : {kupiec['exceptions']}")
    print(f"Observations : {kupiec['observations']}")
    print(f"Taux d'exception : {kupiec['exception_rate']:.2%}")
    print(f"Kupiec : {kupiec}")
    print(f"Christoffersen : {christoffersen}")
    print(f"Couverture conditionnelle : {conditional_coverage}")


plot_data = aligned_data(
    losses,
    var_monte_carlo_95_series
)

exceptions = plot_data["losses"] > plot_data["var"]

plt.figure(figsize=(12, 5))

plt.plot(
    plot_data.index,
    plot_data["losses"],
    label="Pertes réalisées"
)

plt.plot(
    plot_data.index,
    plot_data["var"],
    label="VaR Monte Carlo 95 %"
)

plt.scatter(
    plot_data.index[exceptions],
    plot_data.loc[exceptions, "losses"],
    label="Exceptions",
    color="red"
)

plt.title("VaR Monte Carlo 95 % et pertes réalisées")
plt.xlabel("Date")
plt.ylabel("Perte quotidienne")
plt.legend()
plt.tight_layout()
plt.show()

