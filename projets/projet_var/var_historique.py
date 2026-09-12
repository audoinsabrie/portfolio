import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import chi2


tickers = ["SPY", "EFA", "TLT", "GLD", "VNQ"]

data = yf.download(
    tickers,
    start="2018-01-01",
    end="2026-09-09",
    auto_adjust=True
)

print(data.head())

print("\nDimensions :")
print(data.shape)

print("\nInformations :")
print(data.info())

print("\nValeurs manquantes :")
print(data.isna().sum())


prices = data["Close"]

print("\nPrix de clôture :")
print(prices.head())

print("\nDimensions :")
print(prices.shape)


returns = prices.pct_change().dropna()

weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

portfolio_returns = returns @ weights

losses = -portfolio_returns

print("\nStatistiques des pertes :")
print(losses.describe())


plt.hist(portfolio_returns, bins=50)
plt.xlabel("Rendement quotidien")
plt.ylabel("Fréquence")
plt.title("Distribution des rendements du portefeuille")
plt.show()


plt.hist(losses, bins=50)
plt.xlabel("Perte quotidienne")
plt.ylabel("Fréquence")
plt.title("Distribution des pertes du portefeuille")
plt.show()


print("\n--- Statistiques des pertes ---")

print("Moyenne :", losses.mean())
print("Écart-type :", losses.std())
print("Asymétrie :", losses.skew())
print("Kurtosis :", losses.kurtosis())

print("\nQuantiles :")

print(
    "VaR empirique 95% :",
    losses.quantile(0.95)
)

print(
    "VaR empirique 99% :",
    losses.quantile(0.99)
)

print("\nPertes extrêmes :")

print(
    "Perte maximale :",
    losses.max()
)

print(
    "Perte minimale :",
    losses.min()
)


sm.qqplot(losses, line="q")

plt.title("QQ-plot des pertes quotidiennes")
plt.show()


window = 750


var_historical_95 = (
    losses
    .rolling(window)
    .quantile(0.95)
    .shift(1)
)

var_historical_99 = (
    losses
    .rolling(window)
    .quantile(0.99)
    .shift(1)
)


print("\nVaR historique 95% :")
print(var_historical_95.tail())

print("\nVaR historique 99% :")
print(var_historical_99.tail())


plt.figure(figsize=(12, 6))

plt.plot(
    var_historical_95,
    label="VaR historique 95%"
)

plt.plot(
    var_historical_99,
    label="VaR historique 99%"
)

plt.xlabel("Date")
plt.ylabel("VaR")
plt.title("VaR historique du portefeuille")
plt.legend()

plt.show()


def kupiec_test(losses, var, alpha):

    valid = var.notna()

    losses_valid = losses[valid]
    var_valid = var[valid]

    exceptions = losses_valid > var_valid

    x = exceptions.sum()
    n = len(exceptions)

    p = 1 - alpha
    phat = x / n

    logL0 = (
        (n - x) * np.log(1 - p)
        + x * np.log(p)
    )

    logL1 = (
        (n - x) * np.log(1 - phat)
        + x * np.log(phat)
    )

    LR_uc = -2 * (logL0 - logL1)

    p_value = 1 - chi2.cdf(
        LR_uc,
        df=1
    )

    return x, n, phat, LR_uc, p_value


def christoffersen_test(losses, var):

    valid = var.notna()

    exceptions = (
        losses[valid] > var[valid]
    ).astype(int).values

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

    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)

    pi = (
        n01 + n11
    ) / (
        n00 + n01 + n10 + n11
    )

    logL0 = (
        (n00 + n10) * np.log(1 - pi)
        + (n01 + n11) * np.log(pi)
    )

    logL1 = (
        n00 * np.log(1 - pi0)
        + n01 * np.log(pi0)
        + n10 * np.log(1 - pi1)
        + n11 * np.log(pi1)
    )

    LR_ind = -2 * (logL0 - logL1)

    p_value = 1 - chi2.cdf(
        LR_ind,
        df=1
    )

    return (
        n00,
        n01,
        n10,
        n11,
        LR_ind,
        p_value
    )


def backtesting(losses, var, alpha):

    x, n, phat, LR_uc, p_uc = kupiec_test(
        losses,
        var,
        alpha
    )

    (
        n00,
        n01,
        n10,
        n11,
        LR_ind,
        p_ind
    ) = christoffersen_test(
        losses,
        var
    )

    LR_cc = LR_uc + LR_ind

    p_cc = 1 - chi2.cdf(
        LR_cc,
        df=2
    )

    return {
        "exceptions": x,
        "observations": n,
        "exception_rate": phat,
        "LR_uc": LR_uc,
        "p_uc": p_uc,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "LR_ind": LR_ind,
        "p_ind": p_ind,
        "LR_cc": LR_cc,
        "p_cc": p_cc
    }


result_95 = backtesting(
    losses,
    var_historical_95,
    0.95
)

result_99 = backtesting(
    losses,
    var_historical_99,
    0.99
)


print("\n========================================")
print("HISTORICAL - BACKTESTING 95%")
print("========================================")

print(
    "\nExceptions :",
    result_95["exceptions"]
)

print(
    "Observations :",
    result_95["observations"]
)

print(
    "Taux d'exception :",
    result_95["exception_rate"]
)

print(
    "Taux attendu :",
    0.05
)

print("\nKupiec - Unconditional Coverage")

print(
    "LR :",
    result_95["LR_uc"]
)

print(
    "p-value :",
    result_95["p_uc"]
)

print("\nChristoffersen - Independence")

print(
    "n00 :",
    result_95["n00"]
)

print(
    "n01 :",
    result_95["n01"]
)

print(
    "n10 :",
    result_95["n10"]
)

print(
    "n11 :",
    result_95["n11"]
)

print(
    "LR :",
    result_95["LR_ind"]
)

print(
    "p-value :",
    result_95["p_ind"]
)

print("\nConditional Coverage")

print(
    "LR :",
    result_95["LR_cc"]
)

print(
    "p-value :",
    result_95["p_cc"]
)


print("\n========================================")
print("HISTORICAL - BACKTESTING 99%")
print("========================================")

print(
    "\nExceptions :",
    result_99["exceptions"]
)

print(
    "Observations :",
    result_99["observations"]
)

print(
    "Taux d'exception :",
    result_99["exception_rate"]
)

print(
    "Taux attendu :",
    0.01
)

print("\nKupiec - Unconditional Coverage")

print(
    "LR :",
    result_99["LR_uc"]
)

print(
    "p-value :",
    result_99["p_uc"]
)

print("\nChristoffersen - Independence")

print(
    "n00 :",
    result_99["n00"]
)

print(
    "n01 :",
    result_99["n01"]
)

print(
    "n10 :",
    result_99["n10"]
)

print(
    "n11 :",
    result_99["n11"]
)

print(
    "LR :",
    result_99["LR_ind"]
)

print(
    "p-value :",
    result_99["p_ind"]
)

print("\nConditional Coverage")

print(
    "LR :",
    result_99["LR_cc"]
)

print(
    "p-value :",
    result_99["p_cc"]
)

