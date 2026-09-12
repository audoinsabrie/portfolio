import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import gaussian_kde


tickers = ["SPY", "EFA", "TLT", "GLD", "VNQ"]

data = yf.download(
    tickers,
    start="2018-01-01",
    end="2026-09-09",
    auto_adjust=True
)


prices = data["Close"]

returns = prices.pct_change().dropna()

weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

portfolio_returns = returns @ weights

losses = -portfolio_returns


def kde_var(losses, confidence=0.95):

    kde = gaussian_kde(losses)

    x = np.linspace(
        losses.min() - 4 * losses.std(),
        losses.max() + 4 * losses.std(),
        10000
    )

    density = kde(x)

    dx = x[1] - x[0]

    cdf = np.cumsum(density) * dx

    cdf /= cdf[-1]

    var = np.interp(confidence, cdf, x)

    return var


window = 750

var_kde_95 = pd.Series(
    index=losses.index,
    dtype=float
)

var_kde_99 = pd.Series(
    index=losses.index,
    dtype=float
)


for i in range(window, len(losses)):

    historical_losses = losses.iloc[i-window:i]

    var_kde_95.iloc[i] = kde_var(
        historical_losses,
        confidence=0.95
    )

    var_kde_99.iloc[i] = kde_var(
        historical_losses,
        confidence=0.99
    )


backtest_kde_95 = pd.DataFrame({
    "loss": losses,
    "VaR": var_kde_95
}).dropna()

backtest_kde_95["exception"] = (
    backtest_kde_95["loss"] >
    backtest_kde_95["VaR"]
)


backtest_kde_99 = pd.DataFrame({
    "loss": losses,
    "VaR": var_kde_99
}).dropna()

backtest_kde_99["exception"] = (
    backtest_kde_99["loss"] >
    backtest_kde_99["VaR"]
)


print()
print("KDE VaR 95%")
print("--------------------")
print(
    "Exceptions :",
    backtest_kde_95["exception"].sum()
)
print(
    "Observations :",
    len(backtest_kde_95)
)
print(
    "Taux d'exception :",
    backtest_kde_95["exception"].mean()
)
print(
    "Taux attendu :",
    1 - 0.95
)


print()
print("KDE VaR 99%")
print("--------------------")
print(
    "Exceptions :",
    backtest_kde_99["exception"].sum()
)
print(
    "Observations :",
    len(backtest_kde_99)
)
print(
    "Taux d'exception :",
    backtest_kde_99["exception"].mean()
)
print(
    "Taux attendu :",
    1 - 0.99
)


plt.figure(figsize=(12, 6))

plt.plot(
    backtest_kde_95.index,
    backtest_kde_95["loss"],
    label="Perte réalisée"
)

plt.plot(
    backtest_kde_95.index,
    backtest_kde_95["VaR"],
    label="VaR KDE 95%"
)

exceptions = backtest_kde_95["exception"]

plt.scatter(
    backtest_kde_95.index[exceptions],
    backtest_kde_95.loc[exceptions, "loss"],
    label="Exceptions"
)

plt.xlabel("Date")
plt.ylabel("Perte")
plt.title("VaR KDE 95% et pertes réalisées")
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))

plt.plot(
    backtest_kde_99.index,
    backtest_kde_99["loss"],
    label="Perte réalisée"
)

plt.plot(
    backtest_kde_99.index,
    backtest_kde_99["VaR"],
    label="VaR KDE 99%"
)

exceptions = backtest_kde_99["exception"]

plt.scatter(
    backtest_kde_99.index[exceptions],
    backtest_kde_99.loc[exceptions, "loss"],
    label="Exceptions"
)

plt.xlabel("Date")
plt.ylabel("Perte")
plt.title("VaR KDE 99% et pertes réalisées")
plt.legend()
plt.show()


from scipy.stats import chi2


def kupiec_test(exceptions, confidence):
    
    n = len(exceptions)
    x = np.sum(exceptions)
    
    p = 1 - confidence
    
    if x == 0:
        lr_uc = -2 * (
            n * np.log(1 - p)
        )
    elif x == n:
        lr_uc = -2 * (
            n * np.log(p)
        )
    else:
        pi_hat = x / n
        
        lr_uc = -2 * (
            (n - x) * np.log((1 - p) / (1 - pi_hat))
            + x * np.log(p / pi_hat)
        )
    
    p_value = 1 - chi2.cdf(lr_uc, df=1)
    
    return {
        "lr_uc": lr_uc,
        "p_value": p_value
    }


def christoffersen_test(exceptions):
    
    exceptions = np.asarray(exceptions, dtype=int)
    
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
    
    n0 = n00 + n01
    n1 = n10 + n11
    
    pi01 = n01 / n0 if n0 > 0 else 0
    pi11 = n11 / n1 if n1 > 0 else 0
    
    total_exceptions = n01 + n11
    total_transitions = n00 + n01 + n10 + n11
    
    pi = (
        total_exceptions / total_transitions
        if total_transitions > 0
        else 0
    )
    
    log_likelihood_restricted = 0
    log_likelihood_unrestricted = 0
    
    if pi > 0 and pi < 1:
        log_likelihood_restricted = (
            (n00 + n10) * np.log(1 - pi)
            + (n01 + n11) * np.log(pi)
        )
    
    if pi01 > 0 and pi01 < 1:
        log_likelihood_unrestricted += (
            n00 * np.log(1 - pi01)
            + n01 * np.log(pi01)
        )
    
    if pi11 > 0 and pi11 < 1:
        log_likelihood_unrestricted += (
            n10 * np.log(1 - pi11)
            + n11 * np.log(pi11)
        )
    
    lr_ind = -2 * (
        log_likelihood_restricted
        - log_likelihood_unrestricted
    )
    
    p_value = 1 - chi2.cdf(lr_ind, df=1)
    
    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr_ind": lr_ind,
        "p_value": p_value
    }


def conditional_coverage_test(kupiec_result, christoffersen_result):
    
    lr_cc = (
        kupiec_result["lr_uc"]
        + christoffersen_result["lr_ind"]
    )
    
    p_value = 1 - chi2.cdf(lr_cc, df=2)
    
    return {
        "lr_cc": lr_cc,
        "p_value": p_value
    }


exceptions_kde_95 = (
    backtest_kde_95["exception"]
    .astype(int)
    .values
)

exceptions_kde_99 = (
    backtest_kde_99["exception"]
    .astype(int)
    .values
)


kupiec_kde_95 = kupiec_test(
    exceptions_kde_95,
    confidence=0.95
)

christoffersen_kde_95 = christoffersen_test(
    exceptions_kde_95
)

conditional_kde_95 = conditional_coverage_test(
    kupiec_kde_95,
    christoffersen_kde_95
)


kupiec_kde_99 = kupiec_test(
    exceptions_kde_99,
    confidence=0.99
)

christoffersen_kde_99 = christoffersen_test(
    exceptions_kde_99
)

conditional_kde_99 = conditional_coverage_test(
    kupiec_kde_99,
    christoffersen_kde_99
)


print()
print("========================================")
print("KDE - BACKTESTING 95%")
print("========================================")

print()
print("Kupiec - Unconditional Coverage")
print("LR :", kupiec_kde_95["lr_uc"])
print("p-value :", kupiec_kde_95["p_value"])

print()
print("Christoffersen - Independence")
print("n00 :", christoffersen_kde_95["n00"])
print("n01 :", christoffersen_kde_95["n01"])
print("n10 :", christoffersen_kde_95["n10"])
print("n11 :", christoffersen_kde_95["n11"])
print("LR :", christoffersen_kde_95["lr_ind"])
print("p-value :", christoffersen_kde_95["p_value"])

print()
print("Conditional Coverage")
print("LR :", conditional_kde_95["lr_cc"])
print("p-value :", conditional_kde_95["p_value"])


print()
print("========================================")
print("KDE - BACKTESTING 99%")
print("========================================")

print()
print("Kupiec - Unconditional Coverage")
print("LR :", kupiec_kde_99["lr_uc"])
print("p-value :", kupiec_kde_99["p_value"])

print()
print("Christoffersen - Independence")
print("n00 :", christoffersen_kde_99["n00"])
print("n01 :", christoffersen_kde_99["n01"])
print("n10 :", christoffersen_kde_99["n10"])
print("n11 :", christoffersen_kde_99["n11"])
print("LR :", christoffersen_kde_99["lr_ind"])
print("p-value :", christoffersen_kde_99["p_value"])

print()
print("Conditional Coverage")
print("LR :", conditional_kde_99["lr_cc"])
print("p-value :", conditional_kde_99["p_value"])
