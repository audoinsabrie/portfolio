import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

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

window = 750

rolling_mean = portfolio_returns.rolling(window).mean()
rolling_std = portfolio_returns.rolling(window).std()

z_95 = norm.ppf(0.95)
z_99 = norm.ppf(0.99)

var_parametric_95 = (
    -rolling_mean + z_95 * rolling_std
).shift(1)

var_parametric_99 = (
    -rolling_mean + z_99 * rolling_std
).shift(1)





window = 750

rolling_mean = portfolio_returns.rolling(window).mean()
rolling_std = portfolio_returns.rolling(window).std()

z_95 = norm.ppf(0.95)
z_99 = norm.ppf(0.99)

var_parametric_95 = (
    -rolling_mean + z_95 * rolling_std
).shift(1)

var_parametric_99 = (
    -rolling_mean + z_99 * rolling_std
).shift(1)

print("VaR paramétrique 95 % :")
print(var_parametric_95.tail())

print("\nVaR paramétrique 99 % :")
print(var_parametric_99.tail())


def kupiec_test(losses, var, alpha):

    valid = var.notna()
    exceptions = losses[valid] > var[valid]

    x = exceptions.sum()
    n = len(exceptions)
    taux = x / n

    p = 1 - alpha
    phat = x / n

    logL0 = (n - x) * np.log(1 - p) + x * np.log(p)
    logL1 = (n - x) * np.log(1 - phat) + x * np.log(phat)

    LR_uc = -2 * (logL0 - logL1)
    p_uc = 1 - chi2.cdf(LR_uc, 1)

    return x, n, taux, LR_uc, p_uc


def christoffersen_test(losses, var):

    valid = var.notna()
    exceptions = (losses[valid] > var[valid]).astype(int).values

    n00 = n01 = n10 = n11 = 0

    for i in range(1, len(exceptions)):
        if exceptions[i-1] == 0 and exceptions[i] == 0:
            n00 += 1
        elif exceptions[i-1] == 0 and exceptions[i] == 1:
            n01 += 1
        elif exceptions[i-1] == 1 and exceptions[i] == 0:
            n10 += 1
        else:
            n11 += 1

    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    logL0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)

    logL1 = (
        n00 * np.log(1 - pi0)
        + n01 * np.log(pi0)
        + n10 * np.log(1 - pi1)
        + n11 * np.log(pi1)
    )

    LR_ind = -2 * (logL0 - logL1)
    p_ind = 1 - chi2.cdf(LR_ind, 1)

    return n00, n01, n10, n11, LR_ind, p_ind


for alpha, var in [
    (0.95, var_parametric_95),
    (0.99, var_parametric_99)
]:

    x, n, taux, LR_uc, p_uc = kupiec_test(
        losses, var, alpha
    )

    n00, n01, n10, n11, LR_ind, p_ind = christoffersen_test(
        losses, var
    )

    LR_cc = LR_uc + LR_ind
    p_cc = 1 - chi2.cdf(LR_cc, 2)

    print(f"\nVaR paramétrique {alpha:.0%}")
    print(f"Exceptions : {x}")
    print(f"Observations : {n}")
    print(f"Taux : {taux:.2%}")
    print(f"Kupiec : {LR_uc:.4f} | p-value : {p_uc:.4f}")
    print(f"Christoffersen : {LR_ind:.4f} | p-value : {p_ind:.4f}")
    print(f"Couverture conditionnelle : {LR_cc:.4f} | p-value : {p_cc:.4f}")
    
    
valid = var_parametric_95.notna()

plt.figure(figsize=(12, 6))
plt.plot(losses[valid], label="Perte réalisée")
plt.plot(var_parametric_95[valid], label="VaR paramétrique 95 %")
plt.scatter(
    losses[valid][losses[valid] > var_parametric_95[valid]].index,
    losses[valid][losses[valid] > var_parametric_95[valid]],
    label="Exceptions"
)
plt.xlabel("Date")
plt.ylabel("Perte")
plt.title("VaR paramétrique normale à 95 %")
plt.legend()
plt.show()
