"""
Avoided Emissions in the Portfolio Objective Function
======================================================
Builds on Becquart, Giroux, Guyot & Peignon (2024), "Implied Temperature
Rise & Avoided Emissions: A New Standard for Net-Zero Portfolio Alignment?"

The paper shows that the standard Implied Temperature Rise (ITR) metric,
used as a *portfolio constraint*, does not account for avoided emissions
(companies providing low-carbon solutions), and proposes a corrected ITR
metric (their Eq. 6) that folds an avoided/induced emissions ratio into the
ITR formula itself, then re-runs constrained mean-variance optimization
with the corrected metric as the constraint.

This script implements:
  1. The paper's ITR mechanics: company-level ITR (Eq. 1), portfolio-level
     ITR aggregation (Eq. 2), and the avoided-emissions-corrected ITR (Eq. 6)
  2. The paper's approach: ITR used as a hard constraint on a standard
     mean-variance objective, swept across ITR thresholds
  3. An alternative approach: instead of correcting the constraint metric,
     avoided emissions are added as a reward term directly inside the
     mean-variance OBJECTIVE FUNCTION, so the optimizer trades off
     return, risk, and climate contribution simultaneously rather than
     satisfying a threshold post-hoc.

NOTE ON THE OBJECTIVE FUNCTION
-------------------------------
The exact functional form of "avoided emissions in the objective" is a
design choice. This script implements it as:

    U(w) = w'mu - (gamma/2) w'Sigma w + lambda * w'ratio

  where `ratio` is each company's avoided/induced emissions ratio (as used
  by the paper in Section 4.1) and lambda controls how strongly the
  optimizer is rewarded for tilting toward avoided-emissions-rich
  companies. This is one reasonable formulation, not necessarily the exact
  one you derived — swap in your own scoring/penalty term in
  `climate_score()` and re-run; everything downstream (frontier sweep,
  ITR/ratio tracking, sector plots) will work unchanged.

Because the underlying company-level MSCI/Carbon4 data used in the paper
is proprietary, this script works on a *simulated* cross-section of
companies whose ITR and avoided/induced ratio distributions are calibrated
to match the paper's published descriptive statistics (Tables 2-4), so the
mechanics and conclusions are representative even without the original
data. Swap in real data by replacing `simulate_universe()`.

Usage
-----
    python itr_avoided_emissions_optimization.py

Requirements
------------
    pip install numpy pandas matplotlib scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Config — constants taken directly from the paper
# ---------------------------------------------------------------------------
RNG_SEED = 7
N_ASSETS = 60
GLOBAL_BUDGET_GT = 1491          # GtCO2, Table 2
TCRE = 0.000545                  # degC per GtCO2, Table 2 / MSCI methodology
GLOBAL_BUDGET_X_TCRE = 0.81      # = GlobalBudget * TCRE, fixed by construction (Appendix 6.2)
ITR_FLOOR = 1.19                 # already-realised warming; ITR can't go below this (Eq. 6)

SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
]

# Target ITR by sector, loosely calibrated to Table 3 medians (degC)
SECTOR_ITR_MEDIAN = {
    "Financials": 1.55, "Communication Services": 1.65, "Health Care": 1.80,
    "Real Estate": 1.90, "Information Technology": 2.00, "Consumer Discretionary": 2.25,
    "Industrials": 2.30, "Consumer Staples": 2.40, "Utilities": 3.10,
    "Materials": 3.40, "Energy": 7.70,
}

# Target avoided/induced ratio by sector, loosely calibrated to Table 4 medians
SECTOR_RATIO_MEDIAN = {
    "Communication Services": 0.002, "Health Care": 0.00, "Consumer Staples": 0.006,
    "Energy": 0.007, "Financials": 0.04, "Information Technology": 0.02,
    "Materials": 0.009, "Industrials": 0.034, "Consumer Discretionary": 0.023,
    "Real Estate": 0.075, "Utilities": 0.113,
}


# ---------------------------------------------------------------------------
# 1. Simulated universe (swap for real MSCI/Carbon4/Factset data if you have it)
# ---------------------------------------------------------------------------
def simulate_universe(n_assets=N_ASSETS, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    sectors = rng.choice(SECTORS, size=n_assets)

    # --- Expected returns & covariance: simple one-factor + sector + idio model ---
    market_beta = rng.uniform(0.7, 1.4, n_assets)
    market_premium = 0.06
    sector_premium = rng.normal(0, 0.015, len(SECTORS))
    sector_premium_map = dict(zip(SECTORS, sector_premium))
    idio_premium = rng.normal(0, 0.02, n_assets)

    mu = market_beta * market_premium + np.array([sector_premium_map[s] for s in sectors]) + idio_premium
    mu = np.clip(mu, -0.05, 0.30)

    idio_vol = rng.uniform(0.15, 0.35, n_assets)
    market_vol = 0.16
    cov = np.outer(market_beta, market_beta) * market_vol ** 2
    cov += np.diag(idio_vol ** 2)

    # --- ITR: draw around sector median with dispersion, matching Table 3 shape ---
    itr = np.array([
        max(1.19, rng.lognormal(mean=np.log(SECTOR_ITR_MEDIAN[s]), sigma=0.35))
        for s in sectors
    ])
    itr = np.clip(itr, 1.19, 12.0)

    # --- Avoided/induced emissions ratio: heavy right tail, matching Table 4 ---
    ratio = np.array([
        rng.lognormal(mean=np.log(max(SECTOR_RATIO_MEDIAN[s], 1e-3)), sigma=1.1)
        for s in sectors
    ])
    ratio = np.clip(ratio, 0, 3.0)  # a few extreme outliers exist in the real data too

    # --- Budget intensity (arbitrary units, tCO2 per unit EVIC): drives how much
    #     weight a company's relative overshoot carries in portfolio ITR aggregation ---
    budget_intensity = rng.lognormal(mean=0, sigma=0.6, size=n_assets)

    # Back out relative overshoot implied by the drawn ITR (Eq. 1, inverted):
    # ITR = 2 + relative_overshoot * GLOBAL_BUDGET_X_TCRE
    relative_overshoot = (itr - 2.0) / GLOBAL_BUDGET_X_TCRE
    overshoot_intensity = relative_overshoot * budget_intensity

    df = pd.DataFrame({
        "sector": sectors,
        "mu": mu,
        "itr": itr,
        "ratio": ratio,
        "budget_intensity": budget_intensity,
        "overshoot_intensity": overshoot_intensity,
        "relative_overshoot": relative_overshoot,
    })
    return df, mu, cov


# ---------------------------------------------------------------------------
# 2. ITR mechanics from the paper
# ---------------------------------------------------------------------------
def portfolio_itr(weights, budget_intensity, overshoot_intensity):
    """Eq. 2: portfolio ITR as the ratio of weighted overshoot to weighted budget,
    NOT a simple weighted average of company ITRs (the aggregation is non-linear)."""
    weighted_overshoot = np.sum(weights * overshoot_intensity)
    weighted_budget = np.sum(weights * budget_intensity)
    if weighted_budget <= 0:
        return np.nan
    return 2.0 + (weighted_overshoot / weighted_budget) * GLOBAL_BUDGET_X_TCRE


def corrected_relative_overshoot(relative_overshoot, ratio):
    """Eq. 6: folds the avoided/induced emissions ratio into each company's
    relative overshoot before it feeds into the ITR aggregation."""
    sign = relative_overshoot >= 0
    corrected = np.where(sign, relative_overshoot * (1 - ratio), relative_overshoot * (1 + ratio))
    return corrected


def portfolio_corrected_itr(weights, budget_intensity, relative_overshoot, ratio):
    corrected_rel_overshoot = corrected_relative_overshoot(relative_overshoot, ratio)
    corrected_overshoot_intensity = corrected_rel_overshoot * budget_intensity
    itr = portfolio_itr(weights, budget_intensity, corrected_overshoot_intensity)
    return max(itr, ITR_FLOOR) if not np.isnan(itr) else np.nan


# ---------------------------------------------------------------------------
# 3. Portfolio optimization
# ---------------------------------------------------------------------------
def _weight_constraints(n_assets, w_cap=0.10):
    bounds = [(0.0, w_cap) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, constraints


def optimize_mean_variance(mu, cov, gamma=4.0, w_cap=0.10, x0=None):
    """Baseline: maximize w'mu - (gamma/2) w'Sigma w, long-only, capped weights."""
    n = len(mu)
    bounds, constraints = _weight_constraints(n, w_cap)
    x0 = x0 if x0 is not None else np.full(n, 1 / n)

    def neg_utility(w):
        return -(w @ mu - 0.5 * gamma * w @ cov @ w)

    res = minimize(neg_utility, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                    options={"maxiter": 500, "ftol": 1e-10})
    return res.x


def optimize_objective_tilt(mu, cov, ratio, gamma=4.0, lam=0.0, w_cap=0.10, x0=None):
    """
    Your approach: avoided emissions enter the OBJECTIVE directly.

        U(w) = w'mu - (gamma/2) w'Sigma w + lambda * w'ratio

    lambda = 0 recovers the plain mean-variance optimum.
    """
    n = len(mu)
    bounds, constraints = _weight_constraints(n, w_cap)
    x0 = x0 if x0 is not None else np.full(n, 1 / n)

    def neg_utility(w):
        return -(w @ mu - 0.5 * gamma * w @ cov @ w + lam * (w @ ratio))

    res = minimize(neg_utility, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                    options={"maxiter": 500, "ftol": 1e-10})
    return res.x


def optimize_itr_constrained(mu, cov, budget_intensity, overshoot_intensity, itr_cap,
                              gamma=4.0, w_cap=0.10, x0=None):
    """The paper's approach: standard mean-variance objective, ITR used as a
    hard constraint (portfolio ITR <= itr_cap)."""
    n = len(mu)
    bounds, base_constraints = _weight_constraints(n, w_cap)
    x0 = x0 if x0 is not None else np.full(n, 1 / n)

    def itr_constraint(w):
        # itr_cap - portfolio_itr(w) >= 0
        weighted_overshoot = np.sum(w * overshoot_intensity)
        weighted_budget = np.sum(w * budget_intensity)
        itr = 2.0 + (weighted_overshoot / weighted_budget) * GLOBAL_BUDGET_X_TCRE
        return itr_cap - itr

    constraints = base_constraints + [{"type": "ineq", "fun": itr_constraint}]

    def neg_utility(w):
        return -(w @ mu - 0.5 * gamma * w @ cov @ w)

    res = minimize(neg_utility, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                    options={"maxiter": 500, "ftol": 1e-10})
    return res.x


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def portfolio_metrics(w, mu, cov, df):
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    sharpe = ret / vol if vol > 0 else np.nan
    itr = portfolio_itr(w, df["budget_intensity"].values, df["overshoot_intensity"].values)
    itr_corr = portfolio_corrected_itr(
        w, df["budget_intensity"].values, df["relative_overshoot"].values, df["ratio"].values
    )
    avoided_exposure = w @ df["ratio"].values
    return {
        "return": ret, "vol": vol, "sharpe": sharpe,
        "itr": itr, "itr_corrected": itr_corr, "avoided_exposure": avoided_exposure,
    }


def sector_weights(w, df):
    s = pd.Series(w, index=df.index).groupby(df["sector"]).sum()
    return s.reindex(SECTORS).fillna(0)


# ---------------------------------------------------------------------------
# 5. Sweeps
# ---------------------------------------------------------------------------
def sweep_objective_tilt(mu, cov, df, lambdas, gamma=4.0):
    rows, weights_by_lambda = [], {}
    w_prev = None
    for lam in lambdas:
        w = optimize_objective_tilt(mu, cov, df["ratio"].values, gamma=gamma, lam=lam, x0=w_prev)
        w_prev = w
        m = portfolio_metrics(w, mu, cov, df)
        m["lambda"] = lam
        rows.append(m)
        weights_by_lambda[lam] = w
    return pd.DataFrame(rows), weights_by_lambda


def sweep_itr_constraint(mu, cov, df, itr_caps, gamma=4.0):
    rows, weights_by_cap = [], {}
    w_prev = None
    for cap in itr_caps:
        w = optimize_itr_constrained(
            mu, cov, df["budget_intensity"].values, df["overshoot_intensity"].values,
            itr_cap=cap, gamma=gamma, x0=w_prev,
        )
        w_prev = w
        m = portfolio_metrics(w, mu, cov, df)
        m["itr_cap"] = cap
        rows.append(m)
        weights_by_cap[cap] = w
    return pd.DataFrame(rows), weights_by_cap


# ---------------------------------------------------------------------------
# 6. Reporting & plots
# ---------------------------------------------------------------------------
def print_report(tilt_results, constraint_results):
    print("=" * 70)
    print("  OBJECTIVE-FUNCTION APPROACH (avoided emissions rewarded directly)")
    print("=" * 70)
    print(tilt_results[["lambda", "return", "vol", "sharpe", "itr", "itr_corrected",
                         "avoided_exposure"]].to_string(index=False, float_format="%.4f"))

    print("\n" + "=" * 70)
    print("  PAPER'S APPROACH (standard ITR used as a hard constraint)")
    print("=" * 70)
    print(constraint_results[["itr_cap", "return", "vol", "sharpe", "itr", "itr_corrected",
                               "avoided_exposure"]].to_string(index=False, float_format="%.4f"))


def plot_results(tilt_results, constraint_results, tilt_weights, df, path="itr_optimization_results.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (1) Efficient frontier under objective tilt, colored by lambda
    sc = axes[0, 0].scatter(tilt_results["vol"], tilt_results["return"],
                             c=tilt_results["lambda"], cmap="viridis", s=60)
    axes[0, 0].plot(tilt_results["vol"], tilt_results["return"], color="gray", alpha=0.4, linewidth=1)
    axes[0, 0].set_xlabel("Volatility")
    axes[0, 0].set_ylabel("Expected Return")
    axes[0, 0].set_title("Efficient Frontier as Avoided-Emissions Reward (λ) Increases")
    plt.colorbar(sc, ax=axes[0, 0], label="lambda")

    # (2) Portfolio ITR & avoided exposure vs lambda
    ax2 = axes[0, 1]
    ax2b = ax2.twinx()
    ax2.plot(tilt_results["lambda"], tilt_results["itr"], color="firebrick", marker="o", label="Portfolio ITR (standard)")
    ax2.plot(tilt_results["lambda"], tilt_results["itr_corrected"], color="darkorange", marker="s", label="Portfolio ITR (corrected, Eq.6)")
    ax2b.plot(tilt_results["lambda"], tilt_results["avoided_exposure"], color="seagreen", marker="^", label="Avoided/induced exposure")
    ax2.set_xlabel("lambda (avoided-emissions reward weight)")
    ax2.set_ylabel("Portfolio ITR (°C)")
    ax2b.set_ylabel("Avoided/induced emissions exposure")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax2.set_title("Climate Alignment Emerging from the Objective Tilt")

    # (3) Sector reallocation: lambda=0 vs highest lambda
    lam_lo, lam_hi = tilt_results["lambda"].iloc[0], tilt_results["lambda"].iloc[-1]
    w_lo, w_hi = tilt_weights[lam_lo], tilt_weights[lam_hi]
    sec_lo = sector_weights(w_lo, df)
    sec_hi = sector_weights(w_hi, df)
    x = np.arange(len(SECTORS))
    width = 0.35
    axes[1, 0].bar(x - width / 2, sec_lo.values, width, label=f"λ={lam_lo:g}")
    axes[1, 0].bar(x + width / 2, sec_hi.values, width, label=f"λ={lam_hi:g}")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(SECTORS, rotation=60, ha="right", fontsize=8)
    axes[1, 0].set_ylabel("Portfolio weight")
    axes[1, 0].set_title("Sector Reallocation: Objective Tilt (Your Approach)")
    axes[1, 0].legend()

    # (4) Sharpe ratio comparison: objective tilt vs ITR constraint, aligned by avoided exposure
    axes[1, 1].plot(tilt_results["avoided_exposure"], tilt_results["sharpe"],
                     marker="o", label="Objective-function tilt (yours)")
    axes[1, 1].plot(constraint_results["avoided_exposure"], constraint_results["sharpe"],
                     marker="s", label="ITR-constrained (paper's method)")
    axes[1, 1].set_xlabel("Avoided/induced emissions exposure achieved")
    axes[1, 1].set_ylabel("Sharpe ratio")
    axes[1, 1].set_title("Cost of Climate Alignment: Objective Tilt vs Hard Constraint")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nSaved chart to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df, mu, cov = simulate_universe()

    print("Universe summary (simulated, calibrated to paper's Tables 2-4):")
    print(f"  N assets           : {len(df)}")
    print(f"  ITR   mean/median  : {df['itr'].mean():.2f} / {df['itr'].median():.2f}  "
          f"(paper: 2.64 / 2.07)")
    print(f"  Ratio mean/median  : {df['ratio'].mean():.3f} / {df['ratio'].median():.3f}")

    gamma = 4.0
    lambdas = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0])
    itr_caps = np.array([1.5, 1.6, 1.7, 1.8, 2.0, 10.0])  # 10.0 ~ "unconstrained"

    print("\nRunning objective-function tilt sweep (your approach)...")
    tilt_results, tilt_weights = sweep_objective_tilt(mu, cov, df, lambdas, gamma=gamma)

    print("Running ITR-constrained sweep (paper's approach)...")
    constraint_results, constraint_weights = sweep_itr_constraint(mu, cov, df, itr_caps, gamma=gamma)

    print()
    print_report(tilt_results, constraint_results)

    print("\nGenerating plots...")
    plot_results(tilt_results, constraint_results, tilt_weights, df)
    print("\nDone.")


if __name__ == "__main__":
    main()
