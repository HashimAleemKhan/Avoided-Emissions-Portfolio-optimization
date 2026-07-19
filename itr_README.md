# Avoided Emissions in the Portfolio Objective Function

An extension of Becquart, Giroux, Guyot & Peignon (2024), *"Implied
Temperature Rise & Avoided Emissions: A New Standard for Net-Zero Portfolio
Alignment?"* ([SSRN 4902499](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4902499)).

## Background

The paper shows that the Implied Temperature Rise (ITR) metric — used
by MSCI and widely adopted by investors to assess portfolio alignment with
climate scenarios — does not account for **avoided emissions** (companies
that provide low-carbon solutions to others). Because low-carbon solutions
are often themselves CO2-intensive to produce, this can penalize exactly
the companies most useful to a net-zero transition, causing capital
misallocation.

The paper's fix is to build a **corrected ITR metric** (their Eq. 6) that
folds each company's avoided/induced emissions ratio into the ITR formula,
and then re-runs the same constrained mean-variance optimization (portfolio
ITR ≤ threshold) using the corrected metric instead of the raw one.

## This project's contribution

Rather than correcting the constraint metric, this project puts avoided
emissions **directly into the mean-variance objective function**:

```
U(w) = w'μ − (γ/2) w'Σw + λ · w'ratio
```

where `ratio` is each company's avoided/induced emissions ratio and `λ`
controls how strongly the optimizer is rewarded for tilting toward
avoided-emissions-rich companies. Instead of satisfying a climate
threshold post-hoc, the optimizer trades off return, risk, and climate
contribution simultaneously.

The script implements both approaches side by side so they can be compared
directly:

1. **Paper's approach** — standard mean-variance objective, portfolio ITR
   used as a hard constraint, swept across thresholds (1.5°C to
   unconstrained)
2. **This project's approach** — avoided emissions rewarded directly in
   the objective, swept across reward weights (λ = 0 to 3)

For both, the script tracks: return, volatility, Sharpe ratio, standard
portfolio ITR (Eq. 1–2), the paper's corrected portfolio ITR (Eq. 6), and
achieved avoided/induced emissions exposure — plus the resulting sector
allocation.

## Key result

The ITR-constrained approach barely moves avoided-emissions exposure even
at a binding 1.5°C cap, because the raw ITR constraint isn't targeting
avoided emissions at all — it's an indirect, weak lever. The
objective-tilt approach achieves an order of magnitude more
avoided-emissions exposure for a comparable Sharpe ratio cost, because it
rewards the thing you actually want (avoided emissions) rather than
constraining a proxy metric that wasn't designed to capture it.

## Usage

```bash
pip install -r requirements.txt
python itr_avoided_emissions_optimization.py
```

## Data note

The original paper uses proprietary MSCI ITR and Carbon4 Finance
avoided-emissions data across ~8,800 companies. That data isn't publicly
available, so this script works on a **simulated cross-section of 60
companies**, calibrated so each sector's median ITR and avoided/induced
ratio match the paper's published Tables 3–4. The mechanics (ITR
aggregation, the corrected-ITR formula, the optimization) are implemented
exactly as specified in the paper; only the underlying company data is
synthetic. Swap in real data via `simulate_universe()` if you have access
to MSCI/Carbon4/Factset data.

## Notes / caveats

- The `λ · w'ratio` objective term is one reasonable formulation of
  "avoided emissions in the objective function," not necessarily the
  only one — the `climate_score` logic is isolated in
  `optimize_objective_tilt()` so a different scoring/penalty term can be
  swapped in without touching the rest of the pipeline.
- Long-only, capped-weight (10% max) mean-variance optimization via
  `scipy.optimize.minimize` (SLSQP) — no transaction costs, turnover
  constraints, or short positions.
- As in the paper, the "cost" of climate alignment shown here (Sharpe
  ratio decline) is a modeling result under a specific risk-aversion
  parameter (γ = 4) and simulated covariance structure; the qualitative
  comparison between the two approaches is the point, not the absolute
  Sharpe numbers.
