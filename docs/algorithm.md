# Algorithm & model

## The two-stage stochastic CFLP

Sets: facilities $I$, customers $J$, sampled scenarios $S$ with probabilities
$p_s$. First stage opens facilities $y_i \in \{0,1\}$. The recourse picks flows
$x_{ijs} \ge 0$ and unmet demand $u_{js} \ge 0$ per scenario.

$$
\min \; \sum_i f_i y_i + \sum_s p_s \Big( \sum_{ij} c_{ij} x_{ijs} + \sum_j q_j u_{js} \Big)
$$

subject to, for every scenario $s$:

$$
\sum_i x_{ijs} + u_{js} = d_{js}\;(\forall j), \qquad
\sum_j x_{ijs} \le s_i y_i\;(\forall i), \qquad
x_{ijs} \le d_{js}\, y_i\;(\forall i,j).
$$

The **disaggregated link** $x_{ijs} \le d_{js} y_i$ is redundant given the
capacity rows but tightens the LP relaxation sharply; it is also what makes
HiGHS' presolve solve the OR-Library instances to the published optimum (the weak
aggregated model triggers a presolve reduction that returns a suboptimal
incumbent labelled "optimal").

The unmet variable $u$ gives **relatively complete recourse**: every scenario is
feasible for any $y$ (worst case: serve nobody, pay the penalty $q_j$ per unit).

## Service-level chance constraint

A binary $z_s$ marks a scenario allowed to violate full service:

$$
u_{js} \le d_{js}\, z_s\;(\forall j,s), \qquad \sum_s p_s z_s \le \gamma .
$$

If $z_s = 0$ the scenario must be fully served. The budget caps the probability
mass of violating scenarios at the SAA risk level $\gamma$. For equal-weight
scenarios this is the classic count form $\sum_s z_s \le \lfloor \gamma N \rfloor$.

!!! warning "gamma is not epsilon"
    $\gamma$ is the **in-sample** SAA risk level; the **true** target is
    $\epsilon$ (each scenario served with probability $\ge 1 - \epsilon$). An SAA
    solution is safe for the true constraint only if $\gamma \le \epsilon$;
    $\gamma = 0$ is a guaranteed conservative inner approximation. The config
    keeps them as separate fields and enforces $\gamma \le \epsilon$.

## Benders / L-shaped decomposition

The master keeps $y$ (and $z$) plus an epigraph variable $\theta_s$ per scenario
(multi-cut), minimizing $\sum_i f_i y_i + \sum_s p_s \theta_s$. For a fixed
$(\hat y, \hat z)$ each scenario recourse is a small LP $Q_s(\hat y, \hat z_s)$.
By LP duality, one dual solution gives a cut valid for **all** $(y,z)$:

$$
\theta_s \ge \sum_j \pi_j d_{js} + \sum_i (\alpha_i s_i)\, y_i + \Big(\sum_j \delta_j d_{js}\Big) z_s,
$$

with $\pi$ (demand, free), $\alpha \le 0$ (capacity), $\delta \le 0$ (unmet link).
Relatively complete recourse means **optimality cuts only** — no feasibility
cuts. To keep every $z_s = 0$ subproblem feasible, the master also carries
$\sum_i s_i y_i \ge (\sum_j d_{js})(1 - z_s)$; since the serve graph is complete,
total capacity $\ge$ total demand suffices for full service.

### Pareto-optimal cuts

Facility-location recourse is degenerate: the dual subproblem has many optima, so
a plain cut may be weak. **Magnanti–Wong** picks the dual that is strongest at an
interior **core point**. We use the **Papadakos (2008)** independent-point
variant: generate the cut by solving the subproblem at the core point directly
(no optimality-tie equality), with the core point pulled halfway toward the
master solution each iteration. On degenerate instances this cuts iterations
substantially (e.g. 40 → 25 on a symmetric ring) for the identical optimum.

### Backends

All backends share the same plain-Python cut routine and differ only in search
strategy:

- **classic** — iterative loop on HiGHS. Solve master, solve $S$ subproblems, add
  violated cuts, repeat until the bound gap closes.
- **branch_and_cut** — single tree on SCIP via a constraint handler that
  separates cuts in `consenfolp`/`consenfops` and validates them in `conscheck`.
  SCIP's dual reductions must be disabled (they assume a complete constraint set
  and otherwise cut off the optimum with lazy cuts).
- **gurobi** — single tree on Gurobi; cuts injected with `cbLazy` from the
  integer-solution callback.

## Stochastic-quality measures

- **RP** — the stochastic (recourse) optimum.
- **WS** — wait-and-see: $\sum_s p_s$ times each scenario's perfect-foresight
  optimum. $WS \le RP$.
- **EEV** — cost of the mean-value first stage evaluated over the real scenarios.
  $EEV \ge RP$.
- **EVPI** $= RP - WS$ — value of perfect information.
- **VSS** $= EEV - RP$ — value of the stochastic solution over the mean-value one.
- **SAA gap** — a lower bound from the mean of several SAA replications and an
  upper bound from evaluating one candidate on a large reference sample.

## References

- Santoso, Ahmed, Goetschalckx, Shapiro (2005), *EJOR*.
- Birge & Louveaux (2011), *Introduction to Stochastic Programming*.
- Kleywegt, Shapiro, Homem-de-Mello (2002); Mak, Morton, Wood (1999).
- Luedtke & Ahmed (2008); Pagnoncelli, Ahmed, Shapiro (2009).
- Magnanti & Wong (1981); Papadakos (2008).
- Rahmaniani, Crainic, Gendreau, Rei (2017).
