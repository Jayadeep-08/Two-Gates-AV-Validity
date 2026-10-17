# Two-Gates Validity for Simulation-Based Autonomous Vehicle Safety Assessment

This repository contains the source code, experimental scripts, and research figures for two complementary studies on validating safety conclusions obtained from autonomous vehicle (AV) simulations.

The project focuses on two fundamental questions:

1. **Construct Validity:** Does the chosen risk metric measure the phenomenon it is intended to represent?
2. **Uncertainty Validity:** Does the uncertainty region used in the analysis achieve its claimed statistical coverage?

The two studies examine these questions separately and connect them through **Bridge B**, which evaluates whether the propagation result remains robust when calibration uncertainty is taken into account.

## Research Overview

### Study 1 — Construct Validity and Risk Propagation

Study 1 investigates whether a state-based risk metric can provide evidence of history-dependent risk propagation in sequential driving scenarios.

The analysis examines braking and lane-change scenarios, state persistence, conditional relationships across decisions, and multi-agent propagation.

Key research questions:

- Can a state-based risk score distinguish genuine history-dependent effects from effects explained by the current state?
- How do associations change when intervening states are considered?
- Does a measured association alone establish that risk propagates through a chain of vehicles?

The findings highlight the importance of distinguishing state persistence and statistical association from evidence of additional history-dependent propagation.

### Study 2 — Coverage-Validated Calibration

Study 2 examines uncertainty in the calibration of the Intelligent Driver Model (IDM) using real driving episodes from the highD dataset.

The study evaluates empirical coverage under heterogeneous conditions, corrects an under-covered uncertainty region, and propagates the corrected region through safety simulations.

Key research questions:

- Does a nominal 95% calibration region actually achieve 95% coverage?
- How does calibration uncertainty affect estimated collision rates and policy effect sizes?
- Do controller rankings remain consistent across the evaluated parameter settings?

The analysis demonstrates why uncertainty calibration must be validated before interpreting absolute simulation outcomes. A ranking can remain stable even when the estimated effect size and absolute collision rate are uncertain.

### Bridge B — Connecting Both Studies

Bridge B connects the construct-validity analysis in Study 1 with the coverage-validated calibration in Study 2.

It evaluates the risk-propagation quantity over the validated IDM parameter region rather than relying only on a single nominal parameter estimate.

The evaluated interval was:

\[
A(\theta) \in [-0.442,\,+0.011]\ \text{m}
\]

The worst harmful endpoint was **+0.011 m**, compared with a predefined practical threshold of **0.36 m**.

Within the tested scenarios and validated calibration region, the worst harmful propagation effect remained below the practical threshold. This is a robustness result for the specific experiment, not a universal claim that autonomous vehicles are safe or that risk propagation cannot occur in other conditions.

## Repository Structure

```text
Two-Gates-AV-Validity/
├── analysis/                 # Study 1 analysis scripts and supporting material
├── chain_risk/               # Chain-risk analysis scripts
├── scripts/                  # Study 1 experimental and analysis scripts
├── tiv_revision/             # Study 1 numerical and figure-generation code
├── study1/
│   └── figures/              # Study 1 research figures
└── study2/
    ├── code/
    │   └── src/              # IDM, likelihood, bridge and verdict modules
    ├── scripts/              # Study 2 experimental and analysis scripts
    └── figures/              # Study 2 research figures
```

The exact contents of the figure directories depend on which generated PNG files have been added to the repository.

## Technologies

- Python
- Statistical analysis and numerical experimentation
- Intelligent Driver Model (IDM)
- Real-world highway trajectory data from highD
- Simulation-based safety evaluation
- Empirical coverage validation
- Parameter uncertainty propagation
- Reproducible experiment and figure-generation scripts

## Main Findings

- A state-based metric must be evaluated for construct validity before it is used to support claims about history-dependent effects.
- Nominal uncertainty coverage should not be assumed; it must be checked empirically.
- Under-covered calibration regions can underestimate uncertainty in simulated safety outcomes.
- Controller rankings, effect sizes, and absolute collision-rate estimates have different levels of robustness.
- Bridge B tests whether the specific propagation result remains below a predefined practical threshold across the validated calibration region.

## Reproducibility and Data

The repository contains selected source code and experimental scripts. Running the experiments may require external datasets, additional Python packages, and suitable simulation or computing environments.

The highD dataset and any other externally obtained driving data should be acquired from their respective authorized sources. Dataset files and generated experimental outputs are not necessarily included in this repository.

Before running an experiment, inspect the relevant script for required input paths, command-line arguments, dependencies, and output locations. Results may depend on the dataset version, parameter settings, random seeds, and execution environment.

## Scope and Limitations

The findings are limited to the scenarios, data, model assumptions, parameter regions, and evaluation procedures used in the studies.

The results do not establish universal AV safety, eliminate all possible history-dependent effects, or account for every source of model uncertainty. In particular, calibration uncertainty and model-class uncertainty are distinct.

## Repository

**GitHub:** https://github.com/Jayadeep-08/Two-Gates-AV-Validity

