# Masterthesis: Numerical Simulation of Semilinear Parabolic SPDEs

This research project investigates the numerical analysis and simulation of semilinear parabolic stochastic partial differential equations (SPDEs), specifically one-dimensional semilinear parabolic SPDEs with additive white noise. The solution to such equations is an infinite-dimensional, continuous stochastic process.

Due to the infinite-dimensional nature of the solution and the high irregularity introduced by white noise, efficient discretization schemes are required for simulation.

The objective of this work is to develop an efficient algorithm by appropriately discretizing both the temporal and spatial dimensions in order to simulate the solution of these SPDEs.

---

## Core Methods

The central contribution of this work consists of two algorithms:

### 1. Semilinear Truncation Method

- The spatial domain is discretized via a spectral Galerkin method by projecting onto an `N`-dimensional subspace spanned by the first `N` eigenfunctions.
- The temporal domain is discretized using the exponential Euler method.
  - The time interval is partitioned into `M` equidistant steps.
  - An exponential term is incorporated to regularize the noise component.

It is demonstrated that the resulting fully discrete approximation converges to the infinite-dimensional solution of the SPDE. For suitable choices of `N` and `M`, a strong convergence rate of approximately `1/3` relative to the computational cost `K = M * N` is achieved.


### 2. Semilinear Replacement Method

This method extends the truncation approach:

- Rather than truncating higher spatial dimensions, they are replaced by independent, identically distributed, centered Gaussian random variables.
- This construction uses the representation of the solution in terms of Ornstein–Uhlenbeck processes.


## Numerical Evaluation

Both algorithms are mathematically defined and analyzed from a numerical perspective.

- **Programming language:** Python  
- **Methodology:** Monte Carlo simulations  
- **Evaluation criteria:** Quantitative measures including quadratic variation  

The numerical results indicate that:

- The **Replacement Method outperforms the Truncation Method**
- It yields **higher accuracy at identical computational cost**
- It requires **fewer spatial dimensions** to satisfy predefined error thresholds


## Key Takeaways

- Efficient discretization is essential for the simulation of SPDEs  
- Spectral methods combined with exponential integrators provide robust convergence properties  
- The stochastic replacement of higher dimensions can significantly enhance computational efficiency compared to the truncation method.

---

# Code Overview:

### Functions_used_for_Simulations.py

This module collects the core functions shared across the simulation scripts, including eigenvalues and eigenfunctions of the Laplacian, three nonlinearities, and their spectral approximations. 

It also implements the generic exponential Euler–Galerkin integrator (with and without reuse of reference noise) and the index construction for the semilinear replacement method, serving as a reusable toolbox for all numerical experiments in the thesis.

### Figure_1.py

Implements an exponential Euler-Galerkin scheme for a one-dimensional semilinear parabolic SPDE with three different nonlinearities (zero, a smooth bump, and a cubic term). It simulates three sample paths for each nonlinearity up to final time T=1 on the spatial interval `[0,1]`.

The script plots the resulting solution profiles at time `T=1` for all three nonlinearities side by side, illustrating how the terminal state depends on the chosen nonlinear drift.

### Figure_2-3.py

Uses the exponential Euler-Galerkin scheme to simulate the SPDE solution over time and space for the three nonlinearities and produces 3D surface plots of the solution `X(t,x)` on `[0,1]x[0,1]`.

Additionally, it generates multiple 1D snapshots at selected times, showing how the spatial profile of the solution evolves over time for each nonlinearity (path development plots).

### Figure_4.py

Provides the building blocks for the semilinear replacement method, including the index sets for the noise replacement in spectral space, and reuses the same eigenfunctions, eigenvalues, and nonlinearities as the exponential Euler code.

It is used to implement and compare the semilinear replacement scheme against the full exponential Euler-Galerkin method, focusing on how truncation and replacement of modes influence the numerical approximation error.

### Figure_5a.py

Studies strong convergence with respect to the spatial Galerkin dimension by computing a high-resolution exponential Euler reference solution (large `N_ref` and fixed time discretization) and re-simulating with smaller Galerkin dimensions `N` while reusing the same noise.

The script estimates the L2-strong error at the final time over multiple Monte Carlo runs for various `N` and produces a log-log plot of the strong error versus `N`, including a reference line corresponding to the theoretically expected convergence rate.

### Figure_5b.py

Analyzes strong convergence in the time discretization step by first constructing a fine temporal reference solution with the exponential Euler scheme and then building coarser approximations using a kappa-based aggregation of Brownian increments that preserves the underlying randomness.

It computes the strong error between the reference and each coarse scheme at the final time over many Monte Carlo samples and plots the error against the number of time steps `M` on a log-log scale, together with a theoretical reference rate.

### Figure_6.py

Compares methods in terms of computational cost by coupling the temporal step size and the Galerkin dimension so that `N` scales with `M`, and then using a kappa-parameter to define a family of approximations built from a fixed fine reference simulation.

The script evaluates the strong error for each kappa, translates the setup into an effective computational effort proportional to `M^3`, and plots strong error versus `M^3` on a log-log scale to illustrate the empirical complexity-accuracy trade-off.

### Figure_7-9.py

Investigates temporal quadratic variation of the SPDE solution at interior spatial points for two types of nonlinearities (a smooth bump or a cubic term, selected by commenting one of the definitions) using both the exponential Euler scheme and the semilinear replacement method.

It computes normalized realized temporal quadratic variations over many Monte Carlo paths and compares their empirical distributions to the corresponding Gaussian limit by plotting histograms together with the theoretical normal density.

### Figure_8-10.py

Focuses on the influence of the spatial truncation level `N` on the distribution of temporal quadratic variation, again for two nonlinearities (bump or cubic) and using the exponential Euler-Galerkin method.

For two different Galerkin dimensions (e.g. `N=500` and `N=1000`) and fixed temporal discretization, the script computes normalized temporal quadratic variations over many Monte Carlo samples and plots histograms versus the theoretical Gaussian density.

### Figure_11-13.py

This script investigates spatial quadratic variation of the SPDE solution for two nonlinearities (a smooth bump or a cubic term, selectable by commenting one definition) using both the exponential Euler scheme and the semilinear replacement method. 

It computes normalized realized spatial quadratic variations over many Monte Carlo samples and compares their empirical distributions to the corresponding Gaussian limit via histograms and normal density curves, corresponding to Figures 11 and 13.

### Figure_12-14.py

Analyzes spatial quadratic variation of the SPDE solution by defining a realized spatial quadratic variation functional along the spatial grid and combining it with exponential Euler and semilinear replacement simulations for different nonlinearities (bump or cubic, chosen via the nonlinearity definition).

The script generates Monte Carlo samples of normalized spatial quadratic variations and compares their empirical distributions to a Gaussian reference using histograms and normal density overlays.
