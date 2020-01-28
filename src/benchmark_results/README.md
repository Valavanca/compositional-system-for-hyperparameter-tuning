# Benchmark for solving multi-objective problems


## Problems suites for evaluations

An evaluation will use well-established test suites for multi-objective optimization and estimate the results and performance of the surrogate-based approach.

|                                                       Test suite | Param dim | Problems ID | Objectives | Notes                                                                                                           |                                                                                                                                                                                                 Reference |
|-----------------------------------------------------------------:|-----------|-------------|------------|-----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   [ZDT](https://esa.github.io/pagmo2/docs/cpp/problems/zdt.html) | 2>        | 1-6         | 2          | scalable parameter dimensions                                                                                   |         Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele.  “Comparison of multiobjective evolutionary algorithms: Empirical results.”  Evolutionary computation 8.2 (2000): 173-195. doi: 10.1.1.30.5848 |
|   [WFG](https://esa.github.io/pagmo2/docs/cpp/problems/wfg.html) | 2>        | 1-9         | 2>         | non-separable problems, deceptive problems, truly degenerative problems and mixed shape Pareto front problems | Huband, Simon, Hingston, Philip, Barone, Luigi and While Lyndon.  “A Review of Multi-Objective Test Problems and a Scalable Test Problem Toolkit”.  IEEE Transactions on Evolutionary Computation (2006), |
| [DTLZ](https://esa.github.io/pagmo2/docs/cpp/problems/dtlz.html) | 3>        | 1-7         | 2>         | scalable objective dimensions. The dimension of the parameter space is 𝑘 + 𝑓𝑑𝑖𝑚 − 1                     |                                                                                          K. Deb, L. Thiele, M. Laumanns, E. Zitzler, Scalable test problems for evolutionary  multiobjective optimization |

___

## Solving Approaches

### 1. Random search

> [Problem] -> [Random sampling] -> [Non-dominated sorting]

Cases:
- Number of evaluation
- Sampling plan

### 2. Multi-objective Evolutionary Algorithms (MOEA)

> [Problem] -> [MOEA]

Population size is static.

Cases:
- Algorithm (NSGA2, MOEAD, MACO, NSPSO)
- Evaluation budget

### 3. Multi-output Surrogate with Multi-objective Evolutionary Algorithms (MOEA)

> [Problem] -> [Surrogate] -> [MOEA]

Cases:
- Size of a sample set
- Surrogate model
- Algorithm (NSGA2, MOEAD, MACO, NSPSO)
