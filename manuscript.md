---
author-meta:
- Oleksandr Husak
date-meta: '2020-02-25'
header-includes: '<!--

  Manubot generated metadata rendered from header-includes-template.html.

  Suggest improvements at https://github.com/manubot/manubot/blob/master/manubot/process/header-includes-template.html

  -->

  <meta name="dc.format" content="text/html" />

  <meta name="dc.title" content="Compositional Multi-objective Parameter tuning" />

  <meta name="citation_title" content="Compositional Multi-objective Parameter tuning" />

  <meta property="og:title" content="Compositional Multi-objective Parameter tuning" />

  <meta property="twitter:title" content="Compositional Multi-objective Parameter tuning" />

  <meta name="dc.date" content="2020-02-25" />

  <meta name="citation_publication_date" content="2020-02-25" />

  <meta name="dc.language" content="en-US" />

  <meta name="citation_language" content="en-US" />

  <meta name="dc.relation.ispartof" content="Manubot" />

  <meta name="dc.publisher" content="Manubot" />

  <meta name="citation_journal_title" content="Manubot" />

  <meta name="citation_technical_report_institution" content="Manubot" />

  <meta name="citation_author" content="Oleksandr Husak" />

  <meta name="citation_author_institution" content="Research Student Assistant in TUD" />

  <meta name="citation_author_orcid" content="XXXX-XXXX-XXXX-XXXX" />

  <link rel="canonical" href="https://Valavanca.github.io/compositional-system-for-hyperparameter-tuning/" />

  <meta property="og:url" content="https://Valavanca.github.io/compositional-system-for-hyperparameter-tuning/" />

  <meta property="twitter:url" content="https://Valavanca.github.io/compositional-system-for-hyperparameter-tuning/" />

  <meta name="citation_fulltext_html_url" content="https://Valavanca.github.io/compositional-system-for-hyperparameter-tuning/" />

  <meta name="citation_pdf_url" content="https://Valavanca.github.io/compositional-system-for-hyperparameter-tuning/manuscript.pdf" />

  <link rel="alternate" type="application/pdf" href="https://Valavanca.github.io/compositional-system-for-hyperparameter-tuning/manuscript.pdf" />

  <link rel="alternate" type="text/html" href="https://Valavanca.github.io/compositional-system-for-hyperparameter-tuning/v/7d72f2b6c68e10842430b93c461a91c976fc4282/" />

  <meta name="manubot_html_url_versioned" content="https://Valavanca.github.io/compositional-system-for-hyperparameter-tuning/v/7d72f2b6c68e10842430b93c461a91c976fc4282/" />

  <meta name="manubot_pdf_url_versioned" content="https://Valavanca.github.io/compositional-system-for-hyperparameter-tuning/v/7d72f2b6c68e10842430b93c461a91c976fc4282/manuscript.pdf" />

  <meta property="og:type" content="article" />

  <meta property="twitter:card" content="summary_large_image" />

  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />

  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />

  <meta name="theme-color" content="#ad1457" />

  <!-- end Manubot generated metadata -->'
keywords:
- multi-objective
- surrogate optimization
- software design
- sequential parameter optimization
lang: en-US
title: Compositional Multi-objective Parameter tuning
...






<small><em>
This manuscript
([permalink](https://Valavanca.github.io/compositional-system-for-hyperparameter-tuning/v/7d72f2b6c68e10842430b93c461a91c976fc4282/))
was automatically generated
from [Valavanca/compositional-system-for-hyperparameter-tuning@7d72f2b](https://github.com/Valavanca/compositional-system-for-hyperparameter-tuning/tree/7d72f2b6c68e10842430b93c461a91c976fc4282)
on February 25, 2020.
</em></small>

## Authors



+ **Oleksandr Husak**<br>
    ![ORCID icon](images/orcid.svg){.inline_icon}
    [XXXX-XXXX-XXXX-XXXX](https://orcid.org/XXXX-XXXX-XXXX-XXXX)
    · ![GitHub icon](images/github.svg){.inline_icon}
    [Valavanca](https://github.com/Valavanca)<br>
  <small>
     Research Student Assistant in TUD
  </small>



Introduction. Challenges and Problems.
======================================

Multi-objective optimisation is an established parameter tuning
technique. It is especially suited to solve complex, multidisciplinary
design problems with an accent on system design.

When we talk about several objectives, the intention is to find good
compromises rather than a single solution as in global optimization.
Since the solution for multi-objective optimization problems gives the
appearance to a set of Pareto-optimal points, evolutionary optimization
algorithms are ideal for handling multi-objective optimization problems.

General optimization methods could be classified into derivative and
non-derivative methods. In this thesis focuses on non-derivative
methods, as they are more suitable for parameter tuning. Therefore, they
are also known as black-box methods and do not require any derivatives
of the objective function to calculate the optimum. Other benefits of
these methods are that they are more likely to find a global optimum.

Motivation
----------

The purpose of this study is to introduce new surrogate-design-criteria
for multi-objective hyperparameter optimization software.

Motivation Examples in tuning algorithms:

-   Local search: neighbourhoods, perturbations, tabu length, annealing

-   Tree search: pre-processing, data structures, branching heuristics,
    clause learning deletion

-   Genetic algorithms: population size, mating scheme, crossover,
    mutation rate, local improvement stages, hybridizations

-   Machine Learning: pre-processing, learning rate schedules

-   Deep learning (in addition): layers, dropout constants,
    pre-training, activations functions, units/layer, weight
    initialization

Objectives
----------

Black-box multi-objective problems given a finite number of function
evaluations

Research Questions
------------------

1.  RQ(Cost): Does surrogate-based optimization cheaper in evaluations
    than other multi-goal optimization tools?

2.  RQ(Convergence speed): Does with surrogate-based optimization
    solutions converge faster to Pareto-front than with other multi-goal
    optimization tools?

3.  RQ(Quality): Does surrogate-based optimization return similar or
    better solutions than other optimization tools?

4.  RQ(Extensions and reusability): Reusable compositional system for
    optimization. Is it possible to extend light-weight single-objective
    experiments to heavy-weight multi/many-objective?

In numerous test problems, compositional-surrogate finds comparable
solutions to standard MOEA (NSGA-II, MOEAD, MACO, NSPSO) doing
considerably fewer evaluations (300 vs 5000). Surrogate-based
optimization is recommended when a model is expensive to evaluate.


Foundation
==========

In common old-fashioned software design, engineers carefully convert
overall models into domain-specific tools. In this approach, designers
codify the current understanding of the problem into the parameters.

Parameter tuning
----------------

Given recent advances in computing hardware, software analysts either
validate engineer models or find optimal configuration by using
parameter tuning tools to explore thousands to millions of inputs for
their systems.

In this article assume that parameter tuning is a subset problem of
general, global optimizations. It's also mean that we consider some
fitness function 'f' that converts the parameter vector to output
objectives. Note that the term \"real evaluation\" or \"black-box
evaluation\" as a synonym for the fitness function 'f'. The goal of
parameter tuning as an optimization task lay on fast iterative search
with improvements in each objective dimension. The term \"fast\" means
that the convergence to global optimum is achieved with the least real
evaluations and shorter time frame.

We consider fitness function 'f' as black-box with parameter and
objective space. Parameter space has structure and could consist from
continues and categorical dimensions. Sometimes, some combinations of
parameter settings are forbidden. Each pony from parameter space lead to
some point in objective space. Configurations often yield qualitatively
different behavior. Objective space also could be described as usual
objectives as accuracy, runtime, latency, performance, error rate,
energy, et.s. On each objective should gain the best possible value and
rich system tradeoff.

Optimization technics:

-   Grid search vs Random search

-   Heuristics and Metaheuristic. (Simulated annealing, Evolutionary
    algorithm..) These methods aim at generating approximately optimal
    solutions in a single run. Also could operate with sets of solutions
    being outcomes of multiple objectives.

-   Sequential design (Bayesian optimization, Evolutionary algorithm..)
    Bayesian methods differ from random or grid search in that they use
    past evaluation results to extrapolate and choose the next values to
    evaluate. Limit expensive evaluations of the objective function by
    choosing the next input values based on those that have done well in
    the past.

Optimization cost of black-box:

-   Evaluation may be very expensive

-   Sampling budget is unknown

-   Possibly noisy objectives

-   Feasibility constraints

-   Multi-objectivity

Ideally, we want a method that can explore the search space while also
limiting evaluations of hyperparameter choices. The single criterion in
parameter tuning may not be sufficient to correctly characterize the
behaviour of the configuration space that is why multiple criteria have
to be considered. One way to clarify the task of understanding the space
of possible solutions is to focus on the non-dominated frontier or
Pareto-front, the subset of solutions that are not worse than any other
but better on at least one goal. The difficulty here is that even the
Pareto frontier can be too large to understand.

Multi-objective optimization
----------------------------

Parameter tuning is present in our daily life and comes in a variety of
states. The goal is the rich best possible objective by correctly
choosing the system parameters. Common of optimization problems requires
the simultaneous optimization of multiple, usually contradictory,
objectives. These type of problems are termed as multiobjective
optimization problems. The solution to such problems is a family of
points, that placing on a Pareto front. Knowledge of the Pareto front
allows visualizing appropriate decisions in terms of performance for
each objective.

\"Multi-objective optimization(MOO) deals with such conflicting
objectives. It provides a mathematical framework to arrive at optimal
design state which accommodates the various criteria demanded by the
application. The process of optimizing systematically and simultaneously
a collection of objective functions are called multi-objective
optimization (MOO) [@NTeA6uPp]\".

For a multi-objective problem, we consider \"solution\" as points from
parameter space that lead to non-dominated results in objective space.
This set of points approximate real Pareto-front. Improving \"solution\"
means that sets of points coincide better with real Pareto-front. How to
search for an optimal solution to the multi-objective optimization
problem?

### Scalarizing. Weighted sum methods

Scalarizing approach is built on the traditional techniques to creating
an alternative problem with a single, composite objective function.
Single objective optimization techniques are then applied to this
composite function to obtain a single optimal solution. The weighted-sum
methods it's a well known type of scalarizing technic is applied to
simplify a multiobjective problem. Concatenate the objectives into one
criterion by using magic weighted sum factors. The merged objective is
used to evaluate and define the optimal solution. Weighted sum methods
have difficulties in selecting proper weight especially when there is no
connected a priori knowledge among objectives. Furthermore, Uniform
distribution points in parameters space don't generate uniform
distribution points on objective space. This means that we can't
approximate Pareto-front completely even with multiple optimization
rounds. Some scalarizing technics try to improve exploration of
parameter space by assigning more \"intelligence\" aggregation to the
objectives. Such solutions may be fragile. They change dramatically if
we modify algorithm parameters.

Moreover, the weighting method can not provide a solution among
underparts of the Pareto surface due to "duality gap" for not convex
cases. Even for convex cases, for example, in linear cases, even if we
want to get a point in the middle of a line segment between two points,
we hardly get a peak of Pareto surface, as long as the well-known
simplex method is used. This implies that depending on the structure of
the problem, the linearly weighted sum can not necessarily provide a
solution as DM desires. [@15iylVSF0]

### Multi-Objective Evolutionary Algorithms

Generating the Pareto set can be computationally expensive and is often
infeasible because the complexity of the underlying volume limits exact
techniques from being applicable. For this reason, a number of
stochastic search strategies such as evolutionary algorithms, tabu
search, simulated annealing, and ant colony optimization have been
developed: they usually do not guarantee to identify optimal trade-offs
but try to find a good approximation, i.e., a set of solutions whose
objective vectors are (hopefully) not too far away from the optimal
objective vectors [@1HUtCWxMH].

The evolutionary algorithm (EA) form a class of heuristic search methods
that simulate the process of natural evolution. Using simplifications,
this EA is subsequently determined by the two basic principles:
selection and variation. While selection imitates the competition for
reproduction and resources among living beings, the other principle,
variation, imitates the natural ability to create "new" living beings
through recombination and mutation. Evolutionary algorithm possesses
several characteristics that are desirable for problems including
multiple conflicting objectives, and large and complicated search
spaces. However, EA still need many evaluations of the \"black box\"
system to solve a common multi-objective problem. This is further
complicated by the fact that many such problems are very expensive.
Consolidated, this makes EAs unfeasible for costly and Multy-objective
problem. A good solution is the integration of the surrogate model which
extrapolate and approximate the fitness landscape from samples.
Multi-objective Evolutionary Algorithms (MOEAs) use this surrogate model
as a target for optimization. Assumed that solution from surrogate
nearby to a global optimum. The goal of this thesis is to understand if
the performance of MOEAs approach can be improved by using compositional
surrogates. The key idea of compositional surrogates is the splitting
objective space to multiple surrogates that extrapolate it
independently.Combination of multiple hypotheses should give them the
potential to approximate more complicated problems. This approach avoids
the idea of a single surrogate model, preferring instead to use the
composition hypothesis to split out the terrain of objective space.

The multiple surrogates are analysed on objectives with various
complexity, beside the simple and complicated unimodal structure.
Generating a cloud of candidates is computationally expensive.

Evolutionary optimizers explore populations of candidate solutions in
each generation, some mutator can make changes to the current
population. A select operator then picks the best mutants which are then
combined in some way to become generation i+1. This century, there has
been much new work on multi-objective evolutionary algorithms with two
or three objectives (as well as many-objective optimization, with many
more objectives). Multi-objective Evolutionary Algorithms (MOEAs) are
popular tools to solve optimization problems, because of their
applicability to complex fitness landscapes and solid performance on
problems with large design spaces. While other methods also exist, in
this thesis we will focus on improving approaches with Evolutionary
Algorithms for the Multy-objective optimizations. This search-based
software engineering is a rapidly expanding area of research and a full
survey of that work is beyond the scope of this thesis.

### Metrics for multi-objective solution

In single-objective minimization, the quality of a given solution is
trivial to quantify: the smaller the objective function value, the
better. However, evaluating the quality of an approximation of a Pareto
set is non trivial. The question is important for the comparison of
algorithms or prediction next configuration.

According to [@cgOPGY3M], a Pareto front approximation should satisfy
the following:

-   The distance between the Pareto front and its approximation should
    be minimized.

-   A heigh distribution of the non-dominated points is desirable.

-   The range of the approximated front should be maximized, i.e., for
    each objective, a wide range of values should be covered by the
    non-dominated points.

Metrics for performance indicators partitioned into four groups
according to their properties [@vbJ1WiNp]:

-   cardinality

-   convergence

-   distribution

-   spread

Base on the right metrics general multi-objective algorithm keep making
progress toward the Pareto front in the objective function space. The
goal of optimizing a multi-objective problem is to obtain an
approximation solution set to the reference Pareto front, including the
following subgoals:

-   All solution set are as close as possible to the Pareto front

-   All solution set are as diverse as possible in the objective space

-   Evaluate as few solution as possible

Straightforward applying of the simple coefficient of determination (R2)
is the wrong indicator of success. Evaluations of different sets of
Pareto optimal points is multi-objective task. The necessary objectives
follow for improving solutions:

-   Keep hypervolume low. Reference point is 0 for all objectives.

-   Maximize sparsity of points. Average distance. Crowding Distance.
    Spacing metrics.

-   Maximize non-dominant decisions in the total population

Also distribution and spread indicators is consider in this work.
According to [@DCdONjBz], "the spread metrics try to measure the
extents of the spread achieved in a computed Pareto front
approximation". They are not useful to evaluate the convergence of an
algorithm, or at comparing algorithms. They only make sense when the
Pareto set is composed of several solutions.

For multi-objective optimization (MOO), an algorithm should provide a
set of solutions that realize the optimal trade-offs between the
considered optimization objectives, i.e., Pareto set. Therefore, the
performance comparison of MOO algorithms is based on their Pareto sets.
In this study, three popular metrics are used to quantify the
performance of the algorithms.

-   Hypervolume (HV)[@17FLBEzc1]. This metric represents
    the volume of the objective space that is covered by the individuals
    of a non-dominated solutions set (solutions that belong to a Pareto
    front). The volume is delimited by two points: one point that is
    called the anti-optimal point (A) that is defined as the worst
    solution inside the objective space, and a second optimal point
    (pseudo-optimal) that is calculated by the proposed solution method.
    Determining the hypervolume indicator is a computationally expensive
    task. Even in case of a reasonably small dimension and low number of
    points (e.g. 100 points in 10 dimensions), there are currently no
    known algorithms that can yield the results fast enough for use in
    most multiple-objective optimizers

-   Non-dominated Ratio (NDR). This metric employs the non-dominated
    count of a solution set divided by the total size of solution set.
    Higher values are preferred to lower ones.

-   Spacing [@1AKoZUHA4]. Describe the distribution of Pareto
    points. Fewer space metrics means better coverage of objectives
    values range.

##### Conclusion

For optimization expensive black-box:

-   Scalable algorithms that convert multi-objective to single objective
    problem produce solution that not accurate enough(Scalarizing). Also
    this approach suitable for a limited type of problem.

-   Genetic algorithms. This approach is costly to perform and not
    appropriate for expensive problems.

Optimization gap in obtaining high quality, multi/single-obj solutions
in expensive to evaluate experiments. Experiments as a black box,
derivative-free. Reference to surrogate optimization.

Surrogate optimization
----------------------

[@yGyc3usi]

To dealing with expensive optimization problem more quickly, we can use
surrogate models in the optimization process to approximate the
objective functions of the problem. Approximation of solution is faster
than the whole optimization process can be accelerated. Nevertheless,
the extra time needed to build and update the surrogate models during
the optimization process. In the case of pre-selecting the promising
individuals, the surrogate model is used to find the likely or drop the
low-quality individuals even before they are exactly evaluated, thus
reducing the number of exact evaluations.

In the literature, the term surrogate or model-based optimization is
used where, during the optimization processes, some solutions are not
evaluated with the original objective function, but are approximated
using a model of this function. Different approximation methods are used
to build surrogate models. For single and multiobjective optimization
similar methods are used. These techniques typically return only one
approximated value, which is why in multiobjective problems several
models have to be used, so that each model approximates one objective.
Some of the most commonly used methods are the Response Surface Method
[@WHqOntch], Radial Basis Function [@1FOgcJYL9], Neural
Network, Kriging [@1Y1nHvc2] and Gaussian Process Modeling
[@1FXlw7TX2; @4nfjnHfv].

General classification [@mu8IFeYo]: Within surrogate-model-based
optimization algorithms, a mechanism is needed to find a balance between
the exact and approximate evaluations. In evolutionary algorithms, this
mechanism is called evolution control [@112CjFl6F] and can be either fixed
or adaptive. In fixed evolution control the number of exact function
evaluations that will be performed during the optimization is known in
advance. Fixed evolution control can be further divided into
generation-based control, where in some generations all solutions are
approximated and in the others, they are exactly evaluated [@4bMRxtHA],
and individual based control, where in every generation some (usually
the best) solutions are exactly evaluated and others approximated
[@1AF8tzsvd]. In adaptive evolution control, the number of exactly
evaluated solutions is not known in advance but depends on the accuracy
of the model for the given problem. Adaptive evolution control can be
used in one of two ways: as a part of a memetic search or to pre-select
the promising individuals which are then exactly evaluated [@MlTHme7o].

Surrogate used to expedite search for global optimum. Global accuracy of
surrogate not a priority. Surrogate model is cheaper to evaluate than
the objective.

Bayesian optimization (BO) methods often rely on the assumption that the
objective function is well-behaved, but in practice, the objective
functions are seldom well-behaved even if noise-free observations can be
collected. In [@wTIHyFe9] propose robust surrogate models to
address the issue by focusing on the well- behaved structure informative
for search while ignoring detrimental structure that is challenging to
model data efficiently.

##### Surrogate-model-based MOEA

In [@15iqjq0XF] proposed approaches that apply kind of surrogate
assistant to evaluations and ranging new population. It allows detecting
the most informative examples in population and evaluates them.
Identifies and evaluates just those most informative examples at the end
done fewer evaluations of the real system. Another way to explore
solutions is to apply some heuristic to decompose the total space into
many smaller problems, and then use a simpler optimizer for each region.

Surrogates are also used to rank and filter out offspring according to
Pareto-related indicators like the hypervolume [@iL724s1s], or a
weighted sum of the objectives [@890MGfiJ]. The problem with the
methods that use hypervolume as a way of finding promising solutions is
the calculation time needed to calculate the hypervolume, especially on
many objectives. Another possibility is described in [@16Jfe9RjR], where
the authors present an algorithm that calculates only non-dominated
solutions or solutions that can, because of variance, become
non-dominated.

GP-DEMO [@mu8IFeYo] The algorithm is based on the newly defined
relations for comparing solutions under uncertainty. These relations
minimize the possibility of wrongly performed comparisons of solutions
due to inaccurate surrogate model approximations. Using this confidence
interval, we define new dominance relations that take into account this
uncertainty and propose a new concept for comparing solutions under
uncertainty that requires exact evaluations only in cases where more
certainty is needed.

##### Surrogate with MOEA

Kind of extending the search stage of MOEA with surrogate to simulate
evaluation of population. It transform the problem of searching a new
better population to improving general hypothesis of how and where
Pareto set presented.

In surrogate-model-based multiobjective optimization, approximated
values are often mistakenly used in the solution comparison. As a
consequence, exactly evaluated good solutions can be discarded from the
population because they appear to be dominated by the inaccurate and
over-optimistic approximations. This can slow the optimization process
or even prevent the algorithm from finding the best solutions
[@mu8IFeYo].

##### Discussion

Example of each type of optimization. Justification solution.
Conclusion: Design gap in optimization/parameter tuning. Need to
indicate optimization workflow for expensive process/experiments. The
argument(s) why we need a new architecture. Reference to composition
architecture.

Surrogate based optimization has proven effective in many aspects of
engineering and in applications where data is \"expensive\", or
difficult, to evaluate.

Compositional architecture
--------------------------

We could describe compositional-based surrogate optimization as compound
grey-box system whit a lot of open research areas where surrogate should
improve, managing portfolio, compare of predictions Pareto fronts. As a
developer, you can be focused on a specific problem and don't know how
to implement other components. This is one of the main advantages of the
described approach.

##### Compositional surrogates

Can the same single-objective models be equally applied to various types
of problems in multi-/single-objective optimization? When there is no
correlation between the objectives, a very simple way to solve this kind
of problem is to build independent models, i.e. one for each objective,
and then to use those models to simultaneously extrapolate possible
solutions with MOEA. Nevertheless, the output values correlated, but an
often naive way to build multiple models that able to extrapolate
complex objective space is often given good results.

Later research generalized this approach. MOEA/D (multiobjective
evolutionary algorithm based on decomposition [@Dk9KXQBq]) is a generic
framework that decomposes a multi-objective optimization problem into
many smaller single problems, then applies a second optimizer to each
smaller subproblem, simultaneously.

With multiple models, their flaws can combine, as well as the time
required to build the models. In memetic algorithms, especially if the
surrogate model is not very accurate, a local optimum can be found
instead of the global optimum. But in terms of parameter tuning, this
point should be better than a predefined sampling plan. Evaluation of
this prediction improve surrogate model quality in the near-optimal area
and improve prediction in the next round. For example, OEGADO
[@1C0JI6fEI] creates a surrogate model for each of the objectives.
The best solutions in every objective get also approximated on other
objectives, which helps with finding trade-off individuals. The best
individuals are then exactly evaluated and used to update the models.

Scope of work
-------------

Describe and implement workflow for multi-objective parameter tuning of
the derivative-free, black-box system. Parameter estimation is costly.
The proposed solutions are also suitable for single-criteria
optimization. Problem Setting.

Goal:

1.  Globally optimize an objective function(s) that is expensive to
    evaluate. Single/Multi-objective parameter tuning

2.  Simultaneously optimization scalable objectives

3.  Components reuse. Extensibility with other frameworks

Problem:

1.  A large number of the target black-box evaluations

2.  Interfaces not unify

3.  Code duplication

Solution:

1.  Component-Based Architecture

2.  Compositional-based surrogate optimization with MOEA


Concept
=======

For slow computational problems, it would be useful to modulate a
problem using a quite small number of most informative examples. This
general topic introduces compositional surrogate, as a proxy model that
approximate objectives surfaces and support MOEA to evaluates near a
multi-objective solution and predict better multi-objective samples on
each iteration. Model-based or optimization with a surrogate is the
preferred choice for functional optimization when the evaluation cost is
very large.

As discussed in Section 3, current MOEAs use increasingly complex opera-
tors. Re-implementing these algorithms for each usage scenario becomes
timeconsuming and error-prone. Mainly two groups are affected by this
problem: -- Application engineers who need to choose, implement, and
apply state-of- the-art algorithms without in-depth programming
knowledge and expertise in the optimization domain. -- Developers of
optimization methods who want to evaluate algorithms on different test
problems and compare a variety of competing methods.

A different approach, called PISA (A Platform and programming language
independent Interface for Search Algorithms), was presented in \[3\].
The underlying concept is discussed in the following section.

The basic idea is to divide the implementation of an optimization method
into an algorithm-specific part and an application-specific part as
shown in Fig. 15. The former contains the selection procedure, while the
latter encapsulates the representation of solutions, the generation of
new solutions, and the calculation of objective function values.

Nevertheless it is easy to add the interface functionality to an
existing algorithm or application since the whole communication only
consists of a few text file operations. As a negative consequence, the
data transfer introduces an additional overhead into the optimization.

There is a clear need for a method to provide and distribute
ready-to-use implementations of optimization methods and ready-to-use
benchmark and real- world problems. These modules should be freely
combinable. Since the above- mentioned issues are not constrained to
evolutionary optimization a candidate solution should be applicable to a
broader range of search algorithms.

The main objective of this part is to provide a thorough treatment of
multy-objective parameter tuning with evolutionary algorithm(s)

Key description how to improve solutions for problems in research
questions.

Multi-objective optimizations are frequently encountered in engineering
practices. The solution techniques and parametric selections however are
usually problem-specific. [@BvoEknRH]

Reduce effort for multi-obj prediction/solution
-----------------------------------------------

##### Surrogate model. Hypothesis as a middleware

Key idea is to use hypothesis model as middleware for genetic
multi-objective algorithms. This hypothesis could be compositional and
delineate target objectives.

Reusability in parameter tuning
-------------------------------

Parameter tuning can be splitted down into steps that are common for the
many/single-objective optimizations. Each step in optimization workflow
has variability via implemented interfaces. Single-objective hypotheses
can be combined for multi-objective optimization with compositional
design.

API of metric-learn is compatible with scikit-learn, the leading library
for machine learning in Python. This allows to use all the scikit-learn
routines (for pipelining, model selection, etc) with metric learning
algorithms through a unified interface.

\[!TODO\] Real, integer, ordinal and categorical variables.

Surrogate portfolio
-------------------

A Surrogate(s) is a simplified version of the examples. The
simplifications are meant to discard the superfluous details that are
unlikely to generalize to new instances. However, to decide what data to
discard and what data to keep, you must make hypothesis. For example, a
linear model makes the hypothesis that the data is fundamentally linear
and that the distance between the instances and the straight line is
just noise, which can safely be ignored.

If there is no hypothesis about the data, then there is no reason to
prefer one surrogate over any other. This is called the No Free Lunch
(NFL) theorem. For some datasets the best model is a linear model, while
for other datasets it is a neural network. There is no model that is a
priori guaranteed to work better (hence the name of the theorem). The
only way to know for sure which model is best is to evaluate them all.
Since this is not possible, in practice you make some reasonable
assumptions about the data and you evaluate only a few reasonable
models. For example, for simple tasks you may evaluate linear models
with various levels of regularization, and for a complex problem you may
evaluate various neural networks.

\"No Free Lunch\" (NFL) theorems demonstrate that if an algorithm
performs well on a certain class of problems then it necessarily pays
for that with degraded performance on the set of all remaining problems
Additionally, the name emphasizes the parallel with similar results in
supervised learning.

1.  You have to try multiple types of surrogate(models) to find the best
    one for your data.

2.  A number of NFL theorems were derived that demonstrate the danger of
    comparing algorithms by their performance on a small sample of
    problems.

GALE uses MOEA decomposition but avoids certain open issues with
E-domination and MOEA/D. GALE does the subproblems is determined via a
recursive median split not need some outside oracle to specify E.
Rather, the size of on dimensions synthesized using a PCA-approximation

##### Reusable software

Problem that each optimization framework/library use inner interfaces.
It is necessary to define a standard that implements best practices for
extension libraries [@2iBa9w3z]. We introduce new Model-based
line for parameter tuning.

Conclusions
-----------

Also, to the best of our knowledge, has not been previously reported in
the SBSE literature. GALE's cost reduction of MOEA to O2 log2N
evaluations

1.  Surrogate portfolio. Search a better hypothesis for a specific
    problem at a particular stage of parameter tuning

2.  Large set of evaluation problems for comprehensive and fair
    comparison

3.  Interfaces for reusability and scalability.

Optimization problems involving multiple objectives are common. In this
context, evolutionary computation represents a valuable tool, in
particular -- if we would like to be flexible with respect to the
problem formulation, -- if we are interested in approximating the Pareto
set, and -- if the problem complexity prevents exacts methods from being
applicable.

Flexibility is important if the underlying model is not fixed and may
change or needs further refinement. The advantage of evolutionary
algorithms is that they have minimum requirements regarding the problem
formulation; objectives can be easily added, removed, or modified.
Moreover, due the fact that they operate on a set of solution
candidates, evolutionary algorithms are well-suited to generate Pareto
set approximations. This is reflected by the rapidly increasing interest
in the field of evolutionary multiobjective optimization. Finally, it
has been demonstrated in various applications that evolutionary
algorithms are able to tackle highly complex problems and therefore they
can be seen as an approach complementary to traditional methods such as
integer linear programming.


Implementation. Development
===========================

Without automated tools, it can take days for experts to review just a
few dozen examples. In that same time, an automatic tool can explore
thousands to millions to billions more solutions. People find it an
overwhelming task just to certify the correctness of conclusions
generated from so many results.

Separation of concerns

Managing complex execution Strategies

Variants in the evaluation of sets of solutions for each hypothesis.
Each hypothesis has quality metrics. Solution(s) from each hypothesis
have also own metrics.

There are main approaches how produce single solution:

-   Solution from best hypothesis. Sorting

-   Bagging solution

-   Voting solution

##### Designing a Sampling Plan

\- The most straightforward way of sampling a design space in a uniform
fashion is by [@yGyc3usi] means of a rectangular grid of points. This
is the full factorial sampling technique referred - Latin Squares

Dependencies
------------

Adapted to provide base implementation for stages in parameter tuning
with multi-objective

##### Pagmo2

A Python platform [@14vOEgTO2] to perform parallel
computations of optimisation tasks (global and local) via the
asynchronous generalized island model. All test suites and basic
multi-objective solvers:

Realization of main MOEA:

-   NSGA2. Non-dominated Sorting Genetic Algorithm

-   MOEA/D. Multi Objective Evolutionary Algorithms by Decomposition
    (the DE variant)

-   MACO. Multi-objective Ant Colony Optimizer.

-   NSPSO.

Tests suits:

-   ZDT [@cgOPGY3M] is 6 different two-objective scalable problems
    all beginning from a combination of functions allowing, to measure
    the distance of any point to the Pareto front while creating
    problems.

-   WFG [@1HFetF7vb] was conceived to exceed the functionalities of
    previously implemented test suites. In particular, non-separable
    problems, deceptive problems, truly degenerative problems and mixed
    shape Pareto front problems are thorougly covered, as well as
    scalable problems in both the number of objectives and variables.
    Also, problems with dependencies between position and distance
    related parameters are covered. In their paper the authors identify
    the need for nonseparable multimodal problems to test
    multi-objective optimization algorithms. Given this, they propose a
    set of 9 different scalable multi-objective unconstrained problems.

-   DTLZ [@yHWkop2U]. All problems in this test suite are
    box-constrained continuous n-dimensional multi-objective problems,
    scalable in fitness dimension.

Portfolio with hypothesis
-------------------------

A set of models is defined that can form a partial or complete
hypothesis to describe the problem. Also during the increase of the
experiments may change the model that best describes the existing
problem As a result, there is variability for each problem and
configuration step at the same time. A set of hypotheses can solve this
problem but it takes longer time for cross validation.

Validate hypothesis
-------------------

::: {.epigraph}
"All models are wrong but some are useful"

*-- George Box*
:::

The main task of learning algorithms is to be able to generalize to
unseen data. Surrogate model as learning model should generalize
examples to valid hypothesis. Since we cannot immediately check the
surrogate performance on new, incoming data, it is necessary to
sacrifice a small portion of the examples to check the quality of the
model on it. n case if surrogate model have enoughs score (pass metrics
threshold) we consider it valid and could be processed as subject for
inference(prediction).

### Sampling strategy

Oversampling and undersampling in data analysis. Alleviate imbalance in
the dataset. Imbalance in dataset is not always a problem, more so for
optimization tasks.

The main gain for models not to provide best accuracy on all search
space but provide possible optimum regions. Accuracy in prediction
optimal regions or points from there will direct the search in the right
direction.

Predictor variables can legitimately over- or under-sample. In this
case, provided a carefully check that the model assumptions seem valid.

for other set of parameters, and make a choice from more diverse pool of
models.


Evaluation. Experimental Results
================================

This chapter presents the evaluation of the proposed method on test
problems with diverse objective landscape and with a various number of
search variables.

Roughly speaking, an MOEA is called globally convergent if the sequence
of Pareto front approximations A(t) it produces converges to the true
Pareto front Y while the number of generations t goes to infinity. It is
intuitively clear that this property can only be fulfilled with
unlimited memory resources, as the cardinality of the Pareto front can
be arbitrary large in general \[Convergence properties of some
multi-objective evolutionary algorithms.\].

[@Vb3fs8xv]

Will be used two types of problems: Synthetic and Real physical

Idea: Generate problem from data set and try to optimize it with
parameter tunning from the beginning. Need models with accuracy  96 amd
multiply objectives.

Test suite: ZDT
---------------

This widespread test suite was conceived for two-objective problems and
takes its name from its authors Zitzler, Deb and Thiele.
Ref\["Comparison of multiobjective evolutionary algorithms: Empirical
results.", 2000\]

Test suite: DTLZ
----------------

This widespread test suite was conceived for multiobjective problems
with scalable fitness dimensions and takes its name from its authors
Deb, Thiele, Laumanns and Zitzler. Ref\[\"Scalable Test Problems for
Evolutionary Multiobjective Optimization\", 2005\]

Test suite: WFG
---------------

This test suite was conceived to exceed the functionalities of
previously implemented test suites. In particular, non-separable
problems, deceptive problems, truly degenerative problems and mixed
shape Pareto front problems are thoroughly covered, as well as scalable
problems in both the number of objectives and variables. Also, problems
with dependencies between position and distance related parameters are
covered.

1.  A few unimodal test problems should be present in the test suite.
    Various Pareto optimal geometries and bias conditions should define
    these problems, in order to test how the convergence velocity is
    influenced by these aspects.

2.  The following three Pareto optimal geometries should be present in
    the test suite: degenerate Pareto optimal fronts, disconnected
    Pareto optimal fronts and disconnected Pareto optimal sets.

3.  Many problems should be multimodal, and a few deceptive problems
    should also be covered.

4.  The majority of test problems should be non-separable.

5.  Both non-separable and multimodal problems should also be addressed.

Ref\[ "A Review of Multi-Objective Test Problems and a Scalable Test
Problem Toolkit", 2006\]

Problem suite: CEC 2009
-----------------------

Competition on "Performance Assessment of Constrained / Bound
Constrained Multi-Objective Optimization Algorithms". All problems are
continuous, multi objective problems.

Physical. Real world problem
----------------------------

Computational models describing the behavior of complex physical systems
are often used in the engineering design field to identify better or
optimal solutions with respect to previously defined performance
criteria. Multi-objective optimization problems arise and the set of
optimal compromise solutions (Pareto front) has to be identified by an
effective and complete search procedure in order to let the decision
maker, the designer, to carry out the best choice.

### Materials Selection in mechanical design

### Test generation

### Gold or oil search. Geodesy

### Space crafts

Problem 1: obtain a set of geometric design parameters to get minimum
heat pipe mass and the maximum thermal conductance. Thus, a set of
geometric design parameters lead to minimum pressure total cost and
maximum pressure vessel volume. The alternative solutions are very
difficult to be adopted to practical engineering decision directly.
Ref\[Multi-Objective Optimization Problems in Engineering Design Using
Genetic Algorithm Case Solution\] Problem 2: Solar sailing mission
design

Conclusion
----------

The quality of the results obtained with X was similar to the results
obtained with Y, but with significantly fewer exactly evaluated
solutions during the optimization process.


Related work
============

Many existing approaches can be categorized as multi-objective
optimization. That is why introduce comparison criteria for a clear and
concise demarcation of the approach presented in this thesis:

Comparison Criteria for Related Work.

-   Variability. Exchange surrogate, solver and sampling algorithms as
    components. Variants on each optimization workflow step.

-   Scalability. Extend single-objective problem on the fly to
    multi-objective.

-   Adaptation. Surrogate portfolio.

-   From 0 to hero. Sampling plan depends on surrogate validity. The
    Sobol sequence (and Latin hypercube).

Important Features: Categorical variables, prior knowledge,
multi-objective, feasibility constraints.

Dependencies
------------

##### AUTO-SKLEARN

[@wqiAUHlQ] - CASH (Combined Algorithm Selection
and Hyperparameter optimization) problem

##### TPOT

[@1GyLc6Kgv] Already implemented TPOT automodel as hypothesis
candidate

##### jMetalpy

[@om6wdVM3] Partially implemented some solvers.

##### Hyperopt

##### raw:PlatEMO

[@d1YE9XcQ] raw:PlatEMO: A MATLAB Platform for Evolutionary Multi-Objective
Optimization


Conclusion and Future Work
==========================

General Conclusion
------------------

### BRISE

Modelbase line in parameter tuning for software Product Line for
Parameter Tuning

Future Work
-----------

### Prior knowledge. Transfer learning

What is already implemented and how it could be improved.

-   Model portfolio selection and combination.

-   Prior distribution of parameters. Bayesian kernels.

-   Human in the loop. Reducing the search space.


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>
