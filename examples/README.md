## John Maynard Keynes, a great economist and thinker, said <i>"When the facts change, I change my mind. What do you do, sir?"</i>

𝑃(𝐴|𝑋) - probability of 𝐴 given the evidence  𝑋

Difference between the observed frequency and the true frequency of an event. Maybe it's will be good idee to reduce repetitions.


### Dagster

- Functional. Batch data processing

        Data pipelines should be expressed as 'DAGs' (directed acyclic graphs) of functional, idempotent computations. Individual nodes in the graph consume their inputs, perform some computation, and yield outputs, either with no side effects or with clearly advertised side effects. Given the same inputs and configuration, the computation should always produce the same output. If these computations have external dependencies, these should be parametrizable, so that the computations may execute in different environments.
        - [Functional Data Engineering](https://medium.com/@maximebeauchemin/functional-data-engineering-a-modern-paradigm-for-batch-data-processing-2327ec32c42a)

            To put it simply, immutable data along with versioned logic are key to reproducibility.

- Heterogeneity in data pipelines is the norm, rather than the exception.

- Testable

- Pipeline tests for data quality

___

The core abstraction is the Solid. A Solid is a functional unit of computation that consumes and produces data assets. It has a number of properties:

 - Coarse-grained and for use in __batch__ computations.

 - Defines inputs and outputs, optionally typed within the inner type system.

 - Embeddable in a dependency graph that is constructed by connecting inputs and outputs, rather than just the Solids themselves.

 - ~~Emits a stream of typed, structured events – such as expectations and materializations – that define the semantics of their computation.~~ __?__

 - Defines self-describing, strongly typed configuration.

 - Designed for testability and reuse.

___

### Test

___We can think:__ how can we test whether our model is a bad fit?_ An idea is to compare observed data (which if we recall is a fixed stochastic variable) with artificial dataset which we can simulate. The rationale is that if the simulated dataset does not appear similar, statistically, to the observed dataset, then likely our model is not accurately represented the observed data.


### Problems

- [SCHWEFEL FUNCTION](https://www.sfu.ca/~ssurjano/schwef.html)


### Loss function

In Bayesian inference, we have a mindset that the unknown parameters are really random variables with prior and posterior distributions. Concerning the posterior distribution, a value drawn from it is a possible realization of what the true parameter could be. Given that realization, we can compute a loss associated with an estimate. 

First it will be useful to explain a Bayesian point estimate. The systems and machinery present in the modern world are not built to accept posterior distributions as input.

Similarly, we need to distill our posterior distribution down to a single value (or vector in the multivariate case).
The minimum of the expected loss is called the *Bayes action.* 


### Comparison

- scikit-learn
    For regression tasks, where we are predicting a continuous response variable, a GaussianProcessRegressor is applied by specifying an appropriate covariance function, or kernel.

- GPflow 
    The main innovation of GPflow is that non-conjugate models (i.e. those with a non-normal likelihood) can be fit either using Markov chain Monte Carlo or an approximation via variational inference.
    Priors can be assigned as variable attributes, using any one of GPflow’s set of distribution classes, as appropriate.

- PyMC3
    The PyMC project is a very general Python package for probabilistic programming that can be used to fit nearly any Bayesian model.

>... after the model has been fit, one should look at the posterior distribution and see if it makes sense. If the posterior distribution does not make sense, this implies that additional prior knowledge is available that has not been included in the model, and that contradicts the assumptions of the prior distribution that has been used. It is then appropriate to go back and alter the prior distribution to be more consistent with this external knowledge.

   -Gelman

___
#### Empirical Bayes

Bayesian methods have a prior distribution, with hyperparameters  𝛼 , while empirical methods do not have any notion of a prior. Empirical Bayes combines the two methods by using frequentist methods to select  𝛼 , and then proceeds with Bayesian methods on the original problem.

**Problem:** [Anchoring](https://en.wikipedia.org/wiki/Anchoring) Ideally, all priors should be specified before we observe the data, so that the data does not influence our prior opinions


## Useful priors to know about

- *The Gamma distribution.* Generalization of the Exponential random variable
- *The Wishart distribution.* Random matrices. Appropriate prior for covariance matrices
- *The Beta distribution.* a Beta prior with Binomial observations creates a Beta posterior



## Frameworks

- [Scikit-learn hyperparameter search wrapper](https://scikit-optimize.github.io/notebooks/sklearn-gridsearchcv-replacement.html)
