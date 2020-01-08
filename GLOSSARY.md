# Glossary

> __Algorithmic differentiation (AD)__ is a means of generating derivatives of
mathematical functions that are expressed in computer code (Griewank
2003, Griewank and Walther 2008). The forward mode of AD may be viewed
as performing differentiation of elementary mathematical operations in each
line of source code by means of the chain rule, while the reverse mode may
be seen as traversing the resulting computational graph in reverse order.


> __Numerical differentiation__ Another alternative to derivative-free methods is to estimate the derivative of f by numerical differentiation and then to use the estimates in a
derivative-based method. This approach has the benefit that only zerothorder information (i.e. the function value) is needed; however, depending
on the derivative-based method used, the quality of the derivative estimate
may be a limiting factor