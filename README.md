[![Build Status](https://travis-ci.com/Valavanca/compositional-system-for-hyperparameter-tuning.svg?branch=master)](https://travis-ci.com/Valavanca/compositional-system-for-hyperparameter-tuning)
![Build LaTeX document](https://github.com/Valavanca/compositional-system-for-hyperparameter-tuning/workflows/Build%20LaTeX%20document/badge.svg)

# Compositional Multi-objective parameter tuning

> Developed system and technology for reusing surrogate modeling for parameter tuning in software Product-line/Search-based engineer.
___

## Project structure:

### - Manubot directories:

+ [`content`](content) contains the manuscript source, which includes markdown files as well as inputs for citations and references.
  See [`manuscript_usage.md`](manuscript_usage.md) for more information.
+ [`output`](output) contains the outputs (generated files) from Manubot including the resulting manuscripts.
  You should not edit these files manually, because they will get overwritten.
+ [`webpage`](webpage) is a directory meant to be rendered as a static webpage for viewing the HTML manuscript.
+ [`build`](build) contains commands and tools for building the manuscript.
+ [`ci`](ci) contains files necessary for deployment via continuous integration.
  For the CI configuration, see [`.travis.yml`](.travis.yml).

### - Latex directories

+ [`stthesis_latex`](stthesis_latex) contains latex templates and root file.
+ [`stthesis_latex/content`](stthesis_latex/content) contains thesis chapters and images

___
