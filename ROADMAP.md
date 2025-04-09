# Road map

February 2020; updated, December 2024

The road map below is more big-picture; to get into the weeds, see also [this GH
Project](https://github.com/orgs/JuliaAI/projects/1).

Please visit [contributing guidelines](CONTRIBUTING.md) if interested
in contributing to MLJ.

### Goals

-   **Usability, interoperability, extensibility, reproducibility,**
	and **code transparency**.

-   Offer state-of-art tools for model **composition** and model
	**optimization** (hyper-parameter tuning)

-   Avoid common **pain-points** of other frameworks with MLJ:

	-   identify and list all models that solve a given task

	-   easily perform routine operations requiring a lot of code

	-   easily transform data, from source to algorithm-specific data format

	-   make use of probabilistic predictions: no more inconsistent representations / lack
		of options for performance evaluation

-   Add some focus to julia machine learning software development more
	generally

### Priorities

Priorities are somewhat fluid, depending on funding offers and available talent. However,
we are always keen to review external contributions in any area.

## Future enhancements

### Adding models

- [x] **Integrate deep learning** using [Flux.jl](https://github.com/FluxML/Flux.jl.git) deep learning.  [Done](https://github.com/FluxML/MLJFlux.jl) but can
  improve the experience by:

  - [x] finishing iterative model wrapper [#139](https://github.com/JuliaAI/MLJ.jl/issues/139)

  - [ ] improving performance by implementing data front-end after (see [MLJBase
  #501](https://github.com/JuliaAI/MLJBase.jl/pull/501)) but see also [this relevant discussion](https://github.com/FluxML/MLJFlux.jl/issues/97).


-  [ ] Probabilistic programming:
   [Turing.jl](https://github.com/TuringLang/Turing.jl),
   [Gen](https://github.com/probcomp/Gen),
   [Soss.jl](https://github.com/cscherrer/Soss.jl.git)
   [#157](https://github.com/JuliaAI/MLJ.jl/issues/157)
   [discourse
   thread](https://discourse.julialang.org/t/ppl-connection-to-mlj-jl/28736)
   [done](https://github.com/tlienart/SossMLJ.jl) but experimental and
   requires:

   - [ ] extension of probabilistic scoring functions to
	 "distributions" that can only be sampled.

-   [ ] Feature engineering (python featuretools?, recursive feature
	elimination ✓ done in FeatureSelection.jl :)
	[#426](https://github.com/JuliaAI/MLJ.jl/issues/426) [MLJModels #314](https://github.com/JuliaAI/MLJModels.jl/issues/314)


### Enhancing core functionality

-   [ ] **†** Add more tuning
	strategies. See [here](https://github.com/JuliaAI/MLJTuning.jl#what-is-provided-here)
	for complete
	wish-list. Particular focus on:

	- [ ] Bayesian methods, starting with Gaussian Process methods a
	  la PyMC3. Some preliminary research done.

	- [ ] POC for AD-powered gradient descent [#74](https://github.com/JuliaAI/MLJ.jl/issues/74)

	- [ ] Tuning with adaptive resource allocation, as in
	  Hyperband. This might be implemented elegantly with the help of
	  the recent `IterativeModel` wrapper, which applies, in
	  particular to `TunedModel` instances [see
	  here](https://JuliaAI.github.io/MLJ.jl/dev/controlling_iterative_models/#Using-training-losses,-and-controlling-model-tuning).

	- [ ] Genetic algorithms
[#38](https://github.com/JuliaAI/MLJ.jl/issues/38)

	- [ ] tuning strategies for non-Cartesian spaces of models [MLJTuning
	#18](https://github.com/JuliaAI/MLJTuning.jl/issues/18), architecture search, and other AutoML workflows

- [ ]  Systematic benchmarking, probably modeled on
	[MLaut](https://arxiv.org/abs/1901.03678) [#69](https://github.com/JuliaAI/MLJ.jl/issues/74)

- [ ]   Give `EnsembleModel` a more extendible API and extend beyond bagging
	(boosting, etc) and migrate to a separate repository?
	[#363](https://github.com/JuliaAI/MLJ.jl/issues/363)


### Broadening scope

- [ ] Integrate causal and counterfactual methods for example,
  applications to FAIRness; see [this
  proposal](https://julialang.org/jsoc/gsoc/MLJ/#causal_and_counterfactual_methods_for_fairness_in_machine_learning)

- [ ] Explore the possibility of closer integration of Interpretable
  Machine Learning approaches, such as Shapley values and lime; see
  [Shapley.jl](https://gitlab.com/ExpandingMan/Shapley.jl),
  [ShapML.jl](https://github.com/nredell/ShapML.jl),
  [ShapleyValues.jl](https://github.com/slundberg/ShapleyValues.jl),
  [Shapley.jl (older)](https://github.com/frycast/Shapley.jl) and
  [this
  proposal](https://julialang.org/jsoc/gsoc/MLJ/#interpretable_machine_learning_in_julia)

- [ ] Add sparse data support and better support for NLP models; we
	could use [NaiveBayes.jl](https://github.com/dfdx/NaiveBayes.jl)
	as a POC (currently wrapped only for dense input) but the API
	needs to be finalized first
	{#731](https://github.com/JuliaAI/MLJ.jl/issues/731). Probably
	need a new SparseTables.jl package.

- [x] POC for implementation of time series models classification
	[#303](https://github.com/JuliaAI/MLJ.jl/issues/303),
	[ScientificTypesBase #14](https://github.com/JuliaAI/ScientificTypesBase.jl/issues/14) POC is [here](https://github.com/JuliaAI/TimeSeriesClassification.jl)

- [ ] POC for time series forecasting, along lines of sktime; probably needs [MLJBase
	#502](https://github.com/JuliaAI/MLJBase.jl/issues/502)
	first, and someone to finish [PR on time series
	CV](https://github.com/JuliaAI/MLJBase.jl/pull/331). See also [this proposal](https://julialang.org/jsoc/gsoc/MLJ/#time_series_forecasting_at_scale_-_speed_up_via_julia)

- [ ]   Add tools or a separate repository for visualization in MLJ.

	- [x] Extend visualization of tuning plots beyond two-parameters
	[#85](https://github.com/JuliaAI/MLJ.jl/issues/85)
	(closed).
	[#416](https://github.com/JuliaAI/MLJ.jl/issues/416)
	[Done](https://github.com/JuliaAI/MLJTuning.jl/pull/121) but might be worth adding alternatives suggested in issue.

	- [ ] visualizing decision boundaries? [#342](https://github.com/JuliaAI/MLJ.jl/issues/342)

	- [ ] provide visualizations that MLR3 provides via [mlr3viz](https://github.com/mlr-org/mlr3viz)

- [ ] Add more pre-processing tools:

  - [x] missing value imputation using Gaussian Mixture Model. Done,
	via addition of BetaML model, `MissingImputator`.

  - [ ] improve `autotype` method (from ScientificTypes), perhaps by
	training on a large collection of datasets with manually labelled
	scitype schema.
	
- [ ] Extend integration with [OpenML](https://www.openml.org) WIP @darenasc


### Scalability

- [ ]   Roll out data front-ends for all models after  [MLJBase
  #501](https://github.com/JuliaAI/MLJBase.jl/pull/501)
  is merged.

- [ ]  Online learning support and distributed data
	[#60](https://github.com/JuliaAI/MLJ.jl/issues/60)

- [ ]  DAG scheduling for learning network training
	[#72](https://github.com/JuliaAI/MLJ.jl/issues/72)
	(multithreading first?)

- [ ]  Automated estimates of cpu/memory requirements
	[#71](https://github.com/JuliaAI/MLJ.jl/issues/71)

