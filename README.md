## MLJ Machine Learning Models Tuning Library - 100% Made in Julia

Edoardo Barp, Gergö Bohner, Valentin Churvay, Harvey Devereux, Thibaut Lienart, Franz J Király, Mohammed Nook, Annika Stechemesser, Sebastian Vollmer; Mike Innes in partnership with Julia Computing

First attempt is present in the [AnalyticalEngine.jl](https://github.com/tlienart/AnalyticalEngine.jl) and [Orchestra.jl](https://github.com/svs14/Orchestra.jl)


Current project was started as research study group at the Univeristy of Warwick. It started with a review of existing ML Modules that are available in Julia [in-depth](https://github.com/dominusmi/Julia-Machine-Learning-Review/tree/master/Educational) and [overview](https://github.com/dominusmi/Julia-Machine-Learning-Review/tree/master/Package%20Review).

![alt text](../master/material/packages.jpg)

MAJOR change in design [outlined here](https://nbviewer.jupyter.org/github/gbohner/Julia-Machine-Learning-Review/blob/f3482badf1f275e2a98bf7e338d406d399609f37/MLR/TuningDispatch_framework.ipynb)

**Summary available in [POSTER](../master/material/MLJ-JuliaCon2018-poster.pdf)**

MLJis an attempt to create a framework capable of easily tuning machine learning models.
Thanks to a solid abstraction layer, it allows user to easily add new models to its framework,
without losing any of the features.



**Landmarks:**

- [x] Implement first basic structure
- [x] Implement tuning for continuous parameters
- [x] Implement tuning for discrete parameters
- [x] Basic custom sampling method (K-fold)
- [x] Basic CV with custom score
- [x] Wrap at least a handful of models for regression & classification
- [x] Add multivariable regression methods
- [x] Add automatic labelling for classifiers
- [ ] Find a way to make it clear what arguments a model expects
- [ ] Allow any sampling methods from `MLBase.jl`
- [ ] Add compatibility with multiple targets
- [ ] move to more dispatch based system as outlined [HERE](https://nbviewer.jupyter.org/github/gbohner/Julia-Machine-Learning-Review/blob/f3482badf1f275e2a98bf7e338d406d399609f37/MLR/TuningDispatch_framework.ipynb)
- [ ] Upgrade it to Julia 1.0 [issue #4](https://github.com/alan-turing-institute/mlj/issues/4)


**Known Issues:**
- [ ] Fix stacking storage
- [ ] Get packages change Float to AbstractFloat so that forward diff can work


Notes: Forward diff does not work
