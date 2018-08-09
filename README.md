## MLJ Machine Learning Models Tuning Library - 100% Made in Julia


MAJOR change in design [outlined here](https://github.com/dominusmi/Julia-Machine-Learning-Review/pull/24)

γₘₗ is an attempt to create a framework capable of easily tuning machine learning models.
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


**Known Issues:**
- [ ] Fix stacking storage
- [ ] Get packages change Float to AbstractFloat so that forward diff can work


Notes: Forward diff does not work
