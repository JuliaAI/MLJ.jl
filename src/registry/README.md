## MLJRegistry

A package registry for the Julia machine learning framework
[MLJ](https://github.com/alan-turing-institute/MLJ.jl).

[Models in registered packages](Models.toml)

[Model metadata](Metadata.toml)

Any Julia machine learning model that implements the MLJ interface is
immediately available for use by MLJ. However, models in
*registered* packages are discoverable by all MLJ users - whether or not the packages have been imported - through MLJ's
task interface.

<!-- [![Build -->
<!-- Status](https://travis-ci.com/alan-turing-institute/MLJRegistry.jl.svg?branch=master)](https://travis-ci.com/alan-turing-institute/MLJRegistry.jl) -->


### Background

MLJ is a Julia framework for combining and tuning machine learning
models. To implement a model see the instructions in the MLJ document
["Adding Models for General Use"](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/)
and the MLJ
[CONTRIBUTE.md](https://github.com/alan-turing-institute/MLJ.jl/blob/master/CONTRIBUTE.md)



### How to register a package

For now, new packages can be registered by creating an issue on this repository. 


#### For adminsistrators

To register GreatNewPackage:

- Clone the `MLJRegistry` repository, activate the clone's environment
in the Julia package manager and `add GreatNewPackage` (which adds GreatNewPackage to MLJRegistry/Project.toml).

- In a clean Julia REPL session, run `import MLJRegistry` and
  `@update` (which updates the `Metadata.toml` and
  `Models.toml` files). Quit your REPL session, which is now polluted.

- Commit and make a PR request at [MLJRegistry](https://github.com/alan-turing-institute/MLJRegistry.jl). Once merged, the new metadata is available to MLJ.



