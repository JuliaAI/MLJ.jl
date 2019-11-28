# Instructions for updating the MLJ Model Registry

To register all the models in GreatNewPackage with MLJ:

- In a clone of the master branch of
  [MLJModels](https://github.com/alan-turing-institute/MLJModels.jl),
  change to the `/src/registry/` directory and, in Julia, activate the
  environment specified by the Project.toml there, after checking the
  [compat] conditions there are up to date.
  
- Add `GreatNewPackage` to the environment.

- In some environment in which your MLJModels clone has been added
  using `Pkg.dev`, execute `using MLJModels; @update`. This updates
  `src/registry/Metadata.toml` and `src/registry/Models.toml` (the
  latter is generated for convenience and not used by MLJ).

-  Quit your REPL session, whose namespace is now polluted.

- Push your changes to an appropriate branch of MLJModels to make
  the updated metadata available to users of the next MLJModels tagged
  release.
  
