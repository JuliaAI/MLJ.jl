To register all the models in GreatNewPackage with MLJ:

- In a clone of the master branch of MLJ, change to the
  `/src/registry/` directory and, in Julia, activate the environment
  specified by the Project.toml there, after checking the [compat]
  conditions thre are up to date.
  
- Add `GreatNewPackage` to the environment.

- Execute `using MLJ; MLJ.Registry.@update`. This updates
  `/Metadata.toml` and `/Models.toml` (the latter is generated for
  convenience and not used by MLJ).

-  Quit your REPL session, whose namespace is now polluted.

- Commit and make a PR request to merge your clone with master. Once
  merged, the new metadata is available to users of MLJ#master.
  
- Consider registering an new tagged version of MLJ. 

