# Composing Models

Three common ways of combining multiple models together have out-of-the-box
implementations in MLJ:

- [Linear Pipelines](@ref) (`Pipeline`)- for unbranching chains that take the
  output of one model (e.g., dimension reduction, such as `PCA`) and
  make it the input of the next model in the chain (e.g., a
  classification model, such as `EvoTreeClassifier`). To include
  transformations of the target variable in a supervised pipeline
  model, see [Target Transformations](@ref).
  
- [Homogeneous Ensembles](@ref) (`EnsembleModel`) - for blending the
  predictions of multiple supervised models all of the same type, but
  which receive different views of the training data to reduce overall
  variance. The technique implemented here is known as observation
  [bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating). 
  
- [Model Stacking](@ref) - (`Stack`) for combining the predictions of a smaller
  number of models of possibly *different* types, with the help of an
  adjudicating model.
  
Additionally, more complicated model compositions are possible using:

- [Learning Networks](@ref) - "blueprints" for combining models in flexible ways; these
  are simple transformations of your existing workflows which can be "exported" to define
  new, stand-alone model types.
