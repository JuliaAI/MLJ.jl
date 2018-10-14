using LossFunctions: @_dimcheck

function squared_error(y_true, y_pred)
    # code taken from https://github.com/JuliaML/MLMetrics.jl/blob/master/src/regression.jl#L16-L19
    @_dimcheck length(y_true) == length(y_pred)
    return((y_true - y_pred) .^ 2)
end

function mean_squared_error(y_true, y_pred)
    # code taken from https://github.com/JuliaML/MLMetrics.jl/blob/master/src/regression.jl#L56-L59
    @_dimcheck length(y_true) == length(y_pred)
    return(mean(squared_error(y_true, y_pred)))
end