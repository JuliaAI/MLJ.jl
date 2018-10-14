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

function accuracy(targets::AbstractArray,
    outputs::AbstractArray,
    encoding::BinaryLabelEncoding;
    normalize = true)
@_dimcheck length(targets) == length(outputs)
tp::Int = 0; tn::Int = 0
@inbounds for i = 1:length(targets)
target = targets[i]
output = outputs[i]
tp += true_positives(target, output, encoding)
tn += true_negatives(target, output, encoding)
end
correct = tp + tn
normalize ? Float64(correct/length(targets)) : Float64(correct)
end

function accuracy(object; normalize = true)
correct = true_positives(object) + true_negatives(object)
normalize ? Float64(correct/nobs(object)) : Float64(correct)
end

function accuracy(targets::AbstractArray,
    outputs::AbstractArray,
    encoding::LabelEncoding;
    normalize = true)
@_dimcheck length(targets) == length(outputs)
correct::Int = 0
@inbounds for i = 1:length(targets)
correct += targets[i] == outputs[i]
end
normalize ? Float64(correct/length(targets)) : Float64(correct)
end

function accuracy(targets::AbstractArray,
    outputs::AbstractArray,
    labels::AbstractVector;
    normalize = true)
accuracy(targets, outputs, LabelEnc.NativeLabels(labels), normalize = normalize)::Float64
end

function accuracy(targets::AbstractArray,
    outputs::AbstractArray;
    normalize = true)
accuracy(targets, outputs, _labelenc(targets, outputs), normalize = normalize)::Float64
end