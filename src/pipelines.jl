
## LINEAR LEARNING NETWORKS (FOR INTERNAL USE ONLY)

# constructs and returns an unsupervised node, and its machine, for a
# given model and input. the node is static if model is a function
# instead of bona fide model. Example: `node_(pca, X)` (dynamic) or
# `node_(MLJ.matrix, X)` (static).
function node_(model, X)
    if model isa Model
        mach = machine(model, X)
        return transform(mach, X), mach
    else
        n = node(model, X)
        return n, n.machine
    end
end

# `models_and_functions` can include both functions (static
# operatations) and bona fide models. If `ys == nothing` the learning
# network is assumed to be unsupervised. Otherwise: If `target ===
# nothing`, then no target transform is applied; if `target !==
# nothing` and `inverse === nothing`, then corresponding `transform`
# and `inverse_transform` are applied; if neither `target` nor
# `inverse` are `nothing`, then both `target` and `inverse` are
# assumed to be StaticTransformations, to be applied
# respectively to `ys` and to the output of the `models_and_functions`
# pipeline. Note that target inversion is applied to the output of the
# *last* nodal machine in the pipeline, corresponding to the last
# element of `models_and_functions`.

# No checks whatsoever are performed. Returns a learning network.
function linear_learning_network(Xs, ys, target, inverse, models_and_functions...)

    n = length(models_and_functions)

    if ys !== nothing && target !== nothing
        yt, target_machine = node_(target, ys)
    else
        yt  = ys
    end
    
    nodes = Vector(undef, n + 1)
    nodes[1] = Xs
    
    for i = 2:(n + 1)
        m = models_and_functions[i-1]
        if m isa Supervised
            supervised_machine = machine(m, nodes[i-1], yt)
            nodes[i] = predict(supervised_machine, nodes[i-1])
       else
            nodes[i] = node_(m, nodes[i-1]) |> first
        end
    end

    if target === nothing
        terminal_node=nodes[end]
    else
        if inverse === nothing
            terminal_node = inverse_transform(target_machine, nodes[end])
        else
            terminal_node = node_(inverse, nodes[end]) |> first
        end
    end
 
    return terminal_node

end


## PREPROCESSING

function eval_and_reassign(modl, ex)
    s = gensym()
    evaluated = modl.eval(ex)
    modl.eval(:($s = $evaluated))
    return s, evaluated
end

pipe_alert(message) = throw(ArgumentError("@pipeline error.\n"*
                                     string(message)))
pipe_alert(k::Int) = throw(ArgumentError("@pipeline error $k. "))

# does expression processing, syntax and semantic
# checks. is_probabilistic is `true`, `false` or `missing`.
function pipeline_preprocess(modl, ex, is_probabilistic::Union{Missing,Bool})

    ex isa Expr || pipe_alert(2)
    length(ex.args) > 1 || pipe_alert(3)
    ex.head == :call || pipe_alert(4)
    pipetype_ = ex.args[1]
    pipetype_ isa Symbol || pipe_alert(5)

    fieldnames_ = []               # for `from_network`
    models_ = []                   # for `from_network`
    models_and_functions_ = []     # for `linear_learning_network`
    models_and_functions  = []     # for `linear_learning_network`
    for ex in ex.args[2:end]
        if ex isa Expr
            if ex.head == :kw 
                variable_ = ex.args[1]
                variable_ isa Symbol || pipe_alert(8)
                value_, value = eval_and_reassign(modl, ex.args[2])
                if variable_ == :target
                    if value isa Function
                        value_, value =
                            eval_and_reassign(modl,
                                              :(MLJ.StaticTransformer($value)))
                    end
                    value isa Unsupervised || pipe_alert(9)
                    target_ = value_
                    target = value
                    push!(fieldnames_, :target)
                elseif variable_ == :inverse
                    if value isa Function
                        value_, value =
                            eval_and_reassign(modl,
                                              :(MLJ.StaticTransformer($value)))
                    else
                        pipe_alert(10)
                    end
                    inverse_ = value_
                    inverse = value
                    push!(fieldnames_, :inverse)
                else
                    value isa Model || pipe_alert(11)
                    push!(models_and_functions_, value_)
                    push!(models_and_functions, value)
                    push!(fieldnames_, variable_)
                end
                push!(models_, value_)
            else
                f = modl.eval(ex)
                f isa Function ||
                    pipe_alert("Perhaps a missing name, as `name=$f'?")
                push!(models_and_functions_, ex)
                push!(models_and_functions, f)
            end
        else
            f = modl.eval(ex)
            f isa Function || pipe_alert(7)
            push!(models_and_functions_, ex)
            push!(models_and_functions, f)
        end
    end
    
    (@isdefined target)  || (target = nothing; target_ = :nothing)
    (@isdefined inverse) || (inverse = nothing; inverse_ = :nothing)
    inverse !== nothing && target === nothing &&
        pipe_alert("You have specified `inverse=...` but no `target`.")

    supervised(m) = !(m isa Unsupervised) && !(m isa Function)
    
    supervised_components = filter(supervised, models_and_functions)
        
    length(supervised_components) < 2 ||
        pipe_alert("More than one component of the pipeline is a "*
                   "supervised model .")

    is_supervised  =
        length(supervised_components) == 1
    
    # `kind_` is defined in composites.jl:
    kind = kind_(is_supervised, is_probabilistic)
    ismissing(kind) &&
        pipe_alert("Network has no supervised components and so "*
                  "`is_probabilistic=true` "*
                  "declaration is not allowed. ")

    target isa StaticTransformer && inverse == nothing &&
        pipe_alert("It appears `target` is a function. "*
                   "You must therefore specify `inverse=...` .")

    target == nothing || is_supervised ||
        pipe_alert("`target=...` has been specified but no "*
                   "supervised components have been specified. ")
        
    return (pipetype_, fieldnames_, models_,
            models_and_functions_, target_, inverse_, kind)

end

function pipeline_preprocess(modl, ex, kw_ex)
    kw_ex isa Expr || pipe_alert(10)
    kw_ex.head == :(=) || pipe_alert(11)
    kw_ex.args[1] == :is_probabilistic ||
        pipe_alert("Unrecognized keywork `$(kw_ex.args[1])`.")
    value_ = kw_ex.args[2]
    if value_ == :missing
        value = missing
    else
        value = value_
        value isa Bool ||
            pipe_alert("`is_probabilistic` can only be `true` or `false`.")
    end
    return pipeline_preprocess(modl, ex, value)
end

function pipeline_(modl, ex, kw_ex)

    (pipetype_, fieldnames_, models_, models_and_functions_,
     target_, inverse_, kind) =
         pipeline_preprocess(modl, ex, kw_ex)

    Xs = source(nothing)
    
    if kind === :UnsupervisedNetwork
        ys = source(nothing, kind=:target)
    else
        ys = nothing
    end

    ms = gensym(:ms)

    N_ex = quote
        $ms = $target_
        
        MLJ.linear_learning_network(Xs, ys,  $target_, $inverse_,
                                $(models_and_functions_...))
    end
                                               
    from_network_(modl, pipetype_, fieldnames_, models_, N_ex, kind)
    
    return pipetype_

end

pipeline_(modl, ex) = pipeline_(modl, ex, :(is_probabilistic=missing)) 

macro pipeline(exs...)
    pipetype_ = pipeline_(__module__, exs...)

    esc(quote
        $pipetype_()
        end)

end
    
    



    
        
        
