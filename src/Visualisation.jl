using Plots
plotlyjs()


function plot_storage(storage::MLJStorage; plotting_args=[])
    models = Set(storage.models)
    fig = plot(;plotting_args...)

    for model in models
        markers = []
        measures = []
        indeces = []

        for (i, p_models) in enumerate(storage.models)
            if p_models == model
                push!(indeces, i)
            end
        end

        if model == "multivariate"
            println("averageCV: $(storage.averageCV[indeces])")
        end
        measures = storage.averageCV[indeces]
        for dict in storage.parameters[indeces]
            _marker = ""
            for (key, value) in dict
                _marker = _marker * "$key: $value, "
            end
            push!(markers, _marker)
        end
        x = collect(0+1/length(measures):1/length(measures):1)
        plot!(x, measures, hover=markers, label=model)
    end
    fig
end
