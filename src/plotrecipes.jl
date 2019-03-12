@recipe function f(mach::MLJ.Machine{<:MLJ.EitherTunedModel})
    r = report(mach)
    z = r.measurements
    x = r.parameter_values[:,1]
    y = r.parameter_values[:,2]
    xsc, ysc = r.parameter_scales

    xguide --> r.parameter_names[1]
    yguide --> r.parameter_names[2]
    xscale --> (xsc == :linear ? :identity : xsc)
    yscale --> (ysc == :linear ? :identity : ysc)

    st = get(plotattributes, :seriestype, :scatter)

    if st ∈ (:surface, :heatmap, :contour, :contourf, :wireframe)
        ux = unique(x)
        uy = unique(y)
        m = reshape(z, (length(ux), length(uy)))'
        ux, uy, m
    else
        label --> ""
        seriestype := st
        ms = get(plotattributes, :markersize, 4)
        markersize := 3ms * sqrt.(z)
        marker_z --> z
        x, y
    end
end
