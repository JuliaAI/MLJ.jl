# Acceleration and Parallelism

!!! warning "Experimental API"

    The acceleration API is experimental and may not work correctly in all
    cases, especially if trying to use an acceleration method that your
    version of Julia or installed packages cannot support. The API is also
    subject to breaking changes during minor or major releases without
    warning.

## User-facing interface

To enable composable, extensible acceleration of core MLJ methods,
[ComputationalResources.jl](https://github.com/timholy/ComputationalResources.jl)
is utilized to provide some basic types and functions to make implementing
acceleration easy. However, ambitious users or package authors have the option
to define their own types to be passed as resources to `acceleration`, which
must be `<:ComputationalResources.AbstractResource`.

Methods which support some form of acceleration support the `acceleration`
keyword argument, which can be passed a "resource" from
`ComputationalResources`. For example, passing `acceleration=CPUProcesses()`
will utilize `Distributed`'s multiprocessing functionality to accelerate the
computation, while `acceleration=CPUThreads()` will use Julia's PARTR
threading model to perform acceleration.

The default computational resource is `CPU1()`, which is simply serial
processing via CPU. The default resource can be changed as in this
example: `MLJ.default_resource(CPUProcesses())`. The argument must
always have type `<:ComputationalResource.AbstractResource`. To
inspect the current default, use `MLJ.default_resource()`.

!!! note

    The `CPUThreads()` resource is only available when running a version of
    Julia with `Threads.@spawn` available.
