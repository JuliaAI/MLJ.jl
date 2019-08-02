# NOTE: Copied from Julia's Test stdlib

"""
    @notest_logs [log_patterns...] [keywords] expression

Collect a list of log records generated by `expression` using
`collect_test_logs`, check that they match the sequence `log_patterns`, and
return the value of `expression`.  The `keywords` provide some simple filtering
of log records: the `min_level` keyword controls the minimum log level which
will be collected for the test, the `match_mode` keyword defines how matching
will be performed (the default `:all` checks that all logs and patterns match
pairwise; use `:any` to check that the pattern matches at least once somewhere
in the sequence.)

The most useful log pattern is a simple tuple of the form `(level,message)`.
A different number of tuple elements may be used to match other log metadata,
corresponding to the arguments to passed to `AbstractLogger` via the
`handle_message` function: `(level,message,module,group,id,file,line)`.
Elements which are present will be matched pairwise with the log record fields
using `==` by default, with the special cases that `Symbol`s may be used for
the standard log levels, and `Regex`s in the pattern will match string or
Symbol fields using `occursin`.

# Examples

Consider a function which logs a warning, and several debug messages:

    function foo(n)
        @info "Doing foo with n=\$n"
        for i=1:n
            @debug "Iteration \$i"
        end
        42
    end

We can test the info message using

    @notest_logs (:info,"Doing foo with n=2") foo(2)

If we also wanted to test the debug messages, these need to be enabled with the
`min_level` keyword:

    @notest_logs (:info,"Doing foo with n=2") (:debug,"Iteration 1") (:debug,"Iteration 2") min_level=Debug foo(2)

The macro may be chained with `@test` to also test the returned value:

    @test (@notest_logs (:info,"Doing foo with n=2") foo(2)) == 42

"""
macro test_mlj_logs(exs...)
    length(exs) >= 1 || throw(ArgumentError("""`@notest_logs` needs at least one arguments.
                               Usage: `@notest_logs [msgs...] expr_to_run`"""))
    patterns = Any[]
    kwargs = Any[]
    for e in exs[1:end-1]
        if e isa Expr && e.head == :(=)
            push!(kwargs, esc(Expr(:kw, e.args...)))
        else
            push!(patterns, esc(e))
        end
    end
    expression = exs[end]
    orig_expr = QuoteNode(expression)
    sourceloc = QuoteNode(__source__)
    Base.remove_linenums!(quote
        let testres=nothing, value=nothing
            try
                didmatch,logs,value = match_logs($(patterns...); $(kwargs...)) do
                    $(esc(expression))
                end
                if didmatch
                    testres = Test.Pass(:test, nothing, nothing, value)
                else
                    testres = Test.LogTestFailure($orig_expr, $sourceloc,
                                             $(QuoteNode(exs[1:end-1])), logs)
                end
            catch e
                testres = Error(:test_error, $orig_expr, e, catch_backtrace(), $sourceloc)
            end
            Test.record(Test.get_testset(), testres)
            value
        end
    end)
end

function match_logs(f, patterns...; match_mode::Symbol=:all, kwargs...)
    logs,value = collect_test_logs(f; kwargs...)
    if match_mode == :all
        didmatch = length(logs) == length(patterns) &&
            all(occursin(p, l) for (p,l) in zip(patterns, logs))
    elseif match_mode == :any
        didmatch = all(any(occursin(p, l) for l in logs) for p in patterns)
    end
    didmatch,logs,value
end
