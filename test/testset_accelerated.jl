using Distributed
import ComputationalResources: CPU1, CPUProcesses, CPUThreads

macro testset_accelerated(name::String, var, ex)
    testset_accelerated(name, var, ex)
end
macro testset_accelerated(name::String, var, opts::Expr, ex)
    testset_accelerated(name, var, ex; eval(opts)...)
end
function testset_accelerated(name::String, var, ex; exclude=[])
    final_ex = quote
        $var = CPU1()
        @testset $name $ex
    end
    resources = Any[CPUProcesses()]
    @static if VERSION >= v"1.3.0-DEV.573"
        push!(resources, CPUThreads())
    end
    for res in resources
        if any(x->typeof(res)<:x, exclude)
            push!(final_ex.args, quote
                $var = $res
                @testset $(name*" (accelerated with $(typeof(res).name))") begin
                    @test_broken false
                end
            end)
        else
            push!(final_ex.args, quote
                $var = $res
                @testset $(name*" (accelerated with $(typeof(res).name))") $ex
            end)
        end
    end
    return esc(final_ex)
end
