# helper
function runcommand(cmd)
    @info cmd
    run(cmd)
end

function generate(dir; execute=true, pluto=true)
    quote
        using Pkg
        Pkg.activate(temp=true)
        Pkg.add(name="Literate", rev="fe/pluto")
        using Literate

        const OUTDIR = $dir
        const INFILE = joinpath(OUTDIR, "notebook.jl")

        # generate pluto notebook:
        if $pluto
            TEMPDIR = tempdir()
            Literate.notebook(INFILE, TEMPDIR, flavor=Literate.PlutoFlavor())
            runcommand(`mv $TEMPDIR/notebook.jl $OUTDIR/notebook.pluto.jl`)
        end

        Literate.notebook(INFILE, OUTDIR, execute=false)
        runcommand(
            `mv $OUTDIR/notebook.ipynb $OUTDIR/notebook.unexecuted.ipynb`)
        $execute && Literate.notebook(INFILE, OUTDIR, execute=true)

    end |> eval
end

# Pkg.add("Pluto")
# using Pluto
# Pluto.run(notebook=joinpath(OUTDIR, "notebook.pluto.jl"))

# Pkg.add("IJulia")
# Pkg.instantiate()
# using IJulia
# IJulia.notebook(dir=OUTDIR)
