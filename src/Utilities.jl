function Fakedata(N,d)
    n_obs = 100
    x = randn((n_obs,d))
    y = sum(x*randn(d),2)+randn(n_obs)*0.1

    hcat(x,y)
end

function FakedataClassif(N,d)
    n_obs = 100
    x = randn((n_obs,d))
    y = ( sum(x*randn(d),2) .> mean(sum(x*randn(d),2)) )

    hcat(x,y)
end