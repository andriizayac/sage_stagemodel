using Plots

ulim = 2

# === Beverton-Holt Growth Model
function bvholt(N, R)
    R*N/(1 + (R-1)*N)
end

R = 2
N = [0:.01:ulim;]

# set up base plot
plot(N, bvholt.(N,R), label = "Beverton-Holt", legend = :bottomright, linewidth = 3)
xlims!((0, 2))
ylabel!("H_t+1")
xlabel!("H_t")
Plots.abline!(1, 0, line = :dash, label = "")

# === Logistic growth Model (aka Verhulst)
function logistic(N, r, k)
    r*N*(1 - N/k)
end

r = 2
k = 2
plot!(N, logistic.(N,r, k), label = "Logistic", axes=:none, linewidth = 3)
xlims!((0, 2))

# === Ricker
function ricker(N, r)
    N*exp(r*(1-N))
end

r = .5
plot!(N, ricker.(N,r), label = "Ricker - monotonic", axes=:none, linewidth = 3)
xlims!((0, 2))

# Ricker non-monotonic
r = 2
plot!(N, ricker.(N,r), label = "Ricker non-monotonic", axes=:none, linewidth = 3)
xlims!((0, 2))


# === Expornential growth function

plot(exp, xlims = [0, 5], linewidth = 4, label = :none, xlabel = "H_t", ylabel = "S_t+1", color = :darkblue)
