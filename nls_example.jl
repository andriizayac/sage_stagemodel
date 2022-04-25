using Distributions, Plots, LsqFit

# Example 1
# recursive growth
# === Simulate data
# define non-linear function
bh(x, p) = x .* (p[1] ./ (1 .+ x .* p[2]))

# define prameters 
R, k, sigma = 1.2, 1/50, 1

p = [R, k]
# simulate the predictor
# n0 = [1.0;] 
# n1 = n0
# [push!(n1, bh(n1[j-1], p) + rand(Normal(0, sigma), 1)[1]) for j in 2:50]
n0 = [1:.5:50;]
n1 = bh(n0, p) .+ rand(Normal(0, sigma), length(n0))



scatter(n0, n1, xlabel = "N0", ylabel = "N1", label="BH growth", legend=:bottomright)
Plots.abline!(1, 0, line=:dash, label="1:1")

# Fit bh model to the data using Non-linear least squares  
# estimate parameters
p0 = [0.5, 0.5]
fit = curve_fit(bh, n0, n1, p0)

fit.param
p95 = confidence_interval(fit, 0.05)

phat = [fit.param[1], fit.param[2]]
plot!(n0, bh(n0, phat), color=:purple, label="Fit")

# ================ end

# Example 2
# growth over time
# === Simulate data
# define non-linear function
bht(t, p) = p[2]*p[3] ./ (p[3] .+ (p[2] - p[3]) * (p[1] .^-t) )

# define prameters 
n0 = 1
R, K, n0, sigma = 1.5, 20, 1, 2

p = [R, K, n0]
# simulate the predictor
t = [1.0:1:50;]
n = bht(t, p) .+ rand(Normal(0, sigma), length(t))

scatter(t, n, xlabel = "t", ylabel = "N", label="N")

# Fit bh model to the data using Non-linear least squares  
# estimate parameters
p0 = [1.1, 20.0, 2.0]
fit = curve_fit(bht, t, n, p0)

phat = fit.param
nhat = bht(t, phat)
plot!(t, nhat, color=:purple, label="N_hat")
# ========= end
