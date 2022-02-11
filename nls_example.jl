using Distributions, Plots, LsqFit

# Example 1
# recursive growth
# === Simulate data
# define non-linear function
bh(x, p) = x .* (p[1] ./ (1 .+ x ./ p[2]))

# define prameters 
R, k, sigma = 1.1, 100, 1

p = [R, k]
# simulate the predictor
n0 = [1:.5:50;]
n1 = bh(n0, p) .+ rand(Normal(0, sigma), length(n0))

plot(n0, n1, xlabel = "N0", ylabel = "N1")
Plots.abline!(1, 0, line=:dash, legend=:none)

# Fit bh model to the data using Non-linear least squares  
# estimate parameters
p0 = [0.5, 0.5]
fit = curve_fit(bh, n0, n1, p0)

fit.param
# ================

# Example 2
# growth over time
# === Simulate data
# define non-linear function
bht(t, p) = p[2]*p[3] ./ (p[3] .+ (p[2] - p[3]) * (p[1] .^-t) )

# define prameters 
n0 = 1
R, K, n0, sigma = 1.5, 20, 1, 2

p = [R, k, n0]
# simulate the predictor
t = [1.0:1:50;]
n = bht(t, p) .+ rand(Normal(0, sigma), length(t))

plot(t, n, xlabel = "t", ylabel = "N")

# Fit bh model to the data using Non-linear least squares  
# estimate parameters
p0 = [1.1, 20.0, 2.0]
fit = curve_fit(bht, t, n, p0)

fit.param
