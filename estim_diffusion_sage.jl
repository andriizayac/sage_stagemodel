# This script derives estimates for R (growth), M (density dependence), and D (diffusion) in sagebrush using NLCD data and Bevorton-Holt population model.

# Step 1. Use Dataset 1 to get R_hat and M_hat by fitting a Beverton-Holt model.
# Step 2. Use R_hat and M_hat from Step 1 as fixed parameters in an integro-difference model to estimate D_hat

# === load packages
using Plots, StatsPlots, JLD, CSV, DataFrames
using LinearAlgebra, FFTW, SparseArrays, Optim
using Turing, Distributions, AdvancedHMC, Zygote

# === Step 0: [need only once, keep for records] prep the data for Step 1 and 2
using ArchGDAL

ber = ArchGDAL.readraster("data/berylfire_1988_wgs84utm12n_crop.tif") # Beryl dataset
buf = ArchGDAL.readraster("data/buffalofire_1995_wgs84utm11n_crop.tif") # Buffalo dataset

img_ber = Array{Int64}(ber[:,:,4:end]) 
img_buf = Array{Int64}(buf[:,:,11:end])

JLD.save("data/beryl.jld", "data", img_ber)
JLD.save("data/buffalo.jld", "data", img_buf)
# === end

# === Step 1: 
# = load Dateset 1 
buf = JLD.load("data/buffalo.jld", "data")
ngen1 = size(buf)[3]
# cast 3D array into a nested list
y1 = [buf[:,:,i] for i in 1:ngen1]

# = construct likelihood function
function loglike1(par)
        # decode data
        #y = input;
        ngen1 = length(y1);
        # decode the parameters
        R, M, sigma2 = par;
        # run time series loop: R*N/(1 + N/M)
        mu = [ R .* y1[j-1] ./ (1 .+ y1[j-1]/M) for j in 2:ngen1];
        # likelihood
        logl = 0
        for j in 2:ngen1
            logl += sum(loglikelihood.(Normal.(mu[j-1], exp(sigma2) ), y1[j])) 
        end
        return -logl
end
    
p01 = [0.1, 1, 1]
loglike1(p01)

# === use least squares estimator
fitl1 = optimize(loglike1, p01)
Optim.minimizer(fitl1)

R = Optim.minimizer(fitl1)[1]
M = Optim.minimizer(fitl1)[2]

# === Step 2
# = helper functions
getxy(x,y) = (x,y) # coordinate xy tuples
growth(N, R, M) = R .* N ./ (1 .+ N/M)
K2Ds(xy, D) = 1/(2pi*D) * exp(-norm( xy , 1)^2/(2D))
# ===

# Load Dataset 2 and using R, M as fixed parameters estimate Diffusion
# = load Dateset 2
ber = JLD.load("data/beryl.jld", "data")
ngen2 = size(ber)[3]
# cast 3D array into a nested list
y2 = [ber[:,:,i] for i in 1:ngen2]

# sptail grid and dispersal parameters 
# spatial data
n = size(ber)[1];  x = n*30; dx = 2*x/n; 
# define the spatial arrays in x and y
xf = [range(-x,  x-dx , length = n);] 
yf = [range(-x, x-dx , length = n);] 

xy = getxy.(xf' .* ones(n), ones(n)' .* yf)

# = construct likelihood function
function loglike2(par)
        # decode data
        #y = input;
        ngen2 = length(y2);
        # decode the parameters
        D, alpha, sigma2 = par;
        # Dispersal kernel and FFT 
        Fsker = fft(K2Ds.(xy, D))
        # run time series loop: R*N/(1 + N/M)
        mu = [ growth(y2[j-1], R,  M) .+ 
            alpha .* (1 .- y2[j-1]/M) .* 
            dx .* real.( fftshift( ifft(Fsker .* fft(y2[j-1]) )))  for j in 3:ngen2]

        # likelihood
        logl = sum([sum(loglikelihood.(Normal.(mu[j], exp(sigma2)), y2[j+2])) for j in 1:length(mu)])

        return -logl
end

# optimization 
p02 = [100.0, .5, 1.1]
loglike2(p02)

# === use least squares estimator
lower = [0.0, 0.0, 0]
upper = [Inf, Inf, Inf]
fitl2 = optimize(loglike2, lower, upper, p02)
Optim.minimizer(fitl2)

D = Optim.minimizer(fitl2)[1]
alpha = Optim.minimizer(fitl2)[2]

# === simulate data with point-estimated parameters
Fs = fft(K2Ds.(xy, D))
y2hat = [Float64.(y2[j]) for j in 1:ngen2];

for j in 3:ngen2
    global y2hat[j] = growth(y2hat[j-1], R,  M) .+ 
                        alpha .* (1 .- y2hat[j-1]/M) .* 
                        dx .* real.( fftshift( ifft(Fs .* fft(y2hat[j-1]) )))
end


anim = @animate for i in 1:ngen2
    plot(y2hat[i], st = :surface, camera = (30,50),
    zlims = (0, (R-1)*M),
    xlabel = "x", ylabel = "y",  zlabel = "Population size, H_t")
end every 1
gif(anim, "anim0_out.gif", fps = 4)

plot(y2hat[4], st = :surface, camera = (30,50), xlabel = "x", ylabel = "y",  zlabel = "Population size, H_t")

# ============== end of Step 1 and Step 2


# === Repeat Step 1 and Step 2 using Bayesian approach and Turing.jl package

# === Step 1 
y1data = [y1]

Turing.setadbackend(:forwarddiff)
@model function model1(data)
    y1 = data[1]

    R ~ Truncated(Normal(0, 2), 0, Inf)
    M ~ Truncated(Normal(0, 10), 0, Inf)
    sigma2 ~ Exponential(1)

    p = [R, M, sigma2]
    Turing.@addlogprob! -loglike1(p)
end

n = 500
n_chains = 4
n_adapt = 250
chains_model1 = sample(model1(y1data), Turing.NUTS(n_adapt, 0.65), MCMCThreads(), n, n_chains; discard_adapt=true)

chains_model1

phat1 = Array(chains_model1)[:, 1:2]

# === Step 2
y2data = [y2, xy, dx, R, M]

Turing.setadbackend(:zygote)
@model function model2(data)
    y2, xy, dx, R, M = data

    D ~ Truncated(Normal(0, 100^2), 0, Inf)
    alpha ~ Truncated(Normal(0, 50), 0, Inf)
    sigma2 ~ Exponential(1)

    p = [D, alpha, sigma2]
    Turing.@addlogprob! -loglike2(p)
end

chains_model2 = sample(model2(y2data), Turing.Gibbs(MH(:D, :alpha, :sigma2)), MCMCThreads(), n, n_chains; discard_adapt=true)

chains_model2

phat2 = Array(chains_model2)[:, 1:2]

# === write the estimated parameters into a CSV file
df = DataFrame(R=phat1[:,1], M=phat1[:,2], D=phat2[:,1], alpha = phat2[:,2])

CSV.write("parameter_estimates.csv", df)

# ====