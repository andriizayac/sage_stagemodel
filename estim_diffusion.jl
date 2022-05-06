# This script simulates a simple diffusion process and develops a stat model to recover the diffusion parameter

# === load packages
using Plots, FFTW, LinearAlgebra, SparseArrays, FFTW

# === helper function
include("helper_fns.jl") 
# ===

# Contents:
# 1. Plain diffusion case 
# 2. Stage structured reaction-diffusion case
# 3. Fit IDE model to real data

# ==============================================================================================
# ===  1. Plain diffusion case
# ==============================================================================================

# === 1.a Simulate data

n = 32;  x = 10; dx = 2*x/n; 
# define the spatial arrays in x and y
xf = [range(-x,  x-dx , length = n);] 
yf = [range(-x, x-dx , length = n);] 
	
# number of generation
ngen = 30
	
# diffusion coefficient
D = 3
	
# combine spatial arrays into grid
XF = xf' .* ones(n)
YF = ones(n)' .* yf
	
# set up initial conditions	
h0 = (abs.(XF) .<= 1) .* (abs.(YF) .<= 1) .* 100
plot(h0, st = :surface, camera=(20,50))

# dispersal kernel
xy = getxy.(xf' .* ones(n), ones(n)' .* yf)
K2Ds(xy, D) = 1/(2pi*D) * exp(-norm( xy , 1)^2/(2D))
Fsker = fft(K2Ds.(xy, D))

# initiate
hmat = zeros(n, n, ngen)
hmat[:,:,1] = h0

for j = 2:ngen
	global hmat[:,:,j] = dx .* real( fftshift( ifft(Fsker .* fft(hmat[:,:,j-1])) ) )
end

anim = @animate for i in 1:30
    plot(hmat[:,:,i], st = :surface,
      xlabel = "x", ylabel = "y",  zlabel = "Population size, H_t")
end every 1
gif(anim, "anim0_out.gif", fps = 2)

# === output
ydat = hmat


# === 1.b Fit the simulated model to the data
using Distributions, Optim, LinearAlgebra, Plots, StatsPlots
using Turing, AdvancedHMC, Zygote

# --- reshape and add random noise to data
ydatm = hmat + reshape(randn(length(hmat)), 32, 32, 30)

y = [ydatm[:,:,j] for j in 1:size(ydatm)[3]]
fydatm = [fft(ydatm[:,:,j]) for j in 1:size(ydatm)[3]]


anim = @animate for i in 1:30
    plot(y[i], st = :surface,
      xlabel = "x", ylabel = "y",  zlabel = "Population size, H_t")
end every 1
gif(anim, "anim0_out.gif", fps = 2)

# --- define the likelihood function for the turing model
function loglikeT(input, par)
    # decode data
    y, fydatm, xy, dx = input
    ngen = length(y);
    # decode the parameters
    D, sigma2 = par;
    # Dispersal kernel and FFT
    Fsker = fft(K2Ds.(xy, D));
    # run time series loop
    mu = [dx .* real(fftshift(ifft(Fsker .* fydatm[j-1]))) for j in 2:ngen];
    # likelihood
    logl = sum([sum(loglikelihood.(Normal.(mu[j-1], sigma2), y[j])) for j in 2:ngen])
    return logl
end

Turing.setadbackend(:zygote)
Turing.@model function demo(data)
    # priors 
    b0 ~ Normal(0, 10)
    b1 ~ Exponential(10)
    # aggregate parameters
    p = [b0, b1]
    # minimize nagative log-likelihood (ie, maximize likelihood) 
    Turing.@addlogprob! loglikeT(data, p)
end

input = [y, fydatm, xy, dx]
nchains = 1
n = 2000
outcome1 = sample(demo(input), Gibbs(MH(:b0), MH()), n)


plot(outcome1[1000:end,:,:])


# JLD.save("MH_diff_v0.jld", "outocme1", outcome1)
# outcome = JLD.load("hmc_diff_v0.jld")
# === 1. diffusion case


# ==============================================================================================
# 2. === full simulation and stat model
# ==============================================================================================
using Turing, FFTW, LinearAlgebra
using JLD
ydat, p, xy, dx = JLD.load("hmat_128.jld", "hmat", "p", "xy", "dx") 
ydat = ydat + reshape(randn(length(ydat)), size(ydat))/10

n = size(ydat)[1]
ngen = size(ydat)[3]

anim = @animate for i in 1:ngen
    plot(ydat[:,:,i], st = :surface,
      xlabel = "x", ylabel = "y",  zlabel = "Population size, H_t", title = i)
end every 3
gif(anim, "anim0_out.gif", fps = 3)

y = [ydat[:,:,j] for j in 1:ngen]
fydatm = [fft(y[j]) for j in 1:ngen]

# === constrict a likelihood function

# = Gussian ll
function ide_model(input, p)
        y, fydatm, xy, dx = input
        ngen = length(y)
        # decode the parameters
        R, M, alpha, D, sigma2 = p
        # Dispersal kernel and FFT
        Fsker = fft(K2Ds.(xy, D))
        # run time series loop
        mu = [y[j-1] .+ growth(y[j-1], R,  M) .+ 
            alpha .*(1 .- y[j-1]/M) .* 
            real.( fftshift( ifft(Fsker .* fydatm[j-1] ))) for j in 2:ngen]

        # likelihood - 
        logl = sum([sum(loglikelihood.(Normal.(mu[j-1], sigma2), y[j])) for j in 2:ngen])
        return logl
end



input = [y, fydatm, xy, dx]
growth(N, a, b) = bvholt(N, a, b)
Fsker = fft(K2Ds.(xy, D))

mu = [y[j-1] .+ growth(y[j-1], 1.5,  10.) .+ 
            0.01 .*(1 .- y[j-1]/10.) .* 
            real.( fftshift( ifft(Fsker .* fydatm[j-1] ))) for j in 2:ngen]
            
rho = rand(5)
a = ide_model(input, [p; .1])

Turing.setadbackend(:zygote)
@model function ide_fit(data)
    R ~ Normal(0, 10)
    M ~ Gamma(3,20)
    alpha ~ Exponential(1)
    D ~ Normal(0,10)
    sigma2 ~ Exponential(10)

    p = [R, M, alpha, D, sigma2]
    Turing.@addlogprob! ide_model(data, p)
end

n = 2000
chains = sample(ide_fit(input), Gibbs(MH(:R)), n)


plot(chains[500:end,:,:])


# ==============================================================================================
# 3. === Fit IDE model to real data
# ==============================================================================================
using JLD
sage = JLD.load("data/Buffalo_1995.jld", "sage")[1:32, 1:32,11:end]

ngen = size(sage)[3]
y = [sage[:,:,j] for j in 1:ngen]
logy = [log.(sage[:,:,j] .+ 0.1) for j in 1:ngen]
fydatm = [fft(logy[j]) for j in 1:ngen]

anim = @animate for i in 1:size(sage)[3]
    plot(y[i], st = :surface,
      xlabel = "x", ylabel = "y",  zlabel = "Population size, H_t", title = i)
end every 1
gif(anim, "anim0_out.gif", fps = 2)

# spatial data
n = size(sage)[1];  x = n*30; dx = 2*x/n; 
# define the spatial arrays in x and y
xf = [range(-x,  x-dx , length = n);] 
yf = [range(-x, x-dx , length = n);] 

xy = getxy.(xf' .* ones(n), ones(n)' .* yf)
growth(N, a, b) = bvholt(N, a, b)
K2Ds(xy, D) = 1/(2pi*D) * exp(-norm( xy , 1)^2/(2D))

# = Poisson ll
function ide_model_poisson(input, p)
    y, logy, fydatm, xy, dx = input
    ngen = length(y)
    # decode the parameters
    R, M, alpha, D = p
    # Dispersal kernel and FFT
    Fsker = fft(K2Ds.(xy, D))
    # run time series loop
    mu = [ logy[j-1] .+ growth(logy[j-1], R,  M) .+ 
        alpha .*(1 .- logy[j-1]/M) .* 
        real.( fftshift( ifft(Fsker .* fydatm[j-1] )))  for j in 2:ngen]

    # likelihood - 
    logl = sum([sum(loglikelihood.(Poisson.( exp.(mu[j-1])), y[j])) for j in 2:ngen])
    return logl
end

input = [y, logy, fydatm, xy, dx]

p = [1, 10, 0.01, 2]
ll = ide_model_poisson(input, p)
