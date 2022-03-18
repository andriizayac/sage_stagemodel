using Distributions, Optim, LinearAlgebra, Plots, StatsPlots
using CSV, DataFrames

# Import data
df = DataFrame(CSV.File("data/nlcd_sage_beryl_1988.csv"))

# subset and covert to matrix
dfn = Matrix{Int64}(df[1:30,:])
# plot the trajectories
plot(dfn', legend=:none)

# reshape the data
Y = vec(dfn[:, 2:end])
X = vec(dfn[:, 1:end-1])

Y = Y[X .> 0]
X = log.(X[X .> 0])

# define a simple Beverton-Holt (or some other growth) function
bh(x,p) = x .* (p[1] ./ (1 .+ x ./ p[2]))

# --- Point-estimate log-likelihood function
function loglike(rho)
    beta = rho[1:2]
    theta = exp.(bh(X, beta))
    dist = Poisson.(theta)
    lp = logpdf.(dist, Y)
    logl = sum(lp)
    return -logl
end

p0 = [1.1, 200] # initial parameter values
res = optimize(loglike, p0)
est = res.minimizer
# print estimated parameters
println(est)

# plot the original data and simulated trajectory
#  define the analytical solution for BH model
bht(t, p) = p[2]*p[3] ./ (p[3] .+ (p[2] - p[3]) * (p[1] .^-t) )
# set initial conditions and time array
n0 = 1.0
t = [1.0:1:50;]
# compute the carrying capacity
K = (est[1] - 1) * est[2]
# combine estimated parameters
phat = [est[1], K, n0]
# simulate and plot predicted cover
yhat = exp.(bht(t, phat)) #+ rand(Normal(0, est[3]), length(t))
plot(dfn', legend=:none)
plot!(t, yhat, linewidth=2, legend=:none)

# === add stage structure, no spatial dependence yet
# we can use the same framework, only now our function is a second-order autoregressive time series that accounts for 
Y = vec(dfn[:, 3:end])
X1 = vec(dfn[:, 2:end-1])
X2 = vec(dfn[:, 1:end-2])
# p = [R0, M, alpha]
function bhstage(X1, X2, p) 
    k = (p[1]-1)*p[2]
    return X1 .* (p[1] ./ (1 .+ X1 / p[2])) .+ p[3] .* (1.0 .- X1/k ) .* X2
end
p = [1.1, 100, .01]
bhstage(X1, X2, p)

# --- Point-estimate
function loglike(rho)
    beta = rho[1:3]
    sigma2 = exp(rho[4])
    residual = Y - bhstage(X1, X2, beta)
    dist = Normal(0, sqrt(sigma2))
    lp = logpdf.(dist, residual)
    logl = sum(lp)
    return -logl
end

# define positive constraints and initial parameters
lower = [0.0, 0, 0, 0]
upper = [Inf, Inf, Inf, Inf]
p0 = [1.1, 200, .1, .5] # initial values
# run optimization
res = optimize(loglike, lower, upper, p0)
est = res.minimizer
est[4] = exp(est[4])
# --- end


# === Turing implementation
using Turing
n = length(Y)

@model poisson_regression(X, Y, n) = begin
    b0 ~ truncated(Normal(0, 1), 0, Inf)
    b1 ~ truncated(Normal(0, 10), 0, Inf)

    for i = 1:n
        theta = X[i] * (b0 / (1 + X[i] / b1))
        Y[i] ~ Poisson(exp(theta))
    end
end;

# Sample using NUTS.

num_chains = 4
m = poisson_regression(Y, Y, n)
chain = sample(m, NUTS(200, 0.65), MCMCThreads(), 500, num_chains; discard_adapt=false)

# Taking the first chain
c1 = chain[:,:,1]

# Calculating the exponentiated means
b0_exp = exp(mean(c1[:b0]))
b1_exp = exp(mean(c1[:b1]))


# === read and analyze raster data
using ArchGDAL
img0 = ArchGDAL.readraster("./../../../Downloads/tmp/berylfire_crop.tif")

img = Array{Int64}(img0[:,:,4:end]) 


# define a simple Beverton-Holt (or some other growth) function
bh(x,p) = x .* (p[1] ./ (1 .+ x ./ p[2]))

# --- Poisson stuff
function loglike(rho)
    beta = rho[1:2]
    logl = 0
    for i in 2:size(img)[3]
        X = vec(img[:,:,i-1]); Y = vec(img[:,:,i])
        x = log.(X[X .> 0]); y = Y[X .> 0];
        # x = X; y = Y;
        theta = exp.( bh(x, beta) )
        dist = Poisson.(theta)
        lp = logpdf.(dist, y)
        logl += sum(lp)
    end
    return -logl
end

p0 = [1.1, 200] # initial parameter values
res = optimize(loglike, p0)
est = res.minimizer
# print estimated parameters
println(est)

# plot the original data and simulated trajectory
#  define the analytical solution for BH model
bht(t, p) = p[2]*p[3] ./ (p[3] .+ (p[2] - p[3]) * (p[1] .^-t) )
# set initial conditions and time array
n0 = .5
t = [1.0:1:50;]
# compute the carrying capacity
K = (est[1] - 1) * est[2]
# combine estimated parameters
phat = [est[1], K, n0]
# simulate and plot predicted cover
yhat = exp.(bht(t, phat)) #+ rand(Normal(0, est[3]), length(t))
plot(dfn', legend=:none)
plot!(t, yhat, linewidth=4, legend=:none)


# ===== test
# --- stage-structure stuff using Normal likelihood
using FFTW
include("helper_fns.jl") 
inf(f, xs, ys, D) = [f(x,y, D) for x in xs, y in ys]

# current NLCD subset is 32 x 32 pixels
n = 32;  x = 960; dx = 2*x/n; 
# define the spatial arrays in x and y
xf = [range(-x,  x-dx , length = n);] #.+ 10
yf = [range(-x, x-dx , length = n);] #.+ 10
# combine spatial arrays into grid
XF = xf' .* ones(n)
YF = ones(n)' .* yf	
#  choose a dispersal kernel: K2DL (Laplace), K2DG (Gaussian)
K2Ds(x, y, D) =  1/(2pi*D^2) * exp(-norm( (x, y) , 1)^2/(2D^2))


function loglike(rho)
    # decode the parameters
    beta = rho[1:3]
    Diff = rho[4]
    sigma2 = exp(rho[5])
    # initialize storage output
    s = []
    push!(s, img[:,:,1])
    logl = 0
    # Dispersal kernel and FFT
    sker = inf(K2Ds, xf, yf, Diff)
    Fsker = fft(sker)
    # run time series loop
    for i in 2:size(img)[3]
        y0 = img[:,:,i-1]; y = img[:,:,i]; 
        # process model 
        k = (beta[1]-1) * beta[2]
        mu = bh(y0, beta[1:2]) .+ beta[3] .* (1.0 .- y0/k ) .* s[i-1]
        sl = fft(y0)
        push!(s, real(fftshift(ifft(Fsker .* sl))) )
        
        # likelihood
        dist = Normal.(mu, sqrt(sigma2) )
        lp = logpdf.(dist, y)
        logl += sum(lp)
    end
    return -logl
end

# [Growth, Density, Recr, Diffusion, Variance]
#lower = [0.0, 0, 0, 0, 0]
# lower = [-Inf, -Inf, -Inf, -Inf, -Inf]
# upper = [Inf, Inf, Inf, Inf, Inf]
p0 = [1.1, 100, .1, 1, 2] # initial parameter values

res = optimize(loglike, p0)
est = res.minimizer
# print estimated parameters
println(est)

# plot the original data and simulated trajectory
#  define the analytical solution for BH model
bht(t, p) = p[2]*p[3] ./ (p[3] .+ (p[2] - p[3]) * (p[1] .^-t) )
# set initial conditions and time array
n0 = 1
t = [1.0:1:50;]
# compute the carrying capacity
K = (est[1] - 1) * est[2]
# combine estimated parameters
phat = [est[1], K, n0]
# simulate and plot predicted cover
yhat = bht(t, phat) # + rand(Normal(0, est[3]), length(t))
# plot(dfn', legend=:none)
plot!(t, yhat, linewidth=4, line=:dash, legend=:none)

# ====================== simulate output
ngen = size(img)[3]
hmat = zeros(n, n, ngen)
smat = zeros(n, n, ngen)

K2D(x, y) =  K2DG(x, y)
growth(N, a, b) = bvholt(N, a, b)

sigma = est[5]
r = est[1]; M = est[2]; alpha = est[3]; K = (est[1]-1) * est[2];
D = est[4]

sker = inflate(K2D, xf, yf)
Fsker = fft(sker)

hmat[:,:,1] = img[:,:,1]
smat[:,:,1] = img[:,:,1]
	
htf = img[:,:,2] .+ 1
stf = img[:,:,2]

for j = 2:ngen
	global htf, stf
	hn = bh(htf, [r, M]) .+ alpha .* (1.0 .- htf/K ) .* stf
	sn = hn
		
	fsn = fft(sn)
	eta = rand(truncated(Normal(0, sigma), 0 , Inf), 32^2)
	
    htf = hn + reshape(eta, n, n)
	stf = real( fftshift( ifft(Fsker .* fsn) ) )

	hmat[:,:,j] = htf
	smat[:,:,j] = stf
end

plot(hmat[:,:,10], st = :surface, 
 xlabel = "x", ylabel = "y", zlabel = "Population size, H_t", camera=(0,90) )