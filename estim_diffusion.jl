# This script simulates a simple diffusion process and develops a stat model to recover the diffusion parameter

# === load packages
using Plots, FFTW, LinearAlgebra, SparseArrays, FFTW


# 1. Simulate data 

# === helper function
include("helper_fns.jl") 

# === IDE code
n = 32;  x = 10; dx = 2*x/n; 
# define the spatial arrays in x and y
xf = [range(-x,  x-dx , length = n);] 
yf = [range(-x, x-dx , length = n);] 
	
# number of generation
ngen = 30
	
# diffusion coefficient
D = 1
	
# combine spatial arrays into grid
XF = xf' .* ones(n)
YF = ones(n)' .* yf
	
# xy = getxy.(XF, YF)

# store simulations
hmat = zeros(n, n, ngen + 1)

# set up initial conditions	

h0 = (abs.(XF) .<= 2) .* (abs.(YF) .<= 2) .* 100
plot(h0, st = :surface, camera=(20,50))

# === choose a dispersal kernel: K2DL (Laplace), K2DG (Gaussian)
K2D(x, y) =  K2DG(x, y)

# === simulate 
sker = inflate(K2D, xf, yf)
Fsker = fft(sker)

hmat[:,:,1] = h0

for j = 2:ngen
	fhn = fft(hmat[:,:,j-1])
	
	hmat[:,:,j] = dx .* real( fftshift( ifft(Fsker .* fhn) ) )
end

# idx, idy = Int.(n/2-n/4:n/2+n/4), Int.(n/2-n/4:n/2+n/4)
#hcrop = hmat[idx, idy, :]

anim = @animate for i in 1:30
    plot(hmat[:,:,i], st = :surface,
      xlabel = "x", ylabel = "y",  zlabel = "Population size, H_t")
end every 1
gif(anim, "anim0_out.gif", fps = 2)

# === output
ydat = hmat


# 2. Fit the simulated model to the data
using Distributions, Optim, LinearAlgebra, Plots, StatsPlots
using AdvancedHMC, Zygote
inf(f, xs, ys, D) = [f(x,y, D) for x in xs, y in ys]

xy = getxy.(xf' .* ones(n), ones(n)' .* yf)
K2Ds(x, y, D) =  1/(2pi*D) * exp(-norm( (x, y) , 1)^2/(2D))
K2Dv1(xy, D) = 1/(2pi*D) * exp(-norm( xy , 1)^2/(2D))


# @btime a = real(fft(inf(K2Ds, xf, yf, D)))
# @btime b = real(fft(K2Dv1.(xy, D)))

ydatm = ydat + reshape(randn(length(ydat)), 32, 32, 31)
fydatm = [fft(ydatm[:,:,j]) for j in 1:size(ydatm)[3]]
# @btime real(fftshift( ifft(fft(ydatm[:,:,1])) ));
# === constrict a likelihood function
function loglike(rho)
        ngen = size(ydatm)[3]
        # decode the parameters
        D = rho[1]
        sigma2 = rho[2]
        # initialize storage output
        logl = 0
        # Dispersal kernel and FFT
        Fsker = fft(K2Dv1.(xy, D))
        # run time series loop
        
        mu = [dx .* real(fftshift(ifft(Fsker .* fydatm[j]))) for j in 1:ngen]
        # likelihood
        dist = [Normal.(mu[j], sqrt(2)) for j in 1:ngen]
        logl += sum([sum(logpdf.(dist[j], y[j])) for j in 1:ngen])
        return -logl
end



P = 2; initial_params = rand(P)

n_samples, n_adapts = 100, 80

target(x) =  loglike(x) + sum(logpdf.(Truncated(Normal(0,1), 0, Inf), x))

metric = DiagEuclideanMetric(P)
hamiltonian = Hamiltonian(metric, target, Zygote)

initial_ϵ = find_good_stepsize(hamiltonian, initial_params)
integrator = Leapfrog(initial_ϵ)
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

samples1, stats1 = sample(hamiltonian, proposal, initial_params, 
                        n_samples, adaptor, n_adapts; progress=true);
samples2, stats2 = sample(hamiltonian, proposal, initial_params, 
                        n_samples, adaptor, n_adapts; progress=true);

a11 = map(x -> x[1], samples1)
a12 = map(x -> x[1], samples2)
a21 = map(x -> x[2], samples1)
a22 = map(x -> x[2], samples2)
                        
bayesEst = map( x -> mean(x[1:end]), [a11, a21])
bayesLower = map( x -> quantile(x[1:end], 0.25), [a11, a21])
bayesUpper = map( x -> quantile(x[1:end], 0.75), [a11, a21])

density(a11, label="Chain 1")
density(a21, label="Chain 2")
plot!(-4:4, pdf.(Normal(0, 5), -4:4), label="Prior")

plot(a11, label="Chain 1")
plot!(a21, label="Chain 2")
# === end
