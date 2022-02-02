# === load packages
using Plots, FFTW

# === helper function
inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]

# === IDE code
n = 128;  x = 10; dx = 2*x/n; 
# define the spatial arrays in x and y
xf = [range(-x, x-dx, length = n);]
yf = [range(-x, x-dx, length = n);]
	
# number of generation
ngen = 100
	
# diffusion coefficient
D = .02
	
# combine spatial arrays into grid
XF = xf' .* ones(n)
YF = ones(n)' .* yf
	
sigma = ones(n)
alpha = .01ones(n)
rf = 0.175ones(n)
kf = 20ones(n)
	
# store simulations
hmat = zeros(n, n, ngen + 1)
smat = zeros(n, n, ngen + 1)
	
# set up initial conditions
h0 = (abs.(XF) .>= 9) #.* (abs.(YF) .<= 1)
s0 = (abs.(XF) .>= 9) #.* (abs.(YF) .>= 9) # #  # + rand(npf, npf)
	
# define/choose movement kernels
# Laplace kernel
K2DL(x, y) = 1/sqrt(2*D^2)*exp(-sqrt(2/D^2)*abs(x - y))
# Gaussian kernel
K2DG(x, y) = 1/sqrt(2pi*D^2)*exp(-(x - y)^2/(2D^2))
# Powell kernel (tutorial)
K2DP(x, y) = 1/(4pi*D) * exp(-(x^2 + y^2) / (4D))
# define the kernel
K2D(x,y) =  K2DP(x,y)

# define/choose growth functions
# Beverton-Holt - Contest
bvholt(N, a, b) = a .* N ./ (1 .+ b .* N)
# Ricker - Scramble
ricker(N, a, b) = N .* exp.(a .* (1 .- N ./ b))
# Logistic - Scramble
logistic(N, a, b) = a .* N .* (1 .- N ./ b)

growth(N, a, b) = bvholt(N, a, b)
# === simulate 
sker = inflate(K2D, xf, yf)
	
Fsker = fft(sker)
	
hmat[:,:,1] = h0
smat[:,:,1] = s0
	
htf = h0
stf = s0
for j = 1:ngen
	global htf, stf
	hn = htf .+ growth(htf, .1,  20) .+ alpha .*(1 .- htf/20) .* stf
	sn = sigma .* hn
		
	fsn = fft(sn)
	
	htf = hn
	stf = real( fftshift( ifft(Fsker .* fsn) ) )

	hmat[:,:,j+1] = htf
	smat[:,:,j+1] = stf
end

l = @layout[a; b]
p1 = plot(hmat[:,:,1], st = :surface, 
xlabel = "x", ylabel = "y", 
zlabel = "Population size, H_t", 
title = "t = 0")

tt = 10
p2 = plot(hmat[:,:,tt], st = :surface, 
xlabel = "x", ylabel = "y", 
zlabel = "Population size, H_t", 
title = tt)

plot(p1, p2, layout = l)