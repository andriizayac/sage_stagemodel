# === load packages
using Plots, FFTW

# === helper function
inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]

# === IDE code
npf = 128;  xlf = 10; dxf = 2*xlf/npf; 
# define the spatial arrays in x and y
xf = [range(-xlf, xlf-dxf, length = npf);]
yf = [range(-xlf, xlf-dxf, length = npf);]
	
# number of generation
ngensf = 100
	
# diffusion coefficient
D = .02
	
# combine spatial arrays into grid
XF = xf' .* ones(npf)
YF = ones(npf)' .* yf
	
sigma = ones(npf)
alpha = .01ones(npf)
rf = 0.175ones(npf)
kf = 20ones(npf)
	
# store simulations
hmat = zeros(npf, npf, ngensf + 1)
pmat = zeros(npf, npf, ngensf + 1)
	
# set up initial conditions
p0f = (abs.(XF) .>= 9) #.* (abs.(YF) .<= 1)
h0f = (abs.(XF) .>= 9) #.* (abs.(YF) .>= 9) # #  # + rand(npf, npf)
	
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
pker = inflate(K2D, xf, yf)
	
Fpker = fft(pker)
	
hmat[:,:,1] = h0f
pmat[:,:,1] = p0f
	
htf = h0f
ptf = p0f
for j = 1:ngensf
	global htf, ptf
	hn = htf .+ growth(htf, .1,  20) .+ alpha .*(1 .- htf/20) .* ptf
	pn = sigma .* hn
		
	fpn = fft(pn)
	
	htf = hn
	ptf = real( fftshift( ifft(Fpker .* fpn) ) )

	hmat[:,:,j+1] = htf
	pmat[:,:,j+1] = ptf
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