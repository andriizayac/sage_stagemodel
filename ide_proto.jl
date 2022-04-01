# === load packages
using Plots, FFTW, LinearAlgebra, SparseArrays, FFTW

# === helper function
include("helper_fns.jl") 

# === IDE code
n = 1024;  x = 100; dx = 2*x/n; 
# define the spatial arrays in x and y
xf = [range(-x,  x-dx , length = n);] #.+ 10
yf = [range(-x, x-dx , length = n);] #.+ 10
	
# number of generation
ngen = 100
	
# diffusion coefficient
D = 5
	
# combine spatial arrays into grid
XF = xf' .* ones(n)
YF = ones(n)' .* yf
	
xy = getxy.(XF, YF)

sigma = ones(n)
alpha = .01ones(n)
rf = 0.175ones(n)
kf = 20ones(n)
	
# store simulations
hmat = zeros(n, n, ngen + 1)
smat = zeros(n, n, ngen + 1)

# set up initial conditions	
locIC=512
q=zeros(n)
q[locIC-4]=1
q[locIC-3]=1
q[locIC-2]=1
q[locIC-1]=1
q[locIC]=1
w=zeros(n-1)
w[locIC-5]=1
w[locIC-4]=1
w[locIC-3]=1
w[locIC-2]=1
w[locIC-1]=1
h0 = Matrix(spdiagm(1 => w, 0 =>  q, -1 => w)) # (abs.(XF) .>= 9) #.* (abs.(YF) .<= 1)

s0 = zeros(n, n) # (abs.(XF) .>= 9) #.* (abs.(YF) .>= 9) # #  # + rand(npf, npf)
	

# === choose a dispersal kernel: K2DL (Laplace), K2DG (Gaussian)
K2D(x, y) =  K2DG(x, y)

# === choose growth functions
# Beverton-Holt - Contest [bvholt]
# Ricker - Scramble [ricker]
# Logistic - Scramble [logistic]
growth(N, a, b) = bvholt(N, a, b)


# === simulate 
sker = inflate(K2D, xf, yf)

#surface(sker, camera=(20, 80))
Fsker = fft(sker)
#surface(real.(Fsker), camera=(20, 80))


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
title = "t = 0", 
camera=(0,90))

#add plot statement for 512x512
tt=50
p3 = plot(hmat[256:768,256:768,tt], st = :surface, 
xlabel = "x", ylabel = "y", 
zlabel = "Population size, H_t", 
title = tt,
camera=(0,90)) 


p2 = plot(hmat[:,:,tt], st = :surface, 
xlabel = "x", ylabel = "y", 
zlabel = "Population size, H_t", 
title = tt,  
camera=(0,90))

plot(p1, p2, layout = l)

#set up to store initial and final sparse Matrices
#and incorport the setseed command for repeated simulations