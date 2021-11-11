# === load packages
using Plots, FFTW

# === helper function
inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]

# === IDE code
npf = 64; mup = .02; muh = .02; dt = .5; xlf = 10; dxf = 2*xlf/npf; 
xf = [range(-xlf, xlf-dxf, length = npf);]
yf = [range(-xlf, xlf-dxf, length = npf);]
	
ngensf = 50
	
D = 1
	
XF = xf' .* ones(npf)
YF = ones(npf)' .* yf
	
nf = ones(npf)
alphaf = .01ones(npf)
rf = 0.175ones(npf)
kf = 20ones(npf)
	
hmat = zeros(npf, npf, ngensf + 1)
pmat = zeros(npf, npf, ngensf + 1)
	
# set up initial conditions
p0f = (abs.(XF) .<= 1)
h0f = (abs.(XF) .<= 1) #+ rand(npf, npf)
	
# define movement kernel
K2D(x,y) = 1/(4pi*D) * exp(-(x^2 + y^2) / (4D))
pker = inflate(K2D, xf, yf)
	
Fpker = fft(pker)
	
hmat[:,:,1] = h0f
pmat[:,:,1] = p0f
	
htf = h0f
ptf = p0f
for j = 1:ngensf
	global htf, ptf
	hn = htf .+ htf .* rf .* (1 .- htf ./ kf) .+ alphaf .* ptf
	pn = nf .* hn
		
	fpn = fft(pn)
	
	htf = hn
	ptf = real( fftshift( ifft(Fpker .* fpn) ) )

	hmat[:,:,j+1] = htf
	pmat[:,:,j+1] = ptf
end

plot(hmat[:,:,10], st = :surface)