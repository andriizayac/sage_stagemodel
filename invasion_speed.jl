using LinearAlgebra, Plots

# demographic matrix
sigma1 = 1.1
gamma = .01
sigma2 = .5
phi = 10.5
B = [ sigma1*(1-gamma) phi ; sigma1*gamma sigma2 ]

# dispersal kernel
D = 100
# K = [1 1 ; 1/2D.*exp.(-abs.(x)/D) 1]

# moment generating function based on K 
M(s) = [1 1 ; 1/(1 - D^2 * s^2) 1]

H(s) = M(s) .* B

# asymptotic wave speed
c(s) = 1/s * log( eigen(H(s)).values[end] )

# define the range of s
s = range(1, 200, step = .01)
cs= c.(s)

plot(s, cs, xlabel = "Wave Shape Parameter, s", ylabel = "Wave Speed, c", label = "")

cstar = fzero(c', 2, 3)