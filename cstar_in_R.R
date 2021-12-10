# demographic matrix
sigma1 = 1.1
gamma = .01
sigma2 = .5
phi = 10.5
B = matrix(ncol=2,nrow=2,c(sigma1*(1-gamma), phi, sigma1*gamma, sigma2))

# dispersal kernel
D = 100
# K = [1 1 ; 1/2D.*exp.(-abs.(x)/D) 1]

# moment generating function based on K 
M_s<-function(s){
  matrix(ncol=2,nrow=2,c(1,1/1-D^2*s^2),1)
}

H_s<-function(s){
  M_s(s)*B
}

# asymptotic wave speed

library("popbio")

c_s<-function(s){1/s*log(lambda(H_s(s)))}

# define the range of s
sq = seq(from=1, to=200, by = .01)
cs= c_s(sq)

plot(s, cs, xlab = "Wave Shape Parameter, s", ylab = "Wave Speed, c")

cstar = fzero(c', 2, 3)