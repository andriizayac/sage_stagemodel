# ==== inflate grid from x, y
inflate(f, xs, ys) = [f(x,y) for x in xs, y in ys]
getxy(x,y) = (x,y) # coordinate xy tuples


# === kernels
# 1-D
# Laplace kernel
K1DL(x, y) = 1/sqrt(2*D)*exp(-sqrt(2/D)*abs(x - y))
# Gaussian kernel
K1DG(x, y) = 1/sqrt(2pi*D^2)*exp(-(x - y)^2/(2D^2))


# 2-D
# Laplace kernel
# K2DL(x) = 1/(2*D) * exp(-sqrt(2/D)*norm(x, 1))
K2DL(x, y) = 1/(2*D) * exp(-sqrt(2/D)*norm( (x, y) , 1))
# Gaussian kernel
K2DG(x, y) = 1/(2pi*D^2) * exp(-norm( (x, y) , 1)^2/(2D^2))
# Powell kernel (tutorial)
K2DP(x, y) = 1/(4pi*D) * exp(-(x^2 + y^2) / (4D))

# 2-D kernels with tuple input
# Laplace kernel
# K2DL(x) = 1/(2*D) * exp(-sqrt(2/D)*norm(x, 1))
K2DLs(xy, D) = 1/(2*D) * exp(-sqrt(2/D)*norm( xy , 1))
# Gaussian kernel
K2DGs(xy, D) = 1/(2pi*D^2) * exp(-norm( xy, 1)^2/(2D^2))
# Powell kernel (tutorial)
K2Ds(xy, D) = 1/(4pi*D) * exp(-norm(xy, 1)^2/(2D))


# === Growth functions
# Beverton-Holt - Contest
bvholt(N, a, b) = a .* N ./ (1 .+ b .* N)
# Ricker - Scramble
ricker(N, a, b) = N .* exp.(a .* (1 .- N ./ b))
# Logistic - Scramble
logistic(N, a, b) = a .* N .* (1 .- N ./ b)