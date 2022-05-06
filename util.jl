# ---------------------------------------------------------------
 # create pdf
  using Weave, DSP, Plots
  weave(joinpath(pwd(), "model_description.jmd"),
    out_path = joinpath(pwd(), "model_description_pdf"),
    doctype = "md2html") 

# ---------------------------------------------------------------
 # create a gif animation 
 # requires a series of Plots
anim = @animate for i in 1:10#size(hmat)[3]
  plot(img[:,:,i], st = :surface, zlims = [0, 20],
    xlabel = "x", ylabel = "y",  zlabel = "Population size, H_t")
end every 1
gif(anim, "anim0_out.gif", fps = 2)

# === example of saving the output
using JLD
a = rand(10,10)
b = rand(1000,100, 5)
JLD.save("mydata.jld", "a", a, "b", b)

ain = JLD.load("mydata.jld", "a")
bin = JLD.load("mydata.jld", "b")

h = JLD.load("mydata.jld","b") 
