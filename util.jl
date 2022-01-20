# ---------------------------------------------------------------
 # create pdf
  using Weave, DSP, Plots
  weave(joinpath(pwd(), "model_description.jmd"),
    out_path = joinpath(pwd(), "model_description_pdf"),
    doctype = "md2html") 

# ---------------------------------------------------------------
 # create a gif animation 
 # requires a series of Plots
 anim = @animate for i in 1:100#size(hmat)[3]
  plot(hmat[:,:,i], st = :surface, zlims = [0, 25],
    xlabel = "x", ylabel = "y",  zlabel = "Population size, H_t", 
    title = string("t = ", i, "/", size(hmat)[3]))
end every 1
gif(anim, "anim0_fps10_0.gif", fps = 12)