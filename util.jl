  # ---------------------------------------------------------------
  # create pdf
  using Weave, DSP, Plots
  weave(joinpath(pwd(), "model_description.jmd"),
    out_path = joinpath(pwd(), "model_description_pdf"),
    doctype = "md2pdf") 