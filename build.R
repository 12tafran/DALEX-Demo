rmarkdown::render("presentation.Rmd",
                  output_file = rprojroot::find_rstudio_root_file("dist/index.html"))
fs::dir_delete("dist/img")
rmarkdown::render_supporting_files("static/img", "dist")
