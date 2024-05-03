pkgname <- "eudract"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('eudract')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("clintrials_gov_convert")
### * clintrials_gov_convert

flush(stderr()); flush(stdout())

### Name: clintrials_gov_convert
### Title: applies a conversion using xslt from a simple xml file to a
###   ClinicalTrials.gov compatible file, and checks against the schema
### Aliases: clintrials_gov_convert

### ** Examples

safety_statistics <- safety_summary(safety,
                                    exposed=c("Experimental"=60,"Control"=67))
simple <- tempfile(fileext = ".xml")
eudract <- tempfile(fileext = ".xml")
ct <- tempfile(fileext = ".xml")
simple_safety_xml(safety_statistics, simple)
eudract_convert(input=simple,
                output=eudract)
clintrials_gov_convert(input=simple,
                       original=system.file("extdata", "1234.xml", package ="eudract"),
                output=ct)
## Not run: 
##D   # This needs a real user account to work
##D   clintrials_gov_upload(
##D     input=simple,
##D     orgname="CTU",
##D     username="Student",
##D     password="Guinness",
##D     studyid="1234"
##D     )
##D 
## End(Not run)



cleanEx()
nameEx("clintrials_gov_upload")
### * clintrials_gov_upload

flush(stderr()); flush(stdout())

### Name: clintrials_gov_upload
### Title: applies a conversion using xslt from a simple xml file to a
###   ClinicalTrials.gov compatible file, merges into a study record from
###   the portal, and uploads the result.
### Aliases: clintrials_gov_upload

### ** Examples

safety_statistics <- safety_summary(safety,
                                    exposed=c("Experimental"=60,"Control"=67))
simple <- tempfile(fileext = ".xml")
eudract <- tempfile(fileext = ".xml")
ct <- tempfile(fileext = ".xml")
simple_safety_xml(safety_statistics, simple)
eudract_convert(input=simple,
                output=eudract)
clintrials_gov_convert(input=simple,
                       original=system.file("extdata", "1234.xml", package ="eudract"),
                output=ct)
## Not run: 
##D   # This needs a real user account to work
##D   clintrials_gov_upload(
##D     input=simple,
##D     orgname="CTU",
##D     username="Student",
##D     password="Guinness",
##D     studyid="1234"
##D     )
##D 
## End(Not run)



cleanEx()
nameEx("dot_plot")
### * dot_plot

flush(stderr()); flush(stdout())

### Name: dot_plot
### Title: creates a dot-plot of safety data showing the absolute and
###   relative risks
### Aliases: dot_plot

### ** Examples

safety_statistics <- safety_summary(safety,
           exposed=c("Experimental"=60,"Control"=67))
head( relative_risk(safety_statistics, type="serious") )
fig <- dot_plot(safety_statistics, type="non_serious", base=4)
fig
fig$left.panel <- fig$left.panel + ggplot2::labs(title="Absolute Risk")
fig
temp <- tempfile(fileext=".png")
png(filename = temp)
print(fig)
dev.off()




cleanEx()
nameEx("eudract_convert")
### * eudract_convert

flush(stderr()); flush(stdout())

### Name: eudract_convert
### Title: applies a conversion using xslt from a simple xml file to a
###   eudract compatible file, and checks against the schema
### Aliases: eudract_convert

### ** Examples

safety_statistics <- safety_summary(safety,
                                    exposed=c("Experimental"=60,"Control"=67))
simple <- tempfile(fileext = ".xml")
eudract <- tempfile(fileext = ".xml")
ct <- tempfile(fileext = ".xml")
simple_safety_xml(safety_statistics, simple)
eudract_convert(input=simple,
                output=eudract)
clintrials_gov_convert(input=simple,
                       original=system.file("extdata", "1234.xml", package ="eudract"),
                output=ct)
## Not run: 
##D   # This needs a real user account to work
##D   clintrials_gov_upload(
##D     input=simple,
##D     orgname="CTU",
##D     username="Student",
##D     password="Guinness",
##D     studyid="1234"
##D     )
##D 
## End(Not run)



cleanEx()
nameEx("incidence_table")
### * incidence_table

flush(stderr()); flush(stdout())

### Name: incidence_table
### Title: provide standard structured tables to report incidence rates of
###   AEs by arm
### Aliases: incidence_table

### ** Examples

safety_statistics <- safety_summary(safety,
           exposed=c("Experimental"=60,"Control"=67))
head( incidence_table(safety_statistics, type="serious") )



cleanEx()
nameEx("relative_risk")
### * relative_risk

flush(stderr()); flush(stdout())

### Name: relative_risk
### Title: Calculate relative risks to be reported or plotted as dot plot
### Aliases: relative_risk relative_risk_table order_filter

### ** Examples

safety_statistics <- safety_summary(safety,
           exposed=c("Experimental"=60,"Control"=67))
head( relative_risk(safety_statistics, type="serious") )
relative_risk_table(safety_statistics, type="serious")
rr <- relative_risk(safety_statistics)
rr2 <- order_filter(rr, threshold=2)
dot_plot(rr2)



cleanEx()
nameEx("safety_summary")
### * safety_summary

flush(stderr()); flush(stdout())

### Name: safety_summary
### Title: Calculate frequency tables from a rectangular data frame with
###   one row per subject-event
### Aliases: safety_summary

### ** Examples

safety_statistics <- safety_summary(safety,
                                    exposed=c("Experimental"=60,"Control"=67))
simple <- tempfile(fileext = ".xml")
eudract <- tempfile(fileext = ".xml")
ct <- tempfile(fileext = ".xml")
simple_safety_xml(safety_statistics, simple)
eudract_convert(input=simple,
                output=eudract)
clintrials_gov_convert(input=simple,
                       original=system.file("extdata", "1234.xml", package ="eudract"),
                output=ct)
## Not run: 
##D   # This needs a real user account to work
##D   clintrials_gov_upload(
##D     input=simple,
##D     orgname="CTU",
##D     username="Student",
##D     password="Guinness",
##D     studyid="1234"
##D     )
##D 
## End(Not run)



cleanEx()
nameEx("simple_safety_xml")
### * simple_safety_xml

flush(stderr()); flush(stdout())

### Name: simple_safety_xml
### Title: creates a simple xml file from the input of a safety_summary
###   object
### Aliases: simple_safety_xml

### ** Examples

safety_statistics <- safety_summary(safety,
                                    exposed=c("Experimental"=60,"Control"=67))
simple <- tempfile(fileext = ".xml")
eudract <- tempfile(fileext = ".xml")
ct <- tempfile(fileext = ".xml")
simple_safety_xml(safety_statistics, simple)
eudract_convert(input=simple,
                output=eudract)
clintrials_gov_convert(input=simple,
                       original=system.file("extdata", "1234.xml", package ="eudract"),
                output=ct)
## Not run: 
##D   # This needs a real user account to work
##D   clintrials_gov_upload(
##D     input=simple,
##D     orgname="CTU",
##D     username="Student",
##D     password="Guinness",
##D     studyid="1234"
##D     )
##D 
## End(Not run)



### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
