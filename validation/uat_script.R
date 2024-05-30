# Create details of R installation, Docker FIles and provenance
# Could create a bash script, or shell script to run the following

#docker  images  |  extract the image id
#docker inspect IMAGEID | grab the RepoDigest
#copy the website

# and then run R command line to process the mark up file. 
system(" docker images --digests| grep 'shug0131/cctu'")

Sys.info()
#.Platform
sessionInfo()

# Date stamps from the wiki for CRAN/Rstudio/R
# Explanation of tieing to last-but-one version
# LInk to the github and original Rocker project

#https://github.com/shug0131/rocker-versioned2
#https://rocker-project.org/
#https://github.com/rocker-org

# cd $R_HOME/tests
# sudo chmod a+rwx -R .
# ../bin/R CMD make check
#../bin/R CMD make check-devel
#../bin/R CMD make check-all
#../bin/R CMD make test-BasePackages
#../bin/R CMD make test-Recommended
# Don't actually need the ../bin  

#Ref to the manual where it says how to validate
# https://cran.r-project.org/doc/manuals/r-release/R-admin.html#Testing-a-Unix_002dalike-Installation
# Ref to the R position paper 
rm(list=ls())
Sys.setenv(LC_COLLATE = "C", LC_TIME = "C", LANGUAGE = "en")
basic_both <- tools::testInstalledBasic("both")
basic_internet <- tools::testInstalledBasic("internet")
# take a copy of the test outputs

system("echo $HOSTNAME")

# empty out the output directory first of all. 
Sys.setenv(TEST_MC_CORES=parallel::detectCores())
packages_both <- tools::testInstalledPackages(scope="recommended",
                                              outDir="validation/test_output")


#  Work through the extra packages added on.
pkg <- "cctu"
dir.create(file.path("validation","package_test_output",pkg))
tools::testInstalledPackage(pkg, outDir=file.path("validation","package_test_output",pkg))
# Need to have installed the package with R CMD INSTALL --install-tests to have the 
# test files . But these should be checked within CRAN - so could just pull them from internet. 
# Might need to tweak how cctu is installed, and any other non-public packages.
library(tidyverse)
library(rvest)

pks <- library()$results %>%
  as.data.frame %>%  
  filter( LibPath=="/usr/local/lib/R/site-library", Package!="cctu") %>% 
  pull(Package)

url <- paste0("https://www.r-project.org/nosvn/R.check/r-release-linux-x86_64/",pks,"-00check.html")
status <- rep(NA, length(url))
for( i in 1:length(url)){
  status[i] <- read_html(url[i]) %>% 
    html_element("p") %>% 
    html_text %>% 
    gsub("\\nStatus:","",.) %>% 
    gsub("\\n","",.)
}

link <- paste0("<a href=\"",url,"\">",pks,"</a>" )

knitr::kable(cbind(link, status ),
             format="html",
             escape=FALSE)
             

# Do the risk assessment tools

# Bundle into a layer to add on top of cctu docker image to automate the validation
# not sure how to localise it though. 