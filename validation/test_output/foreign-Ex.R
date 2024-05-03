pkgname <- "foreign"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('foreign')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("lookup.xport")
### * lookup.xport

flush(stderr()); flush(stdout())

### Name: lookup.xport
### Title: Lookup Information on a SAS XPORT Format Library
### Aliases: lookup.xport
### Keywords: file

### ** Examples

## Not run: 
##D ## no XPORT file is installed.
##D lookup.xport("test.xpt")
## End(Not run)



cleanEx()
nameEx("read.S")
### * read.S

flush(stderr()); flush(stdout())

### Name: S3 read functions
### Title: Read an S3 Binary or data.dump File
### Aliases: data.restore read.S
### Keywords: data file

### ** Examples
## if you have an S-PLUS _Data file containing 'myobj'
## Not run: 
##D read.S(file.path("_Data", "myobj"))
##D data.restore("dumpdata", print = TRUE)
## End(Not run)


cleanEx()
nameEx("read.dbf")
### * read.dbf

flush(stderr()); flush(stdout())

### Name: read.dbf
### Title: Read a DBF File
### Aliases: read.dbf
### Keywords: file

### ** Examples

x <- read.dbf(system.file("files/sids.dbf", package="foreign")[1])
str(x)
summary(x)



cleanEx()
nameEx("read.dta")
### * read.dta

flush(stderr()); flush(stdout())

### Name: read.dta
### Title: Read Stata Binary Files
### Aliases: read.dta
### Keywords: file

### ** Examples

write.dta(swiss,swissfile <- tempfile())
read.dta(swissfile)



cleanEx()
nameEx("read.epiinfo")
### * read.epiinfo

flush(stderr()); flush(stdout())

### Name: read.epiinfo
### Title: Read Epi Info Data Files
### Aliases: read.epiinfo
### Keywords: file

### ** Examples

## Not run: 
##D ## That file is not available
##D read.epiinfo("oswego.rec", guess.broken.dates = TRUE, thisyear = "1972")
## End(Not run)



cleanEx()
nameEx("read.mtp")
### * read.mtp

flush(stderr()); flush(stdout())

### Name: read.mtp
### Title: Read a Minitab Portable Worksheet
### Aliases: read.mtp
### Keywords: file

### ** Examples

## Not run: 
##D read.mtp("ex1-10.mtp")
## End(Not run)



cleanEx()
nameEx("read.spss")
### * read.spss

flush(stderr()); flush(stdout())

### Name: read.spss
### Title: Read an SPSS Data File
### Aliases: read.spss
### Keywords: file

### ** Examples

(sav <- system.file("files", "electric.sav", package = "foreign"))
dat <- read.spss(file=sav) 
str(dat)   # list structure with attributes

dat <- read.spss(file=sav, to.data.frame=TRUE) 
str(dat)   # now a data.frame


### Now we use an example file that is not very well structured and 
### hence may need some special treatment with appropriate argument settings.
### Expect lots of warnings as value labels (corresponding to R factor labels) are uncomplete, 
### and an unsupported long string variable is present in the data
(sav <- system.file("files", "testdata.sav", package = "foreign"))

### Examples for add.undeclared.levels:
## add.undeclared.levels = "sort" (default):
x.sort <- read.spss(file=sav, to.data.frame = TRUE)
## add.undeclared.levels = "append":
x.append <- read.spss(file=sav, to.data.frame = TRUE, 
    add.undeclared.levels = "append")
## add.undeclared.levels = "no":
x.no <- read.spss(file=sav, to.data.frame = TRUE, 
    add.undeclared.levels = "no")

levels(x.sort$factor_n_undeclared)
levels(x.append$factor_n_undeclared)
str(x.no$factor_n_undeclared)


### Examples for duplicated.value.labels:
## duplicated.value.labels = "append" (default)
x.append <- read.spss(file=sav, to.data.frame=TRUE)
## duplicated.value.labels = "condense"
x.condense <- read.spss(file=sav, to.data.frame=TRUE, 
    duplicated.value.labels = "condense")

levels(x.append$factor_n_duplicated)
levels(x.condense$factor_n_duplicated)

as.numeric(x.append$factor_n_duplicated)
as.numeric(x.condense$factor_n_duplicated)

    
## Long Strings (>255 chars) are imported in consecutive separate variables 
## (see warning about subtype 14):
x <- read.spss(file=sav, to.data.frame=TRUE, stringsAsFactors=FALSE)

cat.long.string <- function(x, w=70) cat(paste(strwrap(x, width=w), "\n"))

## first part: x$string_500:
cat.long.string(x$string_500)
## second part: x$STRIN0:
cat.long.string(x$STRIN0)
## complete long string:
long.string <- apply(x[,c("string_500", "STRIN0")], 1, paste, collapse="")
cat.long.string(long.string)



cleanEx()
nameEx("read.ssd")
### * read.ssd

flush(stderr()); flush(stdout())

### Name: read.ssd
### Title: Obtain a Data Frame from a SAS Permanent Dataset, via read.xport
### Aliases: read.ssd
### Keywords: file

### ** Examples

## if there were some files on the web we could get a real
## runnable example
## Not run: 
##D R> list.files("trialdata")
##D  [1] "baseline.sas7bdat" "form11.sas7bdat"   "form12.sas7bdat"  
##D  [4] "form13.sas7bdat"   "form22.sas7bdat"   "form23.sas7bdat"  
##D  [7] "form3.sas7bdat"    "form4.sas7bdat"    "form48.sas7bdat"  
##D [10] "form50.sas7bdat"   "form51.sas7bdat"   "form71.sas7bdat"  
##D [13] "form72.sas7bdat"   "form8.sas7bdat"    "form9.sas7bdat"   
##D [16] "form90.sas7bdat"   "form91.sas7bdat"  
##D R> baseline <- read.ssd("trialdata", "baseline")
##D R> form90 <- read.ssd("trialdata", "form90")
##D 
##D ## Or for a Windows example
##D sashome <- "/Program Files/SAS/SAS 9.1"
##D read.ssd(file.path(sashome, "core", "sashelp"), "retail",
##D          sascmd = file.path(sashome, "sas.exe"))
## End(Not run)



cleanEx()
nameEx("read.systat")
### * read.systat

flush(stderr()); flush(stdout())

### Name: read.systat
### Title: Obtain a Data Frame from a Systat File
### Aliases: read.systat
### Keywords: file

### ** Examples

summary(iris)
iris.s <- read.systat(system.file("files/Iris.syd", package="foreign")[1])
str(iris.s)
summary(iris.s)



cleanEx()
nameEx("read.xport")
### * read.xport

flush(stderr()); flush(stdout())

### Name: read.xport
### Title: Read a SAS XPORT Format Library
### Aliases: read.xport
### Keywords: file

### ** Examples

## Not run: 
##D ## no XPORT file is installed
##D read.xport("test.xpt")
## End(Not run)



cleanEx()
nameEx("write.arff")
### * write.arff

flush(stderr()); flush(stdout())

### Name: write.arff
### Title: Write Data into ARFF Files
### Aliases: write.arff
### Keywords: print file

### ** Examples

write.arff(iris, file = "")



cleanEx()
nameEx("write.dbf")
### * write.dbf

flush(stderr()); flush(stdout())

### Name: write.dbf
### Title: Write a DBF File
### Aliases: write.dbf
### Keywords: file

### ** Examples

str(warpbreaks)
try1 <- paste(tempfile(), ".dbf", sep = "")
write.dbf(warpbreaks, try1, factor2char = FALSE)
in1 <- read.dbf(try1)
str(in1)
try2 <- paste(tempfile(), ".dbf", sep = "")
write.dbf(warpbreaks, try2, factor2char = TRUE)
in2 <- read.dbf(try2)
str(in2)
unlink(c(try1, try2))
## Don't show: 
DF <- data.frame(a=c(1:3, NA), b=c(NA, rep(pi, 3)),
                 c=c(TRUE,NA, FALSE, TRUE), d=c("aa", "bb", NA, "dd"),
                 e=I(c("a1", NA, NA, "a4")))
DF$f <- as.Date(c("2001-01-01", NA, NA, "2004-10-26"))
str(DF)
write.dbf(DF, try2)
in2 <- read.dbf(try2)
str(in2)
unlink(try2)
## End(Don't show)


cleanEx()
nameEx("write.dta")
### * write.dta

flush(stderr()); flush(stdout())

### Name: write.dta
### Title: Write Files in Stata Binary Format
### Aliases: write.dta
### Keywords: file

### ** Examples

write.dta(swiss, swissfile <- tempfile())
read.dta(swissfile)



cleanEx()
nameEx("write.foreign")
### * write.foreign

flush(stderr()); flush(stdout())

### Name: write.foreign
### Title: Write Text Files and Code to Read Them
### Aliases: write.foreign
### Keywords: file

### ** Examples
## Not run: 
##D datafile <- tempfile()
##D codefile <- tempfile()
##D write.foreign(esoph, datafile, codefile, package="SPSS")
##D file.show(datafile)
##D file.show(codefile)
##D unlink(datafile)
##D unlink(codefile)
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
