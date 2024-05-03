pkgname <- "codetools"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('codetools')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("checkUsage")
### * checkUsage

flush(stderr()); flush(stdout())

### Name: checkUsage
### Title: Check R Code for Possible Problems
### Aliases: checkUsage checkUsageEnv checkUsagePackage
### Keywords: programming

### ** Examples

checkUsage(checkUsage)
checkUsagePackage("codetools",all=TRUE)
## Not run: checkUsagePackage("base",suppressLocal=TRUE)



cleanEx()
nameEx("findGlobals")
### * findGlobals

flush(stderr()); flush(stdout())

### Name: findGlobals
### Title: Find Global Functions and Variables Used by a Closure
### Aliases: findGlobals
### Keywords: programming

### ** Examples

findGlobals(findGlobals)
findGlobals(findGlobals, merge = FALSE)



cleanEx()
nameEx("showTree")
### * showTree

flush(stderr()); flush(stdout())

### Name: showTree
### Title: Print Lisp-Style Representation of R Expression
### Aliases: showTree
### Keywords: programming

### ** Examples

showTree(quote(-3))
showTree(quote("x"<-1))
showTree(quote("f"(x)))



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
