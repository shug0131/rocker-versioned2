pkgname <- "nnet"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('nnet')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("class.ind")
### * class.ind

flush(stderr()); flush(stdout())

### Name: class.ind
### Title: Generates Class Indicator Matrix from a Factor
### Aliases: class.ind
### Keywords: neural utilities

### ** Examples

# The function is currently defined as
class.ind <- function(cl)
{
  n <- length(cl)
  cl <- as.factor(cl)
  x <- matrix(0, n, length(levels(cl)) )
  x[(1:n) + n*(unclass(cl)-1)] <- 1
  dimnames(x) <- list(names(cl), levels(cl))
  x
}



cleanEx()
nameEx("multinom")
### * multinom

flush(stderr()); flush(stdout())

### Name: multinom
### Title: Fit Multinomial Log-linear Models
### Aliases: multinom add1.multinom anova.multinom coef.multinom
###   drop1.multinom extractAIC.multinom predict.multinom print.multinom
###   summary.multinom print.summary.multinom vcov.multinom
###   model.frame.multinom logLik.multinom
### Keywords: neural models

### ** Examples

oc <- options(contrasts = c("contr.treatment", "contr.poly"))
library(MASS)
example(birthwt)
(bwt.mu <- multinom(low ~ ., bwt))
options(oc)



base::options(contrasts = c(unordered = "contr.treatment",ordered = "contr.poly"))
cleanEx()
nameEx("nnet.Hess")
### * nnet.Hess

flush(stderr()); flush(stdout())

### Name: nnetHess
### Title: Evaluates Hessian for a Neural Network
### Aliases: nnetHess
### Keywords: neural

### ** Examples

# use half the iris data
ir <- rbind(iris3[,,1], iris3[,,2], iris3[,,3])
targets <- matrix(c(rep(c(1,0,0),50), rep(c(0,1,0),50), rep(c(0,0,1),50)),
150, 3, byrow=TRUE)
samp <- c(sample(1:50,25), sample(51:100,25), sample(101:150,25))
ir1 <- nnet(ir[samp,], targets[samp,], size=2, rang=0.1, decay=5e-4, maxit=200)
eigen(nnetHess(ir1, ir[samp,], targets[samp,]), TRUE)$values



cleanEx()
nameEx("nnet")
### * nnet

flush(stderr()); flush(stdout())

### Name: nnet
### Title: Fit Neural Networks
### Aliases: nnet nnet.default nnet.formula add.net norm.net eval.nn
###   coef.nnet print.nnet summary.nnet print.summary.nnet
### Keywords: neural

### ** Examples

# use half the iris data
ir <- rbind(iris3[,,1],iris3[,,2],iris3[,,3])
targets <- class.ind( c(rep("s", 50), rep("c", 50), rep("v", 50)) )
samp <- c(sample(1:50,25), sample(51:100,25), sample(101:150,25))
ir1 <- nnet(ir[samp,], targets[samp,], size = 2, rang = 0.1,
            decay = 5e-4, maxit = 200)
test.cl <- function(true, pred) {
    true <- max.col(true)
    cres <- max.col(pred)
    table(true, cres)
}
test.cl(targets[-samp,], predict(ir1, ir[-samp,]))


# or
ird <- data.frame(rbind(iris3[,,1], iris3[,,2], iris3[,,3]),
        species = factor(c(rep("s",50), rep("c", 50), rep("v", 50))))
ir.nn2 <- nnet(species ~ ., data = ird, subset = samp, size = 2, rang = 0.1,
               decay = 5e-4, maxit = 200)
table(ird$species[-samp], predict(ir.nn2, ird[-samp,], type = "class"))



cleanEx()
nameEx("predict.nnet")
### * predict.nnet

flush(stderr()); flush(stdout())

### Name: predict.nnet
### Title: Predict New Examples by a Trained Neural Net
### Aliases: predict.nnet
### Keywords: neural

### ** Examples

# use half the iris data
ir <- rbind(iris3[,,1], iris3[,,2], iris3[,,3])
targets <- class.ind( c(rep("s", 50), rep("c", 50), rep("v", 50)) )
samp <- c(sample(1:50,25), sample(51:100,25), sample(101:150,25))
ir1 <- nnet(ir[samp,], targets[samp,],size = 2, rang = 0.1,
            decay = 5e-4, maxit = 200)
test.cl <- function(true, pred){
        true <- max.col(true)
        cres <- max.col(pred)
        table(true, cres)
}
test.cl(targets[-samp,], predict(ir1, ir[-samp,]))

# or
ird <- data.frame(rbind(iris3[,,1], iris3[,,2], iris3[,,3]),
        species = factor(c(rep("s",50), rep("c", 50), rep("v", 50))))
ir.nn2 <- nnet(species ~ ., data = ird, subset = samp, size = 2, rang = 0.1,
               decay = 5e-4, maxit = 200)
table(ird$species[-samp], predict(ir.nn2, ird[-samp,], type = "class"))



cleanEx()
nameEx("which.is.max")
### * which.is.max

flush(stderr()); flush(stdout())

### Name: which.is.max
### Title: Find Maximum Position in Vector
### Aliases: which.is.max
### Keywords: utilities

### ** Examples

## Not run: 
##D ## this is incomplete
##D pred <- predict(nnet, test)
##D table(true, apply(pred, 1, which.is.max))
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
