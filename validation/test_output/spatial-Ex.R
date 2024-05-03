pkgname <- "spatial"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('spatial')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("Kaver")
### * Kaver

flush(stderr()); flush(stdout())

### Name: Kaver
### Title: Average K-functions from Simulations
### Aliases: Kaver
### Keywords: spatial

### ** Examples

towns <- ppinit("towns.dat")
par(pty="s")
plot(Kfn(towns, 40), type="b")
plot(Kfn(towns, 10), type="b", xlab="distance", ylab="L(t)")
for(i in 1:10) lines(Kfn(Psim(69), 10))
lims <- Kenvl(10,100,Psim(69))
lines(lims$x,lims$lower, lty=2, col="green")
lines(lims$x,lims$upper, lty=2, col="green")
lines(Kaver(10,25,Strauss(69,0.5,3.5)),  col="red")



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("Kenvl")
### * Kenvl

flush(stderr()); flush(stdout())

### Name: Kenvl
### Title: Compute Envelope and Average of Simulations of K-fns
### Aliases: Kenvl
### Keywords: spatial

### ** Examples

towns <- ppinit("towns.dat")
par(pty="s")
plot(Kfn(towns, 40), type="b")
plot(Kfn(towns, 10), type="b", xlab="distance", ylab="L(t)")
for(i in 1:10) lines(Kfn(Psim(69), 10))
lims <- Kenvl(10,100,Psim(69))
lines(lims$x,lims$lower, lty=2, col="green")
lines(lims$x,lims$upper, lty=2, col="green")
lines(Kaver(10,25,Strauss(69,0.5,3.5)), col="red")



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("Kfn")
### * Kfn

flush(stderr()); flush(stdout())

### Name: Kfn
### Title: Compute K-fn of a Point Pattern
### Aliases: Kfn
### Keywords: spatial

### ** Examples

towns <- ppinit("towns.dat")
par(pty="s")
plot(Kfn(towns, 10), type="s", xlab="distance", ylab="L(t)")



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("Psim")
### * Psim

flush(stderr()); flush(stdout())

### Name: Psim
### Title: Simulate Binomial Spatial Point Process
### Aliases: Psim
### Keywords: spatial

### ** Examples

towns <- ppinit("towns.dat")
par(pty="s")
plot(Kfn(towns, 10), type="s", xlab="distance", ylab="L(t)")
for(i in 1:10) lines(Kfn(Psim(69), 10))



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("SSI")
### * SSI

flush(stderr()); flush(stdout())

### Name: SSI
### Title: Simulates Sequential Spatial Inhibition Point Process
### Aliases: SSI
### Keywords: spatial

### ** Examples

towns <- ppinit("towns.dat")
par(pty = "s")
plot(Kfn(towns, 10), type = "b", xlab = "distance", ylab = "L(t)")
lines(Kaver(10, 25, SSI(69, 1.2)))



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("Strauss")
### * Strauss

flush(stderr()); flush(stdout())

### Name: Strauss
### Title: Simulates Strauss Spatial Point Process
### Aliases: Strauss
### Keywords: spatial

### ** Examples

towns <- ppinit("towns.dat")
par(pty="s")
plot(Kfn(towns, 10), type="b", xlab="distance", ylab="L(t)")
lines(Kaver(10, 25, Strauss(69,0.5,3.5)))



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("anova.trls")
### * anova.trls

flush(stderr()); flush(stdout())

### Name: anova.trls
### Title: Anova tables for fitted trend surface objects
### Aliases: anova.trls anovalist.trls
### Keywords: spatial

### ** Examples

library(stats)
data(topo, package="MASS")
topo0 <- surf.ls(0, topo)
topo1 <- surf.ls(1, topo)
topo2 <- surf.ls(2, topo)
topo3 <- surf.ls(3, topo)
topo4 <- surf.ls(4, topo)
anova(topo0, topo1, topo2, topo3, topo4)
summary(topo4)



cleanEx()
nameEx("correlogram")
### * correlogram

flush(stderr()); flush(stdout())

### Name: correlogram
### Title: Compute Spatial Correlograms
### Aliases: correlogram
### Keywords: spatial

### ** Examples

data(topo, package="MASS")
topo.kr <- surf.ls(2, topo)
correlogram(topo.kr, 25)
d <- seq(0, 7, 0.1)
lines(d, expcov(d, 0.7))



cleanEx()
nameEx("expcov")
### * expcov

flush(stderr()); flush(stdout())

### Name: expcov
### Title: Spatial Covariance Functions
### Aliases: expcov gaucov sphercov
### Keywords: spatial

### ** Examples

data(topo, package="MASS")
topo.kr <- surf.ls(2, topo)
correlogram(topo.kr, 25)
d <- seq(0, 7, 0.1)
lines(d, expcov(d, 0.7))



cleanEx()
nameEx("ppinit")
### * ppinit

flush(stderr()); flush(stdout())

### Name: ppinit
### Title: Read a Point Process Object from a File
### Aliases: ppinit
### Keywords: spatial

### ** Examples

towns <- ppinit("towns.dat")
par(pty="s")
plot(Kfn(towns, 10), type="b", xlab="distance", ylab="L(t)")



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("pplik")
### * pplik

flush(stderr()); flush(stdout())

### Name: pplik
### Title: Pseudo-likelihood Estimation of a Strauss Spatial Point Process
### Aliases: pplik
### Keywords: spatial

### ** Examples

pines <- ppinit("pines.dat")
pplik(pines, 0.7)



cleanEx()
nameEx("predict.trls")
### * predict.trls

flush(stderr()); flush(stdout())

### Name: predict.trls
### Title: Predict method for trend surface fits
### Aliases: predict.trls
### Keywords: spatial

### ** Examples

data(topo, package="MASS")
topo2 <- surf.ls(2, topo)
topo4 <- surf.ls(4, topo)
x <- c(1.78, 2.21)
y <- c(6.15, 6.15)
z2 <- predict(topo2, x, y)
z4 <- predict(topo4, x, y)
cat("2nd order predictions:", z2, "\n4th order predictions:", z4, "\n")



cleanEx()
nameEx("prmat")
### * prmat

flush(stderr()); flush(stdout())

### Name: prmat
### Title: Evaluate Kriging Surface over a Grid
### Aliases: prmat
### Keywords: spatial

### ** Examples

data(topo, package="MASS")
topo.kr <- surf.gls(2, expcov, topo, d=0.7)
prsurf <- prmat(topo.kr, 0, 6.5, 0, 6.5, 50)
contour(prsurf, levels=seq(700, 925, 25))



cleanEx()
nameEx("semat")
### * semat

flush(stderr()); flush(stdout())

### Name: semat
### Title: Evaluate Kriging Standard Error of Prediction over a Grid
### Aliases: semat
### Keywords: spatial

### ** Examples

data(topo, package="MASS")
topo.kr <- surf.gls(2, expcov, topo, d=0.7)
prsurf <- prmat(topo.kr, 0, 6.5, 0, 6.5, 50)
contour(prsurf, levels=seq(700, 925, 25))
sesurf <- semat(topo.kr, 0, 6.5, 0, 6.5, 30)
contour(sesurf, levels=c(22,25))



cleanEx()
nameEx("surf.gls")
### * surf.gls

flush(stderr()); flush(stdout())

### Name: surf.gls
### Title: Fits a Trend Surface by Generalized Least-squares
### Aliases: surf.gls
### Keywords: spatial

### ** Examples

library(MASS)  # for eqscplot
data(topo, package="MASS")
topo.kr <- surf.gls(2, expcov, topo, d=0.7)
trsurf <- trmat(topo.kr, 0, 6.5, 0, 6.5, 50)
eqscplot(trsurf, type = "n")
contour(trsurf, add = TRUE)

prsurf <- prmat(topo.kr, 0, 6.5, 0, 6.5, 50)
contour(prsurf, levels=seq(700, 925, 25))
sesurf <- semat(topo.kr, 0, 6.5, 0, 6.5, 30)
eqscplot(sesurf, type = "n")
contour(sesurf, levels = c(22, 25), add = TRUE)



cleanEx()
nameEx("surf.ls")
### * surf.ls

flush(stderr()); flush(stdout())

### Name: surf.ls
### Title: Fits a Trend Surface by Least-squares
### Aliases: surf.ls
### Keywords: spatial

### ** Examples

library(MASS)  # for eqscplot
data(topo, package="MASS")
topo.kr <- surf.ls(2, topo)
trsurf <- trmat(topo.kr, 0, 6.5, 0, 6.5, 50)
eqscplot(trsurf, type = "n")
contour(trsurf, add = TRUE)
points(topo)

eqscplot(trsurf, type = "n")
contour(trsurf, add = TRUE)
plot(topo.kr, add = TRUE)
title(xlab= "Circle radius proportional to Cook's influence statistic")



cleanEx()
nameEx("trls.influence")
### * trls.influence

flush(stderr()); flush(stdout())

### Name: trls.influence
### Title: Regression diagnostics for trend surfaces
### Aliases: trls.influence plot.trls
### Keywords: spatial

### ** Examples

library(MASS)  # for eqscplot
data(topo, package = "MASS")
topo2 <- surf.ls(2, topo)
infl.topo2 <- trls.influence(topo2)
(cand <- as.data.frame(infl.topo2)[abs(infl.topo2$stresid) > 1.5, ])
cand.xy <- topo[as.integer(rownames(cand)), c("x", "y")]
trsurf <- trmat(topo2, 0, 6.5, 0, 6.5, 50)
eqscplot(trsurf, type = "n")
contour(trsurf, add = TRUE, col = "grey")
plot(topo2, add = TRUE, div = 3)
points(cand.xy, pch = 16, col = "orange")
text(cand.xy, labels = rownames(cand.xy), pos = 4, offset = 0.5)



cleanEx()
nameEx("trmat")
### * trmat

flush(stderr()); flush(stdout())

### Name: trmat
### Title: Evaluate Trend Surface over a Grid
### Aliases: trmat
### Keywords: spatial

### ** Examples

data(topo, package="MASS")
topo.kr <- surf.ls(2, topo)
trsurf <- trmat(topo.kr, 0, 6.5, 0, 6.5, 50)



cleanEx()
nameEx("variogram")
### * variogram

flush(stderr()); flush(stdout())

### Name: variogram
### Title: Compute Spatial Variogram
### Aliases: variogram
### Keywords: spatial

### ** Examples

data(topo, package="MASS")
topo.kr <- surf.ls(2, topo)
variogram(topo.kr, 25)



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
