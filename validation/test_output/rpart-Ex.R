pkgname <- "rpart"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('rpart')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("car.test.frame")
### * car.test.frame

flush(stderr()); flush(stdout())

### Name: car.test.frame
### Title: Automobile Data from 'Consumer Reports' 1990
### Aliases: car.test.frame
### Keywords: datasets

### ** Examples

z.auto <- rpart(Mileage ~ Weight, car.test.frame)
summary(z.auto)



cleanEx()
nameEx("car90")
### * car90

flush(stderr()); flush(stdout())

### Name: car90
### Title: Automobile Data from 'Consumer Reports' 1990
### Aliases: car90
### Keywords: datasets

### ** Examples

data(car90)
plot(car90$Price/1000, car90$Weight,
     xlab = "Price (thousands)", ylab = "Weight (lbs)")
mlowess <- function(x, y, ...) {
    keep <- !(is.na(x) | is.na(y))
    lowess(x[keep], y[keep], ...)
}
with(car90, lines(mlowess(Price/1000, Weight, f = 0.5)))



cleanEx()
nameEx("cu.summary")
### * cu.summary

flush(stderr()); flush(stdout())

### Name: cu.summary
### Title: Automobile Data from 'Consumer Reports' 1990
### Aliases: cu.summary
### Keywords: datasets

### ** Examples

fit <- rpart(Price ~ Mileage + Type + Country, cu.summary)
par(xpd = TRUE)
plot(fit, compress = TRUE)
text(fit, use.n = TRUE)



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("kyphosis")
### * kyphosis

flush(stderr()); flush(stdout())

### Name: kyphosis
### Title: Data on Children who have had Corrective Spinal Surgery
### Aliases: kyphosis
### Keywords: datasets

### ** Examples

fit <- rpart(Kyphosis ~ Age + Number + Start, data = kyphosis)
fit2 <- rpart(Kyphosis ~ Age + Number + Start, data = kyphosis,
              parms = list(prior = c(0.65, 0.35), split = "information"))
fit3 <- rpart(Kyphosis ~ Age + Number + Start, data=kyphosis,
              control = rpart.control(cp = 0.05))
par(mfrow = c(1,2), xpd = TRUE)
plot(fit)
text(fit, use.n = TRUE)
plot(fit2)
text(fit2, use.n = TRUE)



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("meanvar.rpart")
### * meanvar.rpart

flush(stderr()); flush(stdout())

### Name: meanvar.rpart
### Title: Mean-Variance Plot for an Rpart Object
### Aliases: meanvar meanvar.rpart
### Keywords: tree

### ** Examples

z.auto <- rpart(Mileage ~ Weight, car.test.frame)
meanvar(z.auto, log = 'xy')



cleanEx()
nameEx("path.rpart")
### * path.rpart

flush(stderr()); flush(stdout())

### Name: path.rpart
### Title: Follow Paths to Selected Nodes of an Rpart Object
### Aliases: path.rpart
### Keywords: tree

### ** Examples

fit <- rpart(Kyphosis ~ Age + Number + Start, data = kyphosis)
print(fit)
path.rpart(fit, nodes = c(11, 22))



cleanEx()
nameEx("plot.rpart")
### * plot.rpart

flush(stderr()); flush(stdout())

### Name: plot.rpart
### Title: Plot an Rpart Object
### Aliases: plot.rpart
### Keywords: tree

### ** Examples

fit <- rpart(Price ~ Mileage + Type + Country, cu.summary)
par(xpd = TRUE)
plot(fit, compress = TRUE)
text(fit, use.n = TRUE)



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("post.rpart")
### * post.rpart

flush(stderr()); flush(stdout())

### Name: post.rpart
### Title: PostScript Presentation Plot of an Rpart Object
### Aliases: post.rpart post
### Keywords: tree

### ** Examples

## Not run: 
##D z.auto <- rpart(Mileage ~ Weight, car.test.frame)
##D post(z.auto, file = "")   # display tree on active device
##D    # now construct postscript version on file "pretty.ps"
##D    # with no title
##D post(z.auto, file = "pretty.ps", title = " ")
##D z.hp <- rpart(Mileage ~ Weight + HP, car.test.frame)
##D post(z.hp)
## End(Not run)



cleanEx()
nameEx("predict.rpart")
### * predict.rpart

flush(stderr()); flush(stdout())

### Name: predict.rpart
### Title: Predictions from a Fitted Rpart Object
### Aliases: predict.rpart
### Keywords: tree

### ** Examples

z.auto <- rpart(Mileage ~ Weight, car.test.frame)
predict(z.auto)

fit <- rpart(Kyphosis ~ Age + Number + Start, data = kyphosis)
predict(fit, type = "prob")   # class probabilities (default)
predict(fit, type = "vector") # level numbers
predict(fit, type = "class")  # factor
predict(fit, type = "matrix") # level number, class frequencies, probabilities

sub <- c(sample(1:50, 25), sample(51:100, 25), sample(101:150, 25))
fit <- rpart(Species ~ ., data = iris, subset = sub)
fit
table(predict(fit, iris[-sub,], type = "class"), iris[-sub, "Species"])



cleanEx()
nameEx("print.rpart")
### * print.rpart

flush(stderr()); flush(stdout())

### Name: print.rpart
### Title: Print an Rpart Object
### Aliases: print.rpart
### Keywords: tree

### ** Examples

z.auto <- rpart(Mileage ~ Weight, car.test.frame)
z.auto
## Not run: 
##D node), split, n, deviance, yval
##D       * denotes terminal node
##D 
##D  1) root 60 1354.58300 24.58333  
##D    2) Weight>=2567.5 45  361.20000 22.46667  
##D      4) Weight>=3087.5 22   61.31818 20.40909 *
##D      5) Weight<3087.5 23  117.65220 24.43478  
##D       10) Weight>=2747.5 15   60.40000 23.80000 *
##D       11) Weight<2747.5 8   39.87500 25.62500 *
##D    3) Weight<2567.5 15  186.93330 30.93333 *
## End(Not run)


cleanEx()
nameEx("printcp")
### * printcp

flush(stderr()); flush(stdout())

### Name: printcp
### Title: Displays CP table for Fitted Rpart Object
### Aliases: printcp
### Keywords: tree

### ** Examples

z.auto <- rpart(Mileage ~ Weight, car.test.frame)
printcp(z.auto)
## Not run: 
##D Regression tree:
##D rpart(formula = Mileage ~ Weight, data = car.test.frame)
##D 
##D Variables actually used in tree construction:
##D [1] Weight
##D 
##D Root node error: 1354.6/60 = 22.576
##D 
##D         CP nsplit rel error  xerror     xstd 
##D 1 0.595349      0   1.00000 1.03436 0.178526
##D 2 0.134528      1   0.40465 0.60508 0.105217
##D 3 0.012828      2   0.27012 0.45153 0.083330
##D 4 0.010000      3   0.25729 0.44826 0.076998
## End(Not run)


cleanEx()
nameEx("prune.rpart")
### * prune.rpart

flush(stderr()); flush(stdout())

### Name: prune.rpart
### Title: Cost-complexity Pruning of an Rpart Object
### Aliases: prune.rpart prune
### Keywords: tree

### ** Examples

z.auto <- rpart(Mileage ~ Weight, car.test.frame)
zp <- prune(z.auto, cp = 0.1)
plot(zp) #plot smaller rpart object



cleanEx()
nameEx("residuals.rpart")
### * residuals.rpart

flush(stderr()); flush(stdout())

### Name: residuals.rpart
### Title: Residuals From a Fitted Rpart Object
### Aliases: residuals.rpart
### Keywords: tree

### ** Examples

fit <- rpart(skips ~ Opening + Solder + Mask + PadType + Panel,
             data = solder.balance, method = "anova")
summary(residuals(fit))
plot(predict(fit),residuals(fit))



cleanEx()
nameEx("rpart")
### * rpart

flush(stderr()); flush(stdout())

### Name: rpart
### Title: Recursive Partitioning and Regression Trees
### Aliases: rpart
### Keywords: tree

### ** Examples

fit <- rpart(Kyphosis ~ Age + Number + Start, data = kyphosis)
fit2 <- rpart(Kyphosis ~ Age + Number + Start, data = kyphosis,
              parms = list(prior = c(.65,.35), split = "information"))
fit3 <- rpart(Kyphosis ~ Age + Number + Start, data = kyphosis,
              control = rpart.control(cp = 0.05))
par(mfrow = c(1,2), xpd = NA) # otherwise on some devices the text is clipped
plot(fit)
text(fit, use.n = TRUE)
plot(fit2)
text(fit2, use.n = TRUE)



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("rsq.rpart")
### * rsq.rpart

flush(stderr()); flush(stdout())

### Name: rsq.rpart
### Title: Plots the Approximate R-Square for the Different Splits
### Aliases: rsq.rpart
### Keywords: tree

### ** Examples

z.auto <- rpart(Mileage ~ Weight, car.test.frame)
rsq.rpart(z.auto)



cleanEx()
nameEx("snip.rpart")
### * snip.rpart

flush(stderr()); flush(stdout())

### Name: snip.rpart
### Title: Snip Subtrees of an Rpart Object
### Aliases: snip.rpart
### Keywords: tree

### ** Examples

## dataset not in R
## Not run: 
##D z.survey <- rpart(market.survey) # grow the rpart object
##D plot(z.survey) # plot the tree
##D z.survey2 <- snip.rpart(z.survey, toss = 2) # trim subtree at node 2
##D plot(z.survey2) # plot new tree
##D 
##D # can also interactively select the node using the mouse in the
##D # graphics window
## End(Not run)


cleanEx()
nameEx("solder.balance")
### * solder.balance

flush(stderr()); flush(stdout())

### Name: solder.balance
### Title: Soldering of Components on Printed-Circuit Boards
### Aliases: solder.balance solder
### Keywords: datasets

### ** Examples

fit <- rpart(skips ~ Opening + Solder + Mask + PadType + Panel,
             data = solder.balance, method = "anova")
summary(residuals(fit))
plot(predict(fit), residuals(fit))



cleanEx()
nameEx("stagec")
### * stagec

flush(stderr()); flush(stdout())

### Name: stagec
### Title: Stage C Prostate Cancer
### Aliases: stagec
### Keywords: datasets

### ** Examples

require(survival)
rpart(Surv(pgtime, pgstat) ~ ., stagec)



cleanEx()
nameEx("summary.rpart")
### * summary.rpart

flush(stderr()); flush(stdout())

### Name: summary.rpart
### Title: Summarize a Fitted Rpart Object
### Aliases: summary.rpart
### Keywords: tree

### ** Examples

## a regression tree
z.auto <- rpart(Mileage ~ Weight, car.test.frame)
summary(z.auto)

## a classification tree with multiple variables and surrogate splits.
summary(rpart(Kyphosis ~ Age + Number + Start, data = kyphosis))



cleanEx()
nameEx("text.rpart")
### * text.rpart

flush(stderr()); flush(stdout())

### Name: text.rpart
### Title: Place Text on a Dendrogram Plot
### Aliases: text.rpart
### Keywords: tree

### ** Examples

freen.tr <- rpart(y ~ ., freeny)
par(xpd = TRUE)
plot(freen.tr)
text(freen.tr, use.n = TRUE, all = TRUE)



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("xpred.rpart")
### * xpred.rpart

flush(stderr()); flush(stdout())

### Name: xpred.rpart
### Title: Return Cross-Validated Predictions
### Aliases: xpred.rpart
### Keywords: tree

### ** Examples

fit <- rpart(Mileage ~ Weight, car.test.frame)
xmat <- xpred.rpart(fit)
xerr <- (xmat - car.test.frame$Mileage)^2
apply(xerr, 2, sum)   # cross-validated error estimate

# approx same result as rel. error from printcp(fit)
apply(xerr, 2, sum)/var(car.test.frame$Mileage) 
printcp(fit)



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
