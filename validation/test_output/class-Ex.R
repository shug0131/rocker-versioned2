pkgname <- "class"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('class')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("SOM")
### * SOM

flush(stderr()); flush(stdout())

### Name: SOM
### Title: Self-Organizing Maps: Online Algorithm
### Aliases: SOM
### Keywords: classif

### ** Examples

require(graphics)
data(crabs, package = "MASS")

lcrabs <- log(crabs[, 4:8])
crabs.grp <- factor(c("B", "b", "O", "o")[rep(1:4, rep(50,4))])
gr <- somgrid(topo = "hexagonal")
crabs.som <- SOM(lcrabs, gr)
plot(crabs.som)

## 2-phase training
crabs.som2 <- SOM(lcrabs, gr,
    alpha = list(seq(0.05, 0, length.out = 1e4), seq(0.02, 0, length.out = 1e5)),
    radii = list(seq(8, 1, length.out = 1e4), seq(4, 1, length.out = 1e5)))
plot(crabs.som2)



cleanEx()
nameEx("batchSOM")
### * batchSOM

flush(stderr()); flush(stdout())

### Name: batchSOM
### Title: Self-Organizing Maps: Batch Algorithm
### Aliases: batchSOM
### Keywords: classif

### ** Examples

require(graphics)
data(crabs, package = "MASS")

lcrabs <- log(crabs[, 4:8])
crabs.grp <- factor(c("B", "b", "O", "o")[rep(1:4, rep(50,4))])
gr <- somgrid(topo = "hexagonal")
crabs.som <- batchSOM(lcrabs, gr, c(4, 4, 2, 2, 1, 1, 1, 0, 0))
plot(crabs.som)

bins <- as.numeric(knn1(crabs.som$codes, lcrabs, 0:47))
plot(crabs.som$grid, type = "n")
symbols(crabs.som$grid$pts[, 1], crabs.som$grid$pts[, 2],
        circles = rep(0.4, 48), inches = FALSE, add = TRUE)
text(crabs.som$grid$pts[bins, ] + rnorm(400, 0, 0.1),
     as.character(crabs.grp))



cleanEx()
nameEx("condense")
### * condense

flush(stderr()); flush(stdout())

### Name: condense
### Title: Condense training set for k-NN classifier
### Aliases: condense
### Keywords: classif

### ** Examples

train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
keep <- condense(train, cl)
knn(train[keep, , drop=FALSE], test, cl[keep])
keep2 <- reduce.nn(train, keep, cl)
knn(train[keep2, , drop=FALSE], test, cl[keep2])



cleanEx()
nameEx("knn")
### * knn

flush(stderr()); flush(stdout())

### Name: knn
### Title: k-Nearest Neighbour Classification
### Aliases: knn
### Keywords: classif

### ** Examples

train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
knn(train, test, cl, k = 3, prob=TRUE)
attributes(.Last.value)



cleanEx()
nameEx("knn.cv")
### * knn.cv

flush(stderr()); flush(stdout())

### Name: knn.cv
### Title: k-Nearest Neighbour Cross-Validatory Classification
### Aliases: knn.cv
### Keywords: classif

### ** Examples

train <- rbind(iris3[,,1], iris3[,,2], iris3[,,3])
cl <- factor(c(rep("s",50), rep("c",50), rep("v",50)))
knn.cv(train, cl, k = 3, prob = TRUE)
attributes(.Last.value)



cleanEx()
nameEx("knn1")
### * knn1

flush(stderr()); flush(stdout())

### Name: knn1
### Title: 1-Nearest Neighbour Classification
### Aliases: knn1
### Keywords: classif

### ** Examples

train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
knn1(train, test, cl)



cleanEx()
nameEx("lvq1")
### * lvq1

flush(stderr()); flush(stdout())

### Name: lvq1
### Title: Learning Vector Quantization 1
### Aliases: lvq1
### Keywords: classif

### ** Examples

train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
cd <- lvqinit(train, cl, 10)
lvqtest(cd, train)
cd0 <- olvq1(train, cl, cd)
lvqtest(cd0, train)
cd1 <- lvq1(train, cl, cd0)
lvqtest(cd1, train)



cleanEx()
nameEx("lvq2")
### * lvq2

flush(stderr()); flush(stdout())

### Name: lvq2
### Title: Learning Vector Quantization 2.1
### Aliases: lvq2
### Keywords: classif

### ** Examples

train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
cd <- lvqinit(train, cl, 10)
lvqtest(cd, train)
cd0 <- olvq1(train, cl, cd)
lvqtest(cd0, train)
cd2 <- lvq2(train, cl, cd0)
lvqtest(cd2, train)



cleanEx()
nameEx("lvq3")
### * lvq3

flush(stderr()); flush(stdout())

### Name: lvq3
### Title: Learning Vector Quantization 3
### Aliases: lvq3
### Keywords: classif

### ** Examples

train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
cd <- lvqinit(train, cl, 10)
lvqtest(cd, train)
cd0 <- olvq1(train, cl, cd)
lvqtest(cd0, train)
cd3 <- lvq3(train, cl, cd0)
lvqtest(cd3, train)



cleanEx()
nameEx("lvqinit")
### * lvqinit

flush(stderr()); flush(stdout())

### Name: lvqinit
### Title: Initialize a LVQ Codebook
### Aliases: lvqinit
### Keywords: classif

### ** Examples

train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
cd <- lvqinit(train, cl, 10)
lvqtest(cd, train)
cd1 <- olvq1(train, cl, cd)
lvqtest(cd1, train)



cleanEx()
nameEx("lvqtest")
### * lvqtest

flush(stderr()); flush(stdout())

### Name: lvqtest
### Title: Classify Test Set from LVQ Codebook
### Aliases: lvqtest
### Keywords: classif

### ** Examples

# The function is currently defined as
function(codebk, test) knn1(codebk$x, test, codebk$cl)



cleanEx()
nameEx("multiedit")
### * multiedit

flush(stderr()); flush(stdout())

### Name: multiedit
### Title: Multiedit for k-NN Classifier
### Aliases: multiedit
### Keywords: classif

### ** Examples

tr <- sample(1:50, 25)
train <- rbind(iris3[tr,,1], iris3[tr,,2], iris3[tr,,3])
test <- rbind(iris3[-tr,,1], iris3[-tr,,2], iris3[-tr,,3])
cl <- factor(c(rep(1,25),rep(2,25), rep(3,25)), labels=c("s", "c", "v"))
table(cl, knn(train, test, cl, 3))
ind1 <- multiedit(train, cl, 3)
length(ind1)
table(cl, knn(train[ind1, , drop=FALSE], test, cl[ind1], 1))
ntrain <- train[ind1,]; ncl <- cl[ind1]
ind2 <- condense(ntrain, ncl)
length(ind2)
table(cl, knn(ntrain[ind2, , drop=FALSE], test, ncl[ind2], 1))



cleanEx()
nameEx("olvq1")
### * olvq1

flush(stderr()); flush(stdout())

### Name: olvq1
### Title: Optimized Learning Vector Quantization 1
### Aliases: olvq1
### Keywords: classif

### ** Examples

train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
cd <- lvqinit(train, cl, 10)
lvqtest(cd, train)
cd1 <- olvq1(train, cl, cd)
lvqtest(cd1, train)



cleanEx()
nameEx("reduce.nn")
### * reduce.nn

flush(stderr()); flush(stdout())

### Name: reduce.nn
### Title: Reduce Training Set for a k-NN Classifier
### Aliases: reduce.nn
### Keywords: classif

### ** Examples

train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
keep <- condense(train, cl)
knn(train[keep,], test, cl[keep])
keep2 <- reduce.nn(train, keep, cl)
knn(train[keep2,], test, cl[keep2])



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
