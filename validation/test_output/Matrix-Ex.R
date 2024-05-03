pkgname <- "Matrix"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('Matrix')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("BunchKaufman-class")
### * BunchKaufman-class

flush(stderr()); flush(stdout())

### Name: BunchKaufman-class
### Title: Dense Bunch-Kaufman Factorizations
### Aliases: BunchKaufman-class pBunchKaufman-class
###   coerce,BunchKaufman,dtrMatrix-method
###   coerce,pBunchKaufman,dtpMatrix-method
###   determinant,BunchKaufman,logical-method
###   determinant,pBunchKaufman,logical-method
### Keywords: algebra array classes

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("BunchKaufman")
set.seed(1)

n <- 6L
(A <- forceSymmetric(Matrix(rnorm(n * n), n, n)))

## With dimnames, to see that they are propagated :
dimnames(A) <- rep.int(list(paste0("x", seq_len(n))), 2L)

(bk.A <- BunchKaufman(A))
str(e.bk.A <- expand2(bk.A, complete = FALSE), max.level = 2L)
str(E.bk.A <- expand2(bk.A, complete =  TRUE), max.level = 2L)

## Underlying LAPACK representation
(m.bk.A <- as(bk.A, "dtrMatrix"))
stopifnot(identical(as(m.bk.A, "matrix"), `dim<-`(bk.A@x, bk.A@Dim)))

## Number of factors is 2*b+1, b <= n, which can be nontrivial ...
(b <- (length(E.bk.A) - 1L) %/% 2L)

ae1 <- function(a, b, ...) all.equal(as(a, "matrix"), as(b, "matrix"), ...)
ae2 <- function(a, b, ...) ae1(unname(a), unname(b), ...)

## A ~ U DU U', U := prod(Pk Uk) in floating point
stopifnot(exprs = {
    identical(names(e.bk.A), c("U", "DU", "U."))
    identical(e.bk.A[["U" ]], Reduce(`%*%`, E.bk.A[seq_len(b)]))
    identical(e.bk.A[["U."]], t(e.bk.A[["U"]]))
    ae1(A, with(e.bk.A, U %*% DU %*% U.))
})

## Factorization handled as factorized matrix
b <- rnorm(n)
stopifnot(identical(det(A), det(bk.A)),
          identical(solve(A, b), solve(bk.A, b)))



cleanEx()
nameEx("BunchKaufman-methods")
### * BunchKaufman-methods

flush(stderr()); flush(stdout())

### Name: BunchKaufman-methods
### Title: Methods for Bunch-Kaufman Factorization
### Aliases: BunchKaufman BunchKaufman-methods
###   BunchKaufman,dspMatrix-method BunchKaufman,dsyMatrix-method
###   BunchKaufman,matrix-method
### Keywords: algebra array methods

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showMethods("BunchKaufman", inherited = FALSE)
set.seed(0)

data(CAex, package = "Matrix")
class(CAex) # dgCMatrix
isSymmetric(CAex) # symmetric, but not formally

A <- as(CAex, "symmetricMatrix")
class(A) # dsCMatrix

## Have methods for denseMatrix (unpacked and packed),
## but not yet sparseMatrix ...
## Not run: 
##D (bk.A <- BunchKaufman(A))
## End(Not run)
(bk.A <- BunchKaufman(as(A, "unpackedMatrix")))

## A ~ U DU U' in floating point
str(e.bk.A <- expand2(bk.A), max.level = 2L)
stopifnot(all.equal(as(A, "matrix"), as(Reduce(`%*%`, e.bk.A), "matrix")))



cleanEx()
nameEx("CAex")
### * CAex

flush(stderr()); flush(stdout())

### Name: CAex
### Title: Albers' example Matrix with "Difficult" Eigen Factorization
### Aliases: CAex
### Keywords: datasets

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
data(CAex, package = "Matrix")
str(CAex) # of class "dgCMatrix"

image(CAex)# -> it's a simple band matrix with 5 bands
## and the eigen values are basically 1 (42 times) and 0 (30 x):
zapsmall(ev <- eigen(CAex, only.values=TRUE)$values)
## i.e., the matrix is symmetric, hence
sCA <- as(CAex, "symmetricMatrix")
## and
stopifnot(class(sCA) == "dsCMatrix",
          as(sCA, "matrix") == as(CAex, "matrix"))



cleanEx()
nameEx("CHMfactor-class")
### * CHMfactor-class

flush(stderr()); flush(stdout())

### Name: CHMfactor-class
### Title: Sparse Cholesky Factorizations
### Aliases: CHMfactor-class CHMsimpl-class CHMsuper-class dCHMsimpl-class
###   dCHMsuper-class nCHMsimpl-class nCHMsuper-class
###   coerce,CHMsimpl,dtCMatrix-method coerce,CHMsuper,dgCMatrix-method
###   determinant,CHMfactor,logical-method diag,CHMfactor-method
###   update,CHMfactor-method isLDL
### Keywords: algebra array classes programming utilities

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("dCHMsimpl")
showClass("dCHMsuper")
set.seed(2)

m <- 1000L
n <- 200L
M <- rsparsematrix(m, n, 0.01)
A <- crossprod(M)

## With dimnames, to see that they are propagated :
dimnames(A) <- dn <- rep.int(list(paste0("x", seq_len(n))), 2L)

(ch.A <- Cholesky(A)) # pivoted, by default
str(e.ch.A <- expand2(ch.A, LDL =  TRUE), max.level = 2L)
str(E.ch.A <- expand2(ch.A, LDL = FALSE), max.level = 2L)

ae1 <- function(a, b, ...) all.equal(as(a, "matrix"), as(b, "matrix"), ...)
ae2 <- function(a, b, ...) ae1(unname(a), unname(b), ...)

## A ~ P1' L1 D L1' P1 ~ P1' L L' P1 in floating point
stopifnot(exprs = {
    identical(names(e.ch.A), c("P1.", "L1", "D", "L1.", "P1"))
    identical(names(E.ch.A), c("P1.", "L" ,      "L." , "P1"))
    identical(e.ch.A[["P1"]],
              new("pMatrix", Dim = c(n, n), Dimnames = c(list(NULL), dn[2L]),
                  margin = 2L, perm = invertPerm(ch.A@perm, 0L, 1L)))
    identical(e.ch.A[["P1."]], t(e.ch.A[["P1"]]))
    identical(e.ch.A[["L1."]], t(e.ch.A[["L1"]]))
    identical(E.ch.A[["L." ]], t(E.ch.A[["L" ]]))
    identical(e.ch.A[["D"]], Diagonal(x = diag(ch.A)))
    all.equal(E.ch.A[["L"]], with(e.ch.A, L1 %*% sqrt(D)))
    ae1(A, with(e.ch.A, P1. %*% L1 %*% D %*% L1. %*% P1))
    ae1(A, with(E.ch.A, P1. %*% L  %*%         L.  %*% P1))
    ae2(A[ch.A@perm + 1L, ch.A@perm + 1L], with(e.ch.A, L1 %*% D %*% L1.))
    ae2(A[ch.A@perm + 1L, ch.A@perm + 1L], with(E.ch.A, L  %*%         L. ))
})

## Factorization handled as factorized matrix
## (in some cases only optionally, depending on arguments)
b <- rnorm(n)
stopifnot(identical(det(A), det(ch.A, sqrt = FALSE)),
          identical(solve(A, b), solve(ch.A, b, system = "A")))

u1 <- update(ch.A,   A , mult = sqrt(2))
u2 <- update(ch.A, t(M), mult = sqrt(2)) # updating with crossprod(M), not M
stopifnot(all.equal(u1, u2, tolerance = 1e-14))



cleanEx()
nameEx("Cholesky-class")
### * Cholesky-class

flush(stderr()); flush(stdout())

### Name: Cholesky-class
### Title: Dense Cholesky Factorizations
### Aliases: Cholesky-class pCholesky-class
###   coerce,Cholesky,dtrMatrix-method coerce,pCholesky,dtpMatrix-method
###   determinant,Cholesky,logical-method
###   determinant,pCholesky,logical-method diag,Cholesky-method
###   diag,pCholesky-method
### Keywords: algebra array classes

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("Cholesky")
set.seed(1)

m <- 30L
n <- 6L
(A <- crossprod(Matrix(rnorm(m * n), m, n)))

## With dimnames, to see that they are propagated :
dimnames(A) <- dn <- rep.int(list(paste0("x", seq_len(n))), 2L)

(ch.A <- Cholesky(A)) # pivoted, by default
str(e.ch.A <- expand2(ch.A, LDL =  TRUE), max.level = 2L)
str(E.ch.A <- expand2(ch.A, LDL = FALSE), max.level = 2L)

## Underlying LAPACK representation
(m.ch.A <- as(ch.A, "dtrMatrix")) # which is L', not L, because
A@uplo == "U"
stopifnot(identical(as(m.ch.A, "matrix"), `dim<-`(ch.A@x, ch.A@Dim)))

ae1 <- function(a, b, ...) all.equal(as(a, "matrix"), as(b, "matrix"), ...)
ae2 <- function(a, b, ...) ae1(unname(a), unname(b), ...)

## A ~ P1' L1 D L1' P1 ~ P1' L L' P1 in floating point
stopifnot(exprs = {
    identical(names(e.ch.A), c("P1.", "L1", "D", "L1.", "P1"))
    identical(names(E.ch.A), c("P1.", "L" ,      "L." , "P1"))
    identical(e.ch.A[["P1"]],
              new("pMatrix", Dim = c(n, n), Dimnames = c(list(NULL), dn[2L]),
                  margin = 2L, perm = invertPerm(ch.A@perm)))
    identical(e.ch.A[["P1."]], t(e.ch.A[["P1"]]))
    identical(e.ch.A[["L1."]], t(e.ch.A[["L1"]]))
    identical(E.ch.A[["L." ]], t(E.ch.A[["L" ]]))
    identical(e.ch.A[["D"]], Diagonal(x = diag(ch.A)))
    all.equal(E.ch.A[["L"]], with(e.ch.A, L1 %*% sqrt(D)))
    ae1(A, with(e.ch.A, P1. %*% L1 %*% D %*% L1. %*% P1))
    ae1(A, with(E.ch.A, P1. %*% L  %*%         L.  %*% P1))
    ae2(A[ch.A@perm, ch.A@perm], with(e.ch.A, L1 %*% D %*% L1.))
    ae2(A[ch.A@perm, ch.A@perm], with(E.ch.A, L  %*%         L. ))
})

## Factorization handled as factorized matrix
b <- rnorm(n)
all.equal(det(A), det(ch.A), tolerance = 0)
all.equal(solve(A, b), solve(ch.A, b), tolerance = 0)

## For identical results, we need the _unpivoted_ factorization
## computed by det(A) and solve(A, b)
(ch.A.nopivot <- Cholesky(A, perm = FALSE))
stopifnot(identical(det(A), det(ch.A.nopivot)),
          identical(solve(A, b), solve(ch.A.nopivot, b)))



cleanEx()
nameEx("Cholesky")
### * Cholesky

flush(stderr()); flush(stdout())

### Name: Cholesky-methods
### Title: Methods for Cholesky Factorization
### Aliases: Cholesky Cholesky-methods Cholesky,ddiMatrix-method
###   Cholesky,diagonalMatrix-method Cholesky,dsCMatrix-method
###   Cholesky,dsRMatrix-method Cholesky,dsTMatrix-method
###   Cholesky,dspMatrix-method Cholesky,dsyMatrix-method
###   Cholesky,generalMatrix-method Cholesky,matrix-method
###   Cholesky,symmetricMatrix-method Cholesky,triangularMatrix-method
### Keywords: algebra array methods

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showMethods("Cholesky", inherited = FALSE)
set.seed(0)

## ---- Dense ----------------------------------------------------------

## .... Positive definite ..............................................

n <- 6L
(A1 <- crossprod(Matrix(rnorm(n * n), n, n)))
(ch.A1.nopivot <- Cholesky(A1, perm = FALSE))
(ch.A1 <- Cholesky(A1))
stopifnot(exprs = {
    length(ch.A1@perm) == ncol(A1)
    isPerm(ch.A1@perm)
    is.unsorted(ch.A1@perm) # typically not the identity permutation
    length(ch.A1.nopivot@perm) == 0L
})

## A ~ P1' L D L' P1 ~ P1' L L' P1 in floating point
str(e.ch.A1 <- expand2(ch.A1, LDL =  TRUE), max.level = 2L)
str(E.ch.A1 <- expand2(ch.A1, LDL = FALSE), max.level = 2L)
stopifnot(exprs = {
    all.equal(as(A1, "matrix"), as(Reduce(`%*%`, e.ch.A1), "matrix"))
    all.equal(as(A1, "matrix"), as(Reduce(`%*%`, E.ch.A1), "matrix"))
})

## .... Positive semidefinite but not positive definite ................

A2 <- A1
A2[1L, ] <- A2[, 1L] <- 0
A2
try(Cholesky(A2, perm = FALSE)) # fails as not positive definite
ch.A2 <- Cholesky(A2) # returns, with a warning and ...
A2.hat <- Reduce(`%*%`, expand2(ch.A2, LDL = FALSE))
norm(A2 - A2.hat, "2") / norm(A2, "2") # 7.670858e-17

## .... Not positive semidefinite ......................................

A3 <- A1
A3[1L, ] <- A3[, 1L] <- -1
A3
try(Cholesky(A3, perm = FALSE)) # fails as not positive definite
ch.A3 <- Cholesky(A3) # returns, with a warning and ...
A3.hat <- Reduce(`%*%`, expand2(ch.A3, LDL = FALSE))
norm(A3 - A3.hat, "2") / norm(A3, "2") # 1.781568

## Indeed, 'A3' is not positive semidefinite, but 'A3.hat' _is_
ch.A3.hat <- Cholesky(A3.hat)
A3.hat.hat <- Reduce(`%*%`, expand2(ch.A3.hat, LDL = FALSE))
norm(A3.hat - A3.hat.hat, "2") / norm(A3.hat, "2") # 1.777944e-16

## ---- Sparse ---------------------------------------------------------

## Really just three cases modulo permutation :
##
##            type        factorization  minors of P1 A P1'
##   1  simplicial  P1 A P1' = L1 D L1'             nonzero
##   2  simplicial  P1 A P1' = L    L '            positive
##   3  supernodal  P1 A P2' = L    L '            positive

data(KNex, package = "Matrix")
A4 <- crossprod(KNex[["mm"]])

ch.A4 <-
list(pivoted =
     list(simpl1 = Cholesky(A4, perm =  TRUE, super = FALSE, LDL =  TRUE),
          simpl0 = Cholesky(A4, perm =  TRUE, super = FALSE, LDL = FALSE),
          super0 = Cholesky(A4, perm =  TRUE, super =  TRUE             )),
     unpivoted =
     list(simpl1 = Cholesky(A4, perm = FALSE, super = FALSE, LDL =  TRUE),
          simpl0 = Cholesky(A4, perm = FALSE, super = FALSE, LDL = FALSE),
          super0 = Cholesky(A4, perm = FALSE, super =  TRUE             )))
ch.A4

s <- simplify2array
rapply2 <- function(object, f, ...) rapply(object, f, , , how = "list", ...)

s(rapply2(ch.A4, isLDL))
s(m.ch.A4 <- rapply2(ch.A4, expand1, "L")) # giving L = L1 sqrt(D)

## By design, the pivoted and simplicial factorizations
## are more sparse than the unpivoted and supernodal ones ...
s(rapply2(m.ch.A4, object.size))

## Which is nicely visualized by lattice-based methods for 'image'
inm <- c("pivoted", "unpivoted")
jnm <- c("simpl1", "simpl0", "super0")
for(i in 1:2)
  for(j in 1:3)
    print(image(m.ch.A4[[c(i, j)]], main = paste(inm[i], jnm[j])),
          split = c(j, i, 3L, 2L), more = i * j < 6L)

simpl1 <- ch.A4[[c("pivoted", "simpl1")]]
stopifnot(exprs = {
    length(simpl1@perm) == ncol(A4)
    isPerm(simpl1@perm, 0L)
    is.unsorted(simpl1@perm) # typically not the identity permutation
})

## One can expand with and without D regardless of isLDL(.),
## but "without" requires L = L1 sqrt(D), which is conditional
## on min(diag(D)) >= 0, hence "with" is the default
isLDL(simpl1)
stopifnot(min(diag(simpl1)) >= 0)
str(e.ch.A4 <- expand2(simpl1, LDL =  TRUE), max.level = 2L) # default
str(E.ch.A4 <- expand2(simpl1, LDL = FALSE), max.level = 2L)
stopifnot(exprs = {
    all.equal(E.ch.A4[["L" ]], e.ch.A4[["L1" ]] %*% sqrt(e.ch.A4[["D"]]))
    all.equal(E.ch.A4[["L."]], sqrt(e.ch.A4[["D"]]) %*% e.ch.A4[["L1."]])
    all.equal(A4, as(Reduce(`%*%`, e.ch.A4), "symmetricMatrix"))
    all.equal(A4, as(Reduce(`%*%`, E.ch.A4), "symmetricMatrix"))
})

## The "same" permutation matrix with "alternate" representation
## [i, perm[i]] {margin=1} <-> [invertPerm(perm)[j], j] {margin=2}
alt <- function(P) {
    P@margin <- 1L + !(P@margin - 1L) # 1 <-> 2
    P@perm <- invertPerm(P@perm)
    P
}

## Expansions are elegant but inefficient (transposes are redundant)
## hence programmers should consider methods for 'expand1' and 'diag'
stopifnot(exprs = {
    identical(expand1(simpl1, "P1"), alt(e.ch.A4[["P1"]]))
    identical(expand1(simpl1, "L"), E.ch.A4[["L"]])
    identical(Diagonal(x = diag(simpl1)), e.ch.A4[["D"]])
})

## chol(A, pivot = value) is a simple wrapper around
## Cholesky(A, perm = value, LDL = FALSE, super = FALSE),
## returning L' = sqrt(D) L1' _but_ giving no information
## about the permutation P1
selectMethod("chol", "dsCMatrix")
stopifnot(all.equal(chol(A4, pivot = TRUE), E.ch.A4[["L."]]))

## Now a symmetric matrix with positive _and_ negative eigenvalues,
## hence _not_ positive semidefinite
A5 <- new("dsCMatrix",
          Dim = c(7L, 7L),
          p = c(0:1, 3L, 6:7, 10:11, 15L),
          i = c(0L, 0:1, 0:3, 2:5, 3:6),
          x = c(1, 6, 38, 10, 60, 103, -4, 6, -32, -247, -2, -16, -128, -2, -67))
(ev <- eigen(A5, only.values = TRUE)$values)
(t.ev <- table(factor(sign(ev), -1:1))) # the matrix "inertia"

ch.A5 <- Cholesky(A5)
isLDL(ch.A5)
(d.A5 <- diag(ch.A5)) # diag(D) is partly negative

## Sylvester's law of inertia holds here, but not in general
## in finite precision arithmetic
stopifnot(identical(table(factor(sign(d.A5), -1:1)), t.ev))

try(expand1(ch.A5, "L"))         # unable to compute L = L1 sqrt(D)
try(expand2(ch.A5, LDL = FALSE)) # ditto
try(chol(A5, pivot = TRUE))      # ditto

## The default expansion is "square root free" and still works here
str(e.ch.A5 <- expand2(ch.A5, LDL = TRUE), max.level = 2L)
stopifnot(all.equal(A5, as(Reduce(`%*%`, e.ch.A5), "symmetricMatrix")))

## Version of the SuiteSparse library, which includes CHOLMOD
Mv <- Matrix.Version()
Mv[["SuiteSparse"]]



cleanEx()
nameEx("CsparseMatrix-class")
### * CsparseMatrix-class

flush(stderr()); flush(stdout())

### Name: CsparseMatrix-class
### Title: Class "CsparseMatrix" of Sparse Matrices in Column-compressed
###   Form
### Aliases: CsparseMatrix-class Arith,CsparseMatrix,CsparseMatrix-method
###   Arith,CsparseMatrix,numeric-method Arith,numeric,CsparseMatrix-method
###   Compare,CsparseMatrix,CsparseMatrix-method
###   Logic,CsparseMatrix,CsparseMatrix-method
###   coerce,matrix,CsparseMatrix-method coerce,vector,CsparseMatrix-method
###   diag,CsparseMatrix-method diag<-,CsparseMatrix-method
###   t,CsparseMatrix-method .validateCsparse
### Keywords: array classes

### ** Examples

getClass("CsparseMatrix")

## The common validity check function (based on C code):
getValidity(getClass("CsparseMatrix"))



cleanEx()
nameEx("Diagonal")
### * Diagonal

flush(stderr()); flush(stdout())

### Name: Diagonal
### Title: Construct a Diagonal Matrix
### Aliases: Diagonal .sparseDiagonal .trDiagonal .symDiagonal
### Keywords: array utilities

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
Diagonal(3)
Diagonal(x = 10^(3:1))
Diagonal(x = (1:4) >= 2)#-> "ldiMatrix"

## Use Diagonal() + kronecker() for "repeated-block" matrices:
M1 <- Matrix(0+0:5, 2,3)
(M <- kronecker(Diagonal(3), M1))

(S <- crossprod(Matrix(rbinom(60, size=1, prob=0.1), 10,6)))
(SI <- S + 10*.symDiagonal(6)) # sparse symmetric still
stopifnot(is(SI, "dsCMatrix"))
(I4 <- .sparseDiagonal(4, shape="t"))# now (2012-10) unitriangular
stopifnot(I4@diag == "U", all(I4 == diag(4)))
## Don't show: 
  L <- Diagonal(5, TRUE)
  stopifnot(L@diag == "U", identical(L, Diagonal(5) > 0))
## End(Don't show)



cleanEx()
nameEx("Hilbert")
### * Hilbert

flush(stderr()); flush(stdout())

### Name: Hilbert
### Title: Generate a Hilbert matrix
### Aliases: Hilbert
### Keywords: array utilities

### ** Examples

Hilbert(6)



cleanEx()
nameEx("KNex")
### * KNex

flush(stderr()); flush(stdout())

### Name: KNex
### Title: Koenker-Ng Example Sparse Model Matrix and Response Vector
### Aliases: KNex
### Keywords: datasets

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
data(KNex, package = "Matrix")
class(KNex$mm)
dim(KNex$mm)
image(KNex$mm)
str(KNex)

system.time( # a fraction of a second
  sparse.sol <- with(KNex, solve(crossprod(mm), crossprod(mm, y))))

head(round(sparse.sol,3))

## Compare with QR-based solution ("more accurate, but slightly slower"):
system.time(
  sp.sol2 <- with(KNex, qr.coef(qr(mm), y) ))

all.equal(sparse.sol, sp.sol2, tolerance = 1e-13) # TRUE



cleanEx()
nameEx("KhatriRao")
### * KhatriRao

flush(stderr()); flush(stdout())

### Name: KhatriRao
### Title: Khatri-Rao Matrix Product
### Aliases: KhatriRao
### Keywords: algebra arith array utilities

### ** Examples

## Example with very small matrices:
m <- matrix(1:12,3,4)
d <- diag(1:4)
KhatriRao(m,d)
KhatriRao(d,m)
dimnames(m) <- list(LETTERS[1:3], letters[1:4])
KhatriRao(m,d, make.dimnames=TRUE)
KhatriRao(d,m, make.dimnames=TRUE)
dimnames(d) <- list(NULL, paste0("D", 1:4))
KhatriRao(m,d, make.dimnames=TRUE)
KhatriRao(d,m, make.dimnames=TRUE)
dimnames(d) <- list(paste0("d", 10*1:4), paste0("D", 1:4))
(Kmd <- KhatriRao(m,d, make.dimnames=TRUE))
(Kdm <- KhatriRao(d,m, make.dimnames=TRUE))

nm <- as(m, "nsparseMatrix")
nd <- as(d, "nsparseMatrix")
KhatriRao(nm,nd, make.dimnames=TRUE)
KhatriRao(nd,nm, make.dimnames=TRUE)

stopifnot(dim(KhatriRao(m,d)) == c(nrow(m)*nrow(d), ncol(d)))
## border cases / checks:
zm <- nm; zm[] <- FALSE # all FALSE matrix
stopifnot(all(K1 <- KhatriRao(nd, zm) == 0), identical(dim(K1), c(12L, 4L)),
          all(K2 <- KhatriRao(zm, nd) == 0), identical(dim(K2), c(12L, 4L)))

d0 <- d; d0[] <- 0; m0 <- Matrix(d0[-1,])
stopifnot(all(K3 <- KhatriRao(d0, m) == 0), identical(dim(K3), dim(Kdm)),
	  all(K4 <- KhatriRao(m, d0) == 0), identical(dim(K4), dim(Kmd)),
	  all(KhatriRao(d0, d0) == 0), all(KhatriRao(m0, d0) == 0),
	  all(KhatriRao(d0, m0) == 0), all(KhatriRao(m0, m0) == 0),
	  identical(dimnames(KhatriRao(m, d0, make.dimnames=TRUE)), dimnames(Kmd)))

## a matrix with "structural" and non-structural zeros:
m01 <- new("dgCMatrix", i = c(0L, 2L, 0L, 1L), p = c(0L, 0L, 0L, 2L, 4L),
           Dim = 3:4, x = c(1, 0, 1, 0))
D4 <- Diagonal(4, x=1:4) # "as" d
DU <- Diagonal(4)# unit-diagonal: uplo="U"
(K5  <- KhatriRao( d, m01))
K5d  <- KhatriRao( d, m01, sparseY=FALSE)
K5Dd <- KhatriRao(D4, m01, sparseY=FALSE)
K5Ud <- KhatriRao(DU, m01, sparseY=FALSE)
(K6  <- KhatriRao(diag(3),     t(m01)))
K6D  <- KhatriRao(Diagonal(3), t(m01))
K6d  <- KhatriRao(diag(3),     t(m01), sparseY=FALSE)
K6Dd <- KhatriRao(Diagonal(3), t(m01), sparseY=FALSE)
stopifnot(exprs = {
    all(K5 == K5d)
    identical(cbind(c(7L, 10L), c(3L, 4L)),
              which(K5 != 0, arr.ind = TRUE, useNames=FALSE))
    identical(K5d, K5Dd)
    identical(K6, K6D)
    all(K6 == K6d)
    identical(cbind(3:4, 1L),
              which(K6 != 0, arr.ind = TRUE, useNames=FALSE))
    identical(K6d, K6Dd)
})



cleanEx()
nameEx("LU-class")
### * LU-class

flush(stderr()); flush(stdout())

### Name: denseLU-class
### Title: Dense LU Factorizations
### Aliases: denseLU-class coerce,denseLU,dgeMatrix-method
###   determinant,denseLU,logical-method
### Keywords: algebra array classes

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("denseLU")
set.seed(1)

n <- 3L
(A <- Matrix(round(rnorm(n * n), 2L), n, n))

## With dimnames, to see that they are propagated :
dimnames(A) <- dn <- list(paste0("r", seq_len(n)),
                          paste0("c", seq_len(n)))

(lu.A <- lu(A))
str(e.lu.A <- expand2(lu.A), max.level = 2L)

## Underlying LAPACK representation
(m.lu.A <- as(lu.A, "dgeMatrix")) # which is L and U interlaced
stopifnot(identical(as(m.lu.A, "matrix"), `dim<-`(lu.A@x, lu.A@Dim)))

ae1 <- function(a, b, ...) all.equal(as(a, "matrix"), as(b, "matrix"), ...)
ae2 <- function(a, b, ...) ae1(unname(a), unname(b), ...)

## A ~ P1' L U in floating point
stopifnot(exprs = {
    identical(names(e.lu.A), c("P1.", "L", "U"))
    identical(e.lu.A[["P1."]],
              new(  "pMatrix", Dim = c(n, n), Dimnames = c(dn[1L], list(NULL)),
                  margin = 1L, perm = invertPerm(asPerm(lu.A@perm))))
    identical(e.lu.A[["L"]],
              new("dtrMatrix", Dim = c(n, n), Dimnames = list(NULL, NULL),
                  uplo = "L", diag = "U", x = lu.A@x))
    identical(e.lu.A[["U"]],
              new("dtrMatrix", Dim = c(n, n), Dimnames = c(list(NULL), dn[2L]),
                  uplo = "U", diag = "N", x = lu.A@x))
    ae1(A, with(e.lu.A, P1. %*% L %*% U))
    ae2(A[asPerm(lu.A@perm), ], with(e.lu.A, L %*% U))
})

## Factorization handled as factorized matrix
b <- rnorm(n)
stopifnot(identical(det(A), det(lu.A)),
          identical(solve(A, b), solve(lu.A, b)))



cleanEx()
nameEx("Matrix-class")
### * Matrix-class

flush(stderr()); flush(stdout())

### Name: Matrix-class
### Title: Virtual Class "Matrix" of Matrices
### Aliases: Matrix-class !,Matrix-method &,Matrix,ddiMatrix-method
###   &,Matrix,ldiMatrix-method &,Matrix,ndiMatrix-method
###   *,Matrix,ddiMatrix-method *,Matrix,ldiMatrix-method
###   *,Matrix,ndiMatrix-method +,Matrix,missing-method
###   -,Matrix,missing-method Arith,Matrix,Matrix-method
###   Arith,Matrix,lsparseMatrix-method Arith,Matrix,nsparseMatrix-method
###   Logic,ANY,Matrix-method Logic,Matrix,ANY-method
###   Logic,Matrix,nMatrix-method Math2,Matrix-method Ops,ANY,Matrix-method
###   Ops,Matrix,ANY-method Ops,Matrix,NULL-method
###   Ops,Matrix,ddiMatrix-method Ops,Matrix,ldiMatrix-method
###   Ops,Matrix,matrix-method Ops,Matrix,sparseVector-method
###   Ops,NULL,Matrix-method Ops,matrix,Matrix-method
###   ^,Matrix,ddiMatrix-method ^,Matrix,ldiMatrix-method
###   ^,Matrix,ndiMatrix-method as.array,Matrix-method
###   as.complex,Matrix-method as.integer,Matrix-method
###   as.logical,Matrix-method as.matrix,Matrix-method
###   as.numeric,Matrix-method as.vector,Matrix-method
###   coerce,ANY,Matrix-method coerce,Matrix,CsparseMatrix-method
###   coerce,Matrix,RsparseMatrix-method coerce,Matrix,TsparseMatrix-method
###   coerce,Matrix,corMatrix-method coerce,Matrix,dMatrix-method
###   coerce,Matrix,ddenseMatrix-method coerce,Matrix,denseMatrix-method
###   coerce,Matrix,diagonalMatrix-method coerce,Matrix,dpoMatrix-method
###   coerce,Matrix,dppMatrix-method coerce,Matrix,dsparseMatrix-method
###   coerce,Matrix,generalMatrix-method coerce,Matrix,indMatrix-method
###   coerce,Matrix,lMatrix-method coerce,Matrix,ldenseMatrix-method
###   coerce,Matrix,lsparseMatrix-method coerce,Matrix,matrix-method
###   coerce,Matrix,nMatrix-method coerce,Matrix,ndenseMatrix-method
###   coerce,Matrix,nsparseMatrix-method coerce,Matrix,pMatrix-method
###   coerce,Matrix,packedMatrix-method coerce,Matrix,pcorMatrix-method
###   coerce,Matrix,sparseMatrix-method coerce,Matrix,sparseVector-method
###   coerce,Matrix,symmetricMatrix-method
###   coerce,Matrix,triangularMatrix-method
###   coerce,Matrix,unpackedMatrix-method coerce,matrix,Matrix-method
###   coerce,vector,Matrix-method determinant,Matrix,missing-method
###   determinant,Matrix,logical-method dim,Matrix-method
###   dimnames,Matrix-method dimnames<-,Matrix,NULL-method
###   dimnames<-,Matrix,list-method drop,Matrix-method head,Matrix-method
###   initialize,Matrix-method length,Matrix-method tail,Matrix-method
###   unname,Matrix-method zapsmall,Matrix-method c.Matrix Matrix.Version
###   det
### Keywords: array classes

### ** Examples

slotNames("Matrix")

cl <- getClass("Matrix")
names(cl@subclasses) # more than 40 ..

showClass("Matrix")#> output with slots and all subclasses

(M <- Matrix(c(0,1,0,0), 6, 4))
dim(M)
diag(M)
cm <- M[1:4,] + 10*Diagonal(4)
diff(M)
## can reshape it even :
dim(M) <- c(2, 12)
M
stopifnot(identical(M, Matrix(c(0,1,0,0), 2,12)),
          all.equal(det(cm),
                    determinant(as(cm,"matrix"), log=FALSE)$modulus,
                    check.attributes=FALSE))



cleanEx()
nameEx("Matrix")
### * Matrix

flush(stderr()); flush(stdout())

### Name: Matrix
### Title: Construct a Classed Matrix
### Aliases: Matrix
### Keywords: array utilities

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
Matrix(0, 3, 2)             # 3 by 2 matrix of zeros -> sparse
Matrix(0, 3, 2, sparse=FALSE)# -> 'dense'

## 4 cases - 3 different results :
Matrix(0, 2, 2)              # diagonal !
Matrix(0, 2, 2, sparse=FALSE)# (ditto)
Matrix(0, 2, 2,               doDiag=FALSE)# -> sparse symm. "dsCMatrix"
Matrix(0, 2, 2, sparse=FALSE, doDiag=FALSE)# -> dense  symm. "dsyMatrix"

Matrix(1:6, 3, 2)           # a 3 by 2 matrix (+ integer warning)
Matrix(1:6 + 1, nrow=3)

## logical ones:
Matrix(diag(4) >  0) # -> "ldiMatrix" with diag = "U"
Matrix(diag(4) >  0, sparse=TRUE) #  (ditto)
Matrix(diag(4) >= 0) # -> "lsyMatrix" (of all 'TRUE')
## triangular
l3 <- upper.tri(matrix(,3,3))
(M <- Matrix(l3))   # -> "ltCMatrix"
Matrix(! l3)        # -> "ltrMatrix"
as(l3, "CsparseMatrix")# "lgCMatrix"

Matrix(1:9, nrow=3,
       dimnames = list(c("a", "b", "c"), c("A", "B", "C")))
(I3 <- Matrix(diag(3)))# identity, i.e., unit "diagonalMatrix"
str(I3) # note  'diag = "U"' and the empty 'x' slot

(A <- cbind(a=c(2,1), b=1:2))# symmetric *apart* from dimnames
Matrix(A)                    # hence 'dgeMatrix'
(As <- Matrix(A, dimnames = list(NULL,NULL)))# -> symmetric
forceSymmetric(A) # also symmetric, w/ symm. dimnames
stopifnot(is(As, "symmetricMatrix"),
          is(Matrix(0, 3,3), "sparseMatrix"),
          is(Matrix(FALSE, 1,1), "sparseMatrix"))



cleanEx()
nameEx("MatrixClass")
### * MatrixClass

flush(stderr()); flush(stdout())

### Name: MatrixClass
### Title: The Matrix (Super-) Class of a Class
### Aliases: MatrixClass
### Keywords: utilities

### ** Examples

mkA <- setClass("A", contains="dgCMatrix")
(A <- mkA())
stopifnot(identical(
     MatrixClass("A"),
     "dgCMatrix"))



cleanEx()
nameEx("MatrixFactorization-class")
### * MatrixFactorization-class

flush(stderr()); flush(stdout())

### Name: MatrixFactorization-class
### Title: Virtual Class "MatrixFactorization" of Matrix Factorizations
### Aliases: MatrixFactorization-class BunchKaufmanFactorization-class
###   CholeskyFactorization-class SchurFactorization-class LU-class
###   QR-class determinant,MatrixFactorization,missing-method
###   dim,MatrixFactorization-method dimnames,MatrixFactorization-method
###   dimnames<-,MatrixFactorization,NULL-method
###   dimnames<-,MatrixFactorization,list-method
###   length,MatrixFactorization-method show,MatrixFactorization-method
###   unname,MatrixFactorization-method
###   show,BunchKaufmanFactorization-method
###   show,CholeskyFactorization-method show,SchurFactorization-method
###   show,LU-method show,QR-method
### Keywords: algebra array classes

### ** Examples

showClass("MatrixFactorization")



cleanEx()
nameEx("RsparseMatrix-class")
### * RsparseMatrix-class

flush(stderr()); flush(stdout())

### Name: RsparseMatrix-class
### Title: Class "RsparseMatrix" of Sparse Matrices in Row-compressed Form
### Aliases: RsparseMatrix-class coerce,matrix,RsparseMatrix-method
###   coerce,vector,RsparseMatrix-method diag,RsparseMatrix-method
###   diag<-,RsparseMatrix-method t,RsparseMatrix-method
### Keywords: array classes

### ** Examples

showClass("RsparseMatrix")



cleanEx()
nameEx("Schur-class")
### * Schur-class

flush(stderr()); flush(stdout())

### Name: Schur-class
### Title: Schur Factorizations
### Aliases: Schur-class determinant,Schur,logical-method
### Keywords: algebra array classes

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("Schur")
set.seed(0)

n <- 4L
(A <- Matrix(rnorm(n * n), n, n))

## With dimnames, to see that they are propagated :
dimnames(A) <- list(paste0("r", seq_len(n)),
                    paste0("c", seq_len(n)))

(sch.A <- Schur(A))
str(e.sch.A <- expand2(sch.A), max.level = 2L)

## A ~ Q T Q' in floating point
stopifnot(exprs = {
    identical(names(e.sch.A), c("Q", "T", "Q."))
    all.equal(A, with(e.sch.A, Q %*% T %*% Q.))
})

## Factorization handled as factorized matrix
b <- rnorm(n)
stopifnot(all.equal(det(A), det(sch.A)),
          all.equal(solve(A, b), solve(sch.A, b)))

## One of the non-general cases:
Schur(Diagonal(6L))



cleanEx()
nameEx("Schur")
### * Schur

flush(stderr()); flush(stdout())

### Name: Schur-methods
### Title: Methods for Schur Factorization
### Aliases: Schur Schur-methods Schur,dgeMatrix-method
###   Schur,diagonalMatrix-method Schur,dsyMatrix-method
###   Schur,generalMatrix-method Schur,matrix-method
###   Schur,symmetricMatrix-method Schur,triangularMatrix-method
### Keywords: algebra array methods

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showMethods("Schur", inherited = FALSE)
set.seed(0)

Schur(Hilbert(9L)) # real eigenvalues

(A <- Matrix(round(rnorm(25L, sd = 100)), 5L, 5L))
(sch.A <- Schur(A)) # complex eigenvalues

## A ~ Q T Q' in floating point
str(e.sch.A <- expand2(sch.A), max.level = 2L)
stopifnot(all.equal(A, Reduce(`%*%`, e.sch.A)))

(e1 <- eigen(sch.A@T, only.values = TRUE)$values)
(e2 <- eigen(    A  , only.values = TRUE)$values)
(e3 <- sch.A@EValues)

stopifnot(exprs = {
    all.equal(e1, e2, tolerance = 1e-13)
    all.equal(e1, e3[order(Mod(e3), decreasing = TRUE)], tolerance = 1e-13) 
    identical(Schur(A, vectors = FALSE),
              list(T = sch.A@T, EValues = e3))    
    identical(Schur(as(A, "matrix")),
              list(Q = as(sch.A@Q, "matrix"),
                   T = as(sch.A@T, "matrix"), EValues = e3))
})



cleanEx()
nameEx("Subassign-methods")
### * Subassign-methods

flush(stderr()); flush(stdout())

### Name: Subassign-methods
### Title: Methods for "[<-" - Assigning to Subsets for 'Matrix'
### Aliases: [<- [<--methods Subassign-methods
###   [<-,CsparseMatrix,Matrix,missing,replValue-method
###   [<-,CsparseMatrix,index,index,replValue-method
###   [<-,CsparseMatrix,index,index,sparseVector-method
###   [<-,CsparseMatrix,index,missing,replValue-method
###   [<-,CsparseMatrix,index,missing,sparseVector-method
###   [<-,CsparseMatrix,matrix,missing,replValue-method
###   [<-,CsparseMatrix,missing,index,replValue-method
###   [<-,CsparseMatrix,missing,index,sparseVector-method
###   [<-,Matrix,ANY,ANY,ANY-method [<-,Matrix,ANY,ANY,Matrix-method
###   [<-,Matrix,ANY,ANY,matrix-method [<-,Matrix,ANY,missing,Matrix-method
###   [<-,Matrix,ANY,missing,matrix-method
###   [<-,Matrix,ldenseMatrix,missing,replValue-method
###   [<-,Matrix,lsparseMatrix,missing,replValue-method
###   [<-,Matrix,matrix,missing,replValue-method
###   [<-,Matrix,missing,ANY,Matrix-method
###   [<-,Matrix,missing,ANY,matrix-method
###   [<-,Matrix,ndenseMatrix,missing,replValue-method
###   [<-,Matrix,nsparseMatrix,missing,replValue-method
###   [<-,RsparseMatrix,index,index,replValue-method
###   [<-,RsparseMatrix,index,index,sparseVector-method
###   [<-,RsparseMatrix,index,missing,replValue-method
###   [<-,RsparseMatrix,index,missing,sparseVector-method
###   [<-,RsparseMatrix,matrix,missing,replValue-method
###   [<-,RsparseMatrix,missing,index,replValue-method
###   [<-,RsparseMatrix,missing,index,sparseVector-method
###   [<-,TsparseMatrix,Matrix,missing,replValue-method
###   [<-,TsparseMatrix,index,index,replValue-method
###   [<-,TsparseMatrix,index,index,sparseVector-method
###   [<-,TsparseMatrix,index,missing,replValue-method
###   [<-,TsparseMatrix,index,missing,sparseVector-method
###   [<-,TsparseMatrix,matrix,missing,replValue-method
###   [<-,TsparseMatrix,missing,index,replValue-method
###   [<-,TsparseMatrix,missing,index,sparseVector-method
###   [<-,denseMatrix,index,index,replValue-method
###   [<-,denseMatrix,index,missing,replValue-method
###   [<-,denseMatrix,matrix,missing,replValue-method
###   [<-,denseMatrix,missing,index,replValue-method
###   [<-,denseMatrix,missing,missing,ANY-method
###   [<-,diagonalMatrix,index,index,replValue-method
###   [<-,diagonalMatrix,index,index,sparseMatrix-method
###   [<-,diagonalMatrix,index,index,sparseVector-method
###   [<-,diagonalMatrix,index,missing,replValue-method
###   [<-,diagonalMatrix,index,missing,sparseMatrix-method
###   [<-,diagonalMatrix,index,missing,sparseVector-method
###   [<-,diagonalMatrix,matrix,missing,replValue-method
###   [<-,diagonalMatrix,missing,index,replValue-method
###   [<-,diagonalMatrix,missing,index,sparseMatrix-method
###   [<-,diagonalMatrix,missing,index,sparseVector-method
###   [<-,diagonalMatrix,missing,missing,ANY-method
###   [<-,indMatrix,index,index,ANY-method
###   [<-,indMatrix,index,missing,ANY-method
###   [<-,indMatrix,missing,index,ANY-method
###   [<-,indMatrix,missing,missing,ANY-method
###   [<-,sparseMatrix,ANY,ANY,sparseMatrix-method
###   [<-,sparseMatrix,ANY,missing,sparseMatrix-method
###   [<-,sparseMatrix,missing,ANY,sparseMatrix-method
###   [<-,sparseMatrix,missing,missing,ANY-method
###   [<-,sparseVector,index,missing,replValueSp-method
###   [<-,sparseVector,sparseVector,missing,replValueSp-method
### Keywords: array methods

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
set.seed(101)
(a <- m <- Matrix(round(rnorm(7*4),2), nrow = 7))

a[] <- 2.2 # <<- replaces **every** entry
a
## as do these:
a[,] <- 3 ; a[TRUE,] <- 4

m[2, 3]  <- 3.14 # simple number
m[3, 3:4]<- 3:4  # simple numeric of length 2

## sub matrix assignment:
m[-(4:7), 3:4] <- cbind(1,2:4) #-> upper right corner of 'm'
m[3:5, 2:3] <- 0
m[6:7, 1:2] <- Diagonal(2)
m

## rows or columns only:
m[1,] <- 10
m[,2] <- 1:7
m[-(1:6), ] <- 3:0 # not the first 6 rows, i.e. only the 7th
as(m, "sparseMatrix")



cleanEx()
nameEx("TsparseMatrix-class")
### * TsparseMatrix-class

flush(stderr()); flush(stdout())

### Name: TsparseMatrix-class
### Title: Class "TsparseMatrix" of Sparse Matrices in Triplet Form
### Aliases: TsparseMatrix-class coerce,matrix,TsparseMatrix-method
###   coerce,vector,TsparseMatrix-method diag,TsparseMatrix-method
###   diag<-,TsparseMatrix-method t,TsparseMatrix-method
### Keywords: array classes

### ** Examples

showClass("TsparseMatrix")
## or just the subclasses' names
names(getClass("TsparseMatrix")@subclasses)

T3 <- spMatrix(3,4, i=c(1,3:1), j=c(2,4:2), x=1:4)
T3 # only 3 non-zero entries, 5 = 1+4 !
## Don't show: 
stopifnot(nnzero(T3) == 3)
## End(Don't show)



cleanEx()
nameEx("USCounties")
### * USCounties

flush(stderr()); flush(stdout())

### Name: USCounties
### Title: Contiguity Matrix of U.S. Counties
### Aliases: USCounties
### Keywords: datasets

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
data(USCounties, package = "Matrix")
(n <- ncol(USCounties))
I <- .symDiagonal(n)

set.seed(1)
r <- 50L
rho <- 1 / runif(r, 0, 0.5)

system.time(MJ0 <- sapply(rho, function(mult)
    determinant(USCounties + mult * I, logarithm = TRUE)$modulus))

## Can be done faster by updating the Cholesky factor:

C1 <- Cholesky(USCounties, Imult = 2)
system.time(MJ1 <- sapply(rho, function(mult)
    determinant(update(C1, USCounties, mult), sqrt = FALSE)$modulus))
stopifnot(all.equal(MJ0, MJ1))

C2 <- Cholesky(USCounties, super = TRUE, Imult = 2)
system.time(MJ2 <- sapply(rho, function(mult)
    determinant(update(C2, USCounties, mult), sqrt = FALSE)$modulus))
stopifnot(all.equal(MJ0, MJ2))



cleanEx()
nameEx("Xtrct-methods")
### * Xtrct-methods

flush(stderr()); flush(stdout())

### Name: Subscript-methods
### Title: Methods for "[": Extraction or Subsetting in Package 'Matrix'
### Aliases: [ [-methods Subscript-methods [,Matrix,ANY,NULL,ANY-method
###   [,Matrix,NULL,ANY,ANY-method [,Matrix,NULL,NULL,ANY-method
###   [,Matrix,index,index,logical-method
###   [,Matrix,index,index,missing-method
###   [,Matrix,index,missing,logical-method
###   [,Matrix,index,missing,missing-method
###   [,Matrix,lMatrix,missing,missing-method
###   [,Matrix,matrix,missing,missing-method
###   [,Matrix,missing,index,logical-method
###   [,Matrix,missing,index,missing-method
###   [,Matrix,missing,missing,logical-method
###   [,Matrix,missing,missing,missing-method
###   [,Matrix,nMatrix,missing,missing-method
###   [,abIndex,index,ANY,ANY-method [,sparseVector,NULL,ANY,ANY-method
###   [,sparseVector,index,missing,missing-method
###   [,sparseVector,lsparseVector,missing,missing-method
###   [,sparseVector,missing,missing,missing-method
###   [,sparseVector,nsparseVector,missing,missing-method
### Keywords: array methods

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
str(m <- Matrix(round(rnorm(7*4),2), nrow = 7))
stopifnot(identical(m, m[]))
m[2, 3]   # simple number
m[2, 3:4] # simple numeric of length 2
m[2, 3:4, drop=FALSE] # sub matrix of class 'dgeMatrix'
## rows or columns only:
m[1,]     # first row, as simple numeric vector
m[,1:2]   # sub matrix of first two columns

showMethods("[", inherited = FALSE)



cleanEx()
nameEx("abIndex-class")
### * abIndex-class

flush(stderr()); flush(stdout())

### Name: abIndex-class
### Title: Class "abIndex" of Abstract Index Vectors
### Aliases: abIndex-class seqMat-class Arith,abIndex,abIndex-method
###   Arith,abIndex,numLike-method Arith,numLike,abIndex-method
###   Ops,ANY,abIndex-method Ops,abIndex,ANY-method
###   Ops,abIndex,abIndex-method Summary,abIndex-method
###   as.integer,abIndex-method as.numeric,abIndex-method
###   as.vector,abIndex-method coerce,abIndex,integer-method
###   coerce,abIndex,numeric-method coerce,abIndex,seqMat-method
###   coerce,abIndex,vector-method coerce,logical,abIndex-method
###   coerce,numeric,abIndex-method drop,abIndex-method
###   length,abIndex-method show,abIndex-method
###   coerce,numeric,seqMat-method coerce,seqMat,abIndex-method
###   coerce,seqMat,numeric-method
### Keywords: classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("abIndex")
ii <- c(-3:40, 20:70)
str(ai <- as(ii, "abIndex"))# note
ai # -> show() method

stopifnot(identical(-3:20,
                    as(abIseq1(-3,20), "vector")))



cleanEx()
nameEx("abIseq")
### * abIseq

flush(stderr()); flush(stdout())

### Name: abIseq
### Title: Sequence Generation of "abIndex", Abstract Index Vectors
### Aliases: abIseq abIseq1 c.abIndex
### Keywords: manip utilities

### ** Examples

stopifnot(identical(-3:20,
                    as(abIseq1(-3,20), "vector")))

try( ## (arithmetic) not yet implemented
abIseq(1, 50, by = 3)
)




cleanEx()
nameEx("all.equal-methods")
### * all.equal-methods

flush(stderr()); flush(stdout())

### Name: all.equal-methods
### Title: Matrix Package Methods for Function all.equal()
### Aliases: all.equal all.equal-methods all.equal,Matrix,Matrix-method
###   all.equal,Matrix,sparseVector-method all.equal,Matrix,vector-method
###   all.equal,abIndex,abIndex-method all.equal,abIndex,numLike-method
###   all.equal,numLike,abIndex-method all.equal,sparseVector,Matrix-method
###   all.equal,sparseVector,sparseVector-method
###   all.equal,sparseVector,vector-method all.equal,vector,Matrix-method
###   all.equal,vector,sparseVector-method
### Keywords: arith logic methods programming

### ** Examples

showMethods("all.equal")

(A <- spMatrix(3,3, i= c(1:3,2:1), j=c(3:1,1:2), x = 1:5))
ex <- expand(lu. <- lu(A))
stopifnot( all.equal(as(A[lu.@p + 1L, lu.@q + 1L], "CsparseMatrix"),
                     lu.@L %*% lu.@U),
           with(ex, all.equal(as(P %*% A %*% t(Q), "CsparseMatrix"),
                              L %*% U)),
           with(ex, all.equal(as(A, "CsparseMatrix"),
                              t(P) %*% L %*% U %*% Q)))



cleanEx()
nameEx("atomicVector-class")
### * atomicVector-class

flush(stderr()); flush(stdout())

### Name: atomicVector-class
### Title: Virtual Class "atomicVector" of Atomic Vectors
### Aliases: atomicVector-class Ops,atomicVector,sparseVector-method
###   coerce,atomicVector,dsparseVector-method
###   coerce,atomicVector,sparseVector-method
### Keywords: classes

### ** Examples

showClass("atomicVector")



cleanEx()
nameEx("band")
### * band

flush(stderr()); flush(stdout())

### Name: band-methods
### Title: Extract bands of a matrix
### Aliases: band band-methods triu triu-methods tril tril-methods
###   band,CsparseMatrix-method band,RsparseMatrix-method
###   band,TsparseMatrix-method band,denseMatrix-method
###   band,diagonalMatrix-method band,indMatrix-method band,matrix-method
###   triu,CsparseMatrix-method triu,RsparseMatrix-method
###   triu,TsparseMatrix-method triu,denseMatrix-method
###   triu,diagonalMatrix-method triu,indMatrix-method triu,matrix-method
###   tril,CsparseMatrix-method tril,RsparseMatrix-method
###   tril,TsparseMatrix-method tril,denseMatrix-method
###   tril,diagonalMatrix-method tril,indMatrix-method tril,matrix-method
### Keywords: array methods

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
## A random sparse matrix :
set.seed(7)
m <- matrix(0, 5, 5)
m[sample(length(m), size = 14)] <- rep(1:9, length=14)
(mm <- as(m, "CsparseMatrix"))

tril(mm)        # lower triangle
tril(mm, -1)    # strict lower triangle
triu(mm,  1)    # strict upper triangle
band(mm, -1, 2) # general band
(m5 <- Matrix(rnorm(25), ncol = 5))
tril(m5)        # lower triangle
tril(m5, -1)    # strict lower triangle
triu(m5, 1)     # strict upper triangle
band(m5, -1, 2) # general band
(m65 <- Matrix(rnorm(30), ncol = 5))  # not square
triu(m65)       # result not "dtrMatrix" unless square
(sm5 <- crossprod(m65)) # symmetric
   band(sm5, -1, 1)# "dsyMatrix": symmetric band preserves symmetry property
as(band(sm5, -1, 1), "sparseMatrix")# often preferable
(sm <- round(crossprod(triu(mm/2)))) # sparse symmetric ("dsC*")
band(sm, -1,1) # remains "dsC", *however*
band(sm, -2,1) # -> "dgC"
## Don't show: 
 ## this uses special methods
(x.x <- crossprod(mm))
tril(x.x)
xx <- tril(x.x) + triu(x.x, 1) ## the same as x.x (but stored differently):
txx <- t(as(xx, "symmetricMatrix"))
stopifnot(identical(triu(x.x), t(tril(x.x))),
	  identical(class(x.x), class(txx)),
	  identical(as(x.x, "generalMatrix"), as(txx, "generalMatrix")))
## End(Don't show)



cleanEx()
nameEx("bandSparse")
### * bandSparse

flush(stderr()); flush(stdout())

### Name: bandSparse
### Title: Construct Sparse Banded Matrix from (Sup-/Super-) Diagonals
### Aliases: bandSparse
### Keywords: array utilities

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
diags <- list(1:30, 10*(1:20), 100*(1:20))
s1 <- bandSparse(13, k = -c(0:2, 6), diag = c(diags, diags[2]), symm=TRUE)
s1
s2 <- bandSparse(13, k =  c(0:2, 6), diag = c(diags, diags[2]), symm=TRUE)
stopifnot(identical(s1, t(s2)), is(s1,"dsCMatrix"))

## a pattern Matrix of *full* (sub-)diagonals:
bk <- c(0:4, 7,9)
(s3 <- bandSparse(30, k = bk, symm = TRUE))

## If you want a pattern matrix, but with "sparse"-diagonals,
## you currently need to go via logical sparse:
lLis <- lapply(list(rpois(20, 2), rpois(20, 1), rpois(20, 3))[c(1:3, 2:3, 3:2)],
               as.logical)
(s4 <- bandSparse(20, k = bk, symm = TRUE, diag = lLis))
(s4. <- as(drop0(s4), "nsparseMatrix"))

n <- 1e4
bk <- c(0:5, 7,11)
bMat <- matrix(1:8, n, 8, byrow=TRUE)
bLis <- as.data.frame(bMat)
B  <- bandSparse(n, k = bk, diag = bLis)
Bs <- bandSparse(n, k = bk, diag = bLis, symmetric=TRUE)
B [1:15, 1:30]
Bs[1:15, 1:30]
## can use a list *or* a matrix for specifying the diagonals:
stopifnot(identical(B,  bandSparse(n, k = bk, diag = bMat)),
	  identical(Bs, bandSparse(n, k = bk, diag = bMat, symmetric=TRUE))
          , inherits(B, "dtCMatrix") # triangular!
)



cleanEx()
nameEx("bdiag")
### * bdiag

flush(stderr()); flush(stdout())

### Name: bdiag
### Title: Construct a Block Diagonal Matrix
### Aliases: bdiag .bdiag
### Keywords: array utilities

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
bdiag(matrix(1:4, 2), diag(3))
## combine "Matrix" class and traditional matrices:
bdiag(Diagonal(2), matrix(1:3, 3,4), diag(3:2))

mlist <- list(1, 2:3, diag(x=5:3), 27, cbind(1,3:6), 100:101)
bdiag(mlist)
stopifnot(identical(bdiag(mlist), 
                    bdiag(lapply(mlist, as.matrix))))

ml <- c(as(matrix((1:24)%% 11 == 0, 6,4),"nMatrix"),
        rep(list(Diagonal(2, x=TRUE)), 3))
mln <- c(ml, Diagonal(x = 1:3))
stopifnot(is(bdiag(ml), "lsparseMatrix"),
          is(bdiag(mln),"dsparseMatrix") )

## random (diagonal-)block-triangular matrices:
rblockTri <- function(nb, max.ni, lambda = 3) {
   .bdiag(replicate(nb, {
         n <- sample.int(max.ni, 1)
         tril(Matrix(rpois(n * n, lambda = lambda), n, n)) }))
}

(T4 <- rblockTri(4, 10, lambda = 1))
image(T1 <- rblockTri(12, 20))


##' Fast version of Matrix :: .bdiag() -- for the case of *many*  (k x k) matrices:
##' @param lmat list(<mat1>, <mat2>, ....., <mat_N>)  where each mat_j is a  k x k 'matrix'
##' @return a sparse (N*k x N*k) matrix of class  \code{"\linkS4class{dgCMatrix}"}.
bdiag_m <- function(lmat) {
    ## Copyright (C) 2016 Martin Maechler, ETH Zurich
    if(!length(lmat)) return(new("dgCMatrix"))
    stopifnot(is.list(lmat), is.matrix(lmat[[1]]),
              (k <- (d <- dim(lmat[[1]]))[1]) == d[2], # k x k
              all(vapply(lmat, dim, integer(2)) == k)) # all of them
    N <- length(lmat)
    if(N * k > .Machine$integer.max)
        stop("resulting matrix too large; would be  M x M, with M=", N*k)
    M <- as.integer(N * k)
    ## result: an   M x M  matrix
    new("dgCMatrix", Dim = c(M,M),
        ## 'i :' maybe there's a faster way (w/o matrix indexing), but elegant?
        i = as.vector(matrix(0L:(M-1L), nrow=k)[, rep(seq_len(N), each=k)]),
        p = k * 0L:M,
        x = as.double(unlist(lmat, recursive=FALSE, use.names=FALSE)))
}

l12 <- replicate(12, matrix(rpois(16, lambda = 6.4), 4, 4),
                 simplify=FALSE)
dim(T12 <- bdiag_m(l12))# 48 x 48
T12[1:20, 1:20]



cleanEx()
nameEx("boolean-matprod")
### * boolean-matprod

flush(stderr()); flush(stdout())

### Name: boolmatmult-methods
### Title: Boolean Arithmetic Matrix Products: '%&%' and Methods
### Aliases: %&% %&%-methods boolmatmult-methods %&%,ANY,ANY-method
###   %&%,ANY,Matrix-method %&%,ANY,matrix-method
###   %&%,ANY,sparseVector-method %&%,ANY,vector-method
###   %&%,CsparseMatrix,CsparseMatrix-method
###   %&%,CsparseMatrix,RsparseMatrix-method
###   %&%,CsparseMatrix,TsparseMatrix-method
###   %&%,CsparseMatrix,denseMatrix-method
###   %&%,CsparseMatrix,diagonalMatrix-method
###   %&%,CsparseMatrix,matrix-method %&%,CsparseMatrix,vector-method
###   %&%,Matrix,ANY-method %&%,Matrix,indMatrix-method
###   %&%,Matrix,pMatrix-method %&%,Matrix,sparseVector-method
###   %&%,RsparseMatrix,CsparseMatrix-method
###   %&%,RsparseMatrix,RsparseMatrix-method
###   %&%,RsparseMatrix,TsparseMatrix-method
###   %&%,RsparseMatrix,denseMatrix-method
###   %&%,RsparseMatrix,diagonalMatrix-method
###   %&%,RsparseMatrix,matrix-method %&%,RsparseMatrix,vector-method
###   %&%,TsparseMatrix,CsparseMatrix-method
###   %&%,TsparseMatrix,RsparseMatrix-method
###   %&%,TsparseMatrix,TsparseMatrix-method
###   %&%,TsparseMatrix,denseMatrix-method
###   %&%,TsparseMatrix,diagonalMatrix-method
###   %&%,TsparseMatrix,matrix-method %&%,TsparseMatrix,vector-method
###   %&%,denseMatrix,CsparseMatrix-method
###   %&%,denseMatrix,RsparseMatrix-method
###   %&%,denseMatrix,TsparseMatrix-method
###   %&%,denseMatrix,denseMatrix-method
###   %&%,denseMatrix,diagonalMatrix-method %&%,denseMatrix,matrix-method
###   %&%,denseMatrix,vector-method %&%,diagonalMatrix,CsparseMatrix-method
###   %&%,diagonalMatrix,RsparseMatrix-method
###   %&%,diagonalMatrix,TsparseMatrix-method
###   %&%,diagonalMatrix,denseMatrix-method
###   %&%,diagonalMatrix,diagonalMatrix-method
###   %&%,diagonalMatrix,matrix-method %&%,diagonalMatrix,vector-method
###   %&%,indMatrix,Matrix-method %&%,indMatrix,indMatrix-method
###   %&%,indMatrix,matrix-method %&%,indMatrix,pMatrix-method
###   %&%,indMatrix,vector-method %&%,matrix,ANY-method
###   %&%,matrix,CsparseMatrix-method %&%,matrix,RsparseMatrix-method
###   %&%,matrix,TsparseMatrix-method %&%,matrix,denseMatrix-method
###   %&%,matrix,diagonalMatrix-method %&%,matrix,indMatrix-method
###   %&%,matrix,matrix-method %&%,matrix,pMatrix-method
###   %&%,matrix,sparseVector-method %&%,matrix,vector-method
###   %&%,pMatrix,Matrix-method %&%,pMatrix,indMatrix-method
###   %&%,pMatrix,matrix-method %&%,pMatrix,pMatrix-method
###   %&%,pMatrix,vector-method %&%,sparseVector,ANY-method
###   %&%,sparseVector,Matrix-method %&%,sparseVector,matrix-method
###   %&%,sparseVector,sparseVector-method %&%,sparseVector,vector-method
###   %&%,vector,ANY-method %&%,vector,CsparseMatrix-method
###   %&%,vector,RsparseMatrix-method %&%,vector,TsparseMatrix-method
###   %&%,vector,denseMatrix-method %&%,vector,diagonalMatrix-method
###   %&%,vector,indMatrix-method %&%,vector,matrix-method
###   %&%,vector,pMatrix-method %&%,vector,sparseVector-method
###   %&%,vector,vector-method
### Keywords: algebra array logic methods

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
set.seed(7)
L <- Matrix(rnorm(20) > 1,    4,5)
(N <- as(L, "nMatrix"))
L. <- L; L.[1:2,1] <- TRUE; L.@x[1:2] <- FALSE; L. # has "zeros" to drop0()
D <- Matrix(round(rnorm(30)), 5,6) # -> values in -1:1 (for this seed)
L %&% D
stopifnot(identical(L %&% D, N %&% D),
          all(L %&% D == as((L %*% abs(D)) > 0, "sparseMatrix")))

## cross products , possibly with  boolArith = TRUE :
crossprod(N)     # -> sparse patter'n' (TRUE/FALSE : boolean arithmetic)
crossprod(N  +0) # -> numeric Matrix (with same "pattern")
stopifnot(all(crossprod(N) == t(N) %&% N),
          identical(crossprod(N), crossprod(N +0, boolArith=TRUE)),
          identical(crossprod(L), crossprod(N   , boolArith=FALSE)))
crossprod(D, boolArith =  TRUE) # pattern: "nsCMatrix"
crossprod(L, boolArith =  TRUE) #  ditto
crossprod(L, boolArith = FALSE) # numeric: "dsCMatrix"



cleanEx()
nameEx("cBind")
### * cBind

flush(stderr()); flush(stdout())

### Name: cbind2-methods
### Title: 'cbind()' and 'rbind()' recursively built on cbind2/rbind2
### Aliases: cbind2 cbind2-methods rbind2 rbind2-methods
###   cbind2,Matrix,Matrix-method cbind2,Matrix,NULL-method
###   cbind2,Matrix,matrix-method cbind2,Matrix,missing-method
###   cbind2,Matrix,vector-method cbind2,NULL,Matrix-method
###   cbind2,matrix,Matrix-method cbind2,vector,Matrix-method
###   rbind2,Matrix,Matrix-method rbind2,Matrix,NULL-method
###   rbind2,Matrix,matrix-method rbind2,Matrix,missing-method
###   rbind2,Matrix,vector-method rbind2,NULL,Matrix-method
###   rbind2,matrix,Matrix-method rbind2,vector,Matrix-method
### Keywords: array manip methods

### ** Examples

(a <- matrix(c(2:1,1:2), 2,2))

(M1 <- cbind(0, rbind(a, 7))) # a traditional matrix

D <- Diagonal(2)
(M2 <- cbind(4, a, D, -1, D, 0)) # a sparse Matrix

stopifnot(validObject(M2), inherits(M2, "sparseMatrix"),
          dim(M2) == c(2,9))



cleanEx()
nameEx("chol")
### * chol

flush(stderr()); flush(stdout())

### Name: chol-methods
### Title: Compute the Cholesky Factor of a Matrix
### Aliases: chol chol-methods chol,ddiMatrix-method
###   chol,diagonalMatrix-method chol,dsCMatrix-method
###   chol,dsRMatrix-method chol,dsTMatrix-method chol,dspMatrix-method
###   chol,dsyMatrix-method chol,generalMatrix-method
###   chol,symmetricMatrix-method chol,triangularMatrix-method
### Keywords: algebra array methods

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
showMethods("chol", inherited = FALSE)
set.seed(0)

## ---- Dense ----------------------------------------------------------

## chol(x, pivot = value) wrapping Cholesky(x, perm = value)
selectMethod("chol", "dsyMatrix")

## Except in packed cases where pivoting is not yet available
selectMethod("chol", "dspMatrix")

## .... Positive definite ..............................................

(A1 <- new("dsyMatrix", Dim = c(2L, 2L), x = c(1, 2, 2, 5)))
(R1.nopivot <- chol(A1))
(R1 <- chol(A1, pivot = TRUE))

## In 2-by-2 cases, we know that the permutation is 1:2 or 2:1,
## even if in general 'chol' does not say ...

stopifnot(exprs = {
   all.equal(  A1           , as(crossprod(R1.nopivot), "dsyMatrix"))
   all.equal(t(A1[2:1, 2:1]), as(crossprod(R1        ), "dsyMatrix"))
   identical(Cholesky(A1)@perm, 2:1) # because 5 > 1
})

## .... Positive semidefinite but not positive definite ................

(A2 <- new("dpoMatrix", Dim = c(2L, 2L), x = c(1, 2, 2, 4)))
try(R2.nopivot <- chol(A2)) # fails as not positive definite
(R2 <- chol(A2, pivot = TRUE)) # returns, with a warning and ...

stopifnot(exprs = {
   all.equal(t(A2[2:1, 2:1]), as(crossprod(R2), "dsyMatrix"))
   identical(Cholesky(A2)@perm, 2:1) # because 4 > 1
})

## .... Not positive semidefinite ......................................

(A3 <- new("dsyMatrix", Dim = c(2L, 2L), x = c(1, 2, 2, 3)))
try(R3.nopivot <- chol(A3)) # fails as not positive definite
(R3 <- chol(A3, pivot = TRUE)) # returns, with a warning and ...

## _Not_ equal: see details and examples in help("Cholesky")
all.equal(t(A3[2:1, 2:1]), as(crossprod(R3), "dsyMatrix"))

## ---- Sparse ---------------------------------------------------------

## chol(x, pivot = value) wrapping
## Cholesky(x, perm = value, LDL = FALSE, super = FALSE)
selectMethod("chol", "dsCMatrix")

## Except in diagonal cases which are handled "directly"
selectMethod("chol", "ddiMatrix")

(A4 <- toeplitz(as(c(10, 0, 1, 0, 3), "sparseVector")))
(ch.A4.nopivot <- Cholesky(A4, perm = FALSE, LDL = FALSE, super = FALSE))
(ch.A4 <- Cholesky(A4, perm = TRUE, LDL = FALSE, super = FALSE))
(R4.nopivot <- chol(A4))
(R4 <- chol(A4, pivot = TRUE))

det4 <- det(A4)
b4 <- rnorm(5L)
x4 <- solve(A4, b4)

stopifnot(exprs = {
    identical(R4.nopivot, expand1(ch.A4.nopivot, "L."))
    identical(R4, expand1(ch.A4, "L."))
    all.equal(A4, crossprod(R4.nopivot))
    all.equal(A4[ch.A4@perm + 1L, ch.A4@perm + 1L], crossprod(R4))
    all.equal(diag(R4.nopivot), sqrt(diag(ch.A4.nopivot)))
    all.equal(diag(R4), sqrt(diag(ch.A4)))
    all.equal(sqrt(det4), det(R4.nopivot))
    all.equal(sqrt(det4), det(R4))
    all.equal(det4, det(ch.A4.nopivot, sqrt = FALSE))
    all.equal(det4, det(ch.A4, sqrt = FALSE))
    all.equal(x4, solve(R4.nopivot, solve(t(R4.nopivot), b4)))
    all.equal(x4, solve(ch.A4.nopivot, b4))
    all.equal(x4, solve(ch.A4, b4))
})



cleanEx()
nameEx("chol2inv-methods")
### * chol2inv-methods

flush(stderr()); flush(stdout())

### Name: chol2inv-methods
### Title: Inverse from Cholesky Factor
### Aliases: chol2inv chol2inv-methods chol2inv,ANY-method
###   chol2inv,ddiMatrix-method chol2inv,diagonalMatrix-method
###   chol2inv,dtCMatrix-method chol2inv,dtRMatrix-method
###   chol2inv,dtTMatrix-method chol2inv,dtrMatrix-method
###   chol2inv,dtpMatrix-method chol2inv,generalMatrix-method
###   chol2inv,symmetricMatrix-method chol2inv,triangularMatrix-method
### Keywords: algebra array methods

### ** Examples

(A <- Matrix(cbind(c(1, 1, 1), c(1, 2, 4), c(1, 4, 16))))
(R <- chol(A))
(L <- t(R))
(R2i <- chol2inv(R))
(L2i <- chol2inv(R))
stopifnot(exprs = {
    all.equal(R2i, tcrossprod(solve(R)))
    all.equal(L2i,  crossprod(solve(L)))
    all.equal(as(R2i %*% A, "matrix"), diag(3L)) # the identity 
    all.equal(as(L2i %*% A, "matrix"), diag(3L)) # ditto
})



cleanEx()
nameEx("colSums")
### * colSums

flush(stderr()); flush(stdout())

### Name: colSums-methods
### Title: Form Row and Column Sums and Means
### Aliases: colSums colSums-methods colMeans colMeans-methods rowSums
###   rowSums-methods rowMeans rowMeans-methods
###   colSums,CsparseMatrix-method colSums,RsparseMatrix-method
###   colSums,TsparseMatrix-method colSums,denseMatrix-method
###   colSums,diagonalMatrix-method colSums,indMatrix-method
###   colMeans,CsparseMatrix-method colMeans,RsparseMatrix-method
###   colMeans,TsparseMatrix-method colMeans,denseMatrix-method
###   colMeans,diagonalMatrix-method colMeans,indMatrix-method
###   rowSums,CsparseMatrix-method rowSums,RsparseMatrix-method
###   rowSums,TsparseMatrix-method rowSums,denseMatrix-method
###   rowSums,diagonalMatrix-method rowSums,indMatrix-method
###   rowMeans,CsparseMatrix-method rowMeans,RsparseMatrix-method
###   rowMeans,TsparseMatrix-method rowMeans,denseMatrix-method
###   rowMeans,diagonalMatrix-method rowMeans,indMatrix-method
### Keywords: algebra arith array methods

### ** Examples

(M <- bdiag(Diagonal(2), matrix(1:3, 3,4), diag(3:2))) # 7 x 8
colSums(M)
d <- Diagonal(10, c(0,0,10,0,2,rep(0,5)))
MM <- kronecker(d, M)
dim(MM) # 70 80
length(MM@x) # 160, but many are '0' ; drop those:
MM <- drop0(MM)
length(MM@x) # 32
  cm <- colSums(MM)
(scm <- colSums(MM, sparseResult = TRUE))
stopifnot(is(scm, "sparseVector"),
          identical(cm, as.numeric(scm)))
rowSums (MM, sparseResult = TRUE) # 14 of 70 are not zero
colMeans(MM, sparseResult = TRUE) # 16 of 80 are not zero
## Since we have no 'NA's, these two are equivalent :
stopifnot(identical(rowMeans(MM, sparseResult = TRUE),
                    rowMeans(MM, sparseResult = TRUE, na.rm = TRUE)),
	  rowMeans(Diagonal(16)) == 1/16,
	  colSums(Diagonal(7)) == 1)

## dimnames(x) -->  names( <value> ) :
dimnames(M) <- list(paste0("r", 1:7), paste0("V",1:8))
M
colSums(M)
rowMeans(M)
## Assertions :
stopifnot(exprs = {
    all.equal(colSums(M),
              structure(c(1,1,6,6,6,6,3,2), names = colnames(M)))
    all.equal(rowMeans(M),
              structure(c(1,1,4,8,12,3,2)/8, names = paste0("r", 1:7)))
})



cleanEx()
nameEx("condest")
### * condest

flush(stderr()); flush(stdout())

### Name: condest
### Title: Compute Approximate CONDition number and 1-Norm of (Large)
###   Matrices
### Aliases: condest onenormest
### Keywords: algebra math utilities

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
data(KNex, package = "Matrix")
mtm <- with(KNex, crossprod(mm))
system.time(ce <- condest(mtm))
sum(abs(ce$v)) ## || v ||_1  == 1
## Prove that  || A v || = || A || / est  (as ||v|| = 1):
stopifnot(all.equal(norm(mtm %*% ce$v),
                    norm(mtm) / ce$est))

## reciprocal
1 / ce$est
system.time(rc <- rcond(mtm)) # takes ca  3 x  longer
rc
all.equal(rc, 1/ce$est) # TRUE -- the approximation was good

one <- onenormest(mtm)
str(one) ## est = 12.3
## the maximal column:
which(one$v == 1) # mostly 4, rarely 1, depending on random seed



cleanEx()
nameEx("dMatrix-class")
### * dMatrix-class

flush(stderr()); flush(stdout())

### Name: dMatrix-class
### Title: (Virtual) Class "dMatrix" of "double" Matrices
### Aliases: dMatrix-class lMatrix-class Compare,dMatrix,logical-method
###   Compare,dMatrix,numeric-method Compare,logical,dMatrix-method
###   Compare,numeric,dMatrix-method Logic,dMatrix,logical-method
###   Logic,dMatrix,numeric-method Logic,dMatrix,sparseVector-method
###   Logic,logical,dMatrix-method Logic,numeric,dMatrix-method
###   Ops,dMatrix,dMatrix-method Ops,dMatrix,ddiMatrix-method
###   Ops,dMatrix,lMatrix-method Ops,dMatrix,ldiMatrix-method
###   Ops,dMatrix,nMatrix-method coerce,matrix,dMatrix-method
###   coerce,vector,dMatrix-method Arith,lMatrix,numeric-method
###   Arith,lMatrix,logical-method Arith,logical,lMatrix-method
###   Arith,numeric,lMatrix-method Compare,lMatrix,logical-method
###   Compare,lMatrix,numeric-method Compare,logical,lMatrix-method
###   Compare,numeric,lMatrix-method Logic,lMatrix,logical-method
###   Logic,lMatrix,numeric-method Logic,lMatrix,sparseVector-method
###   Logic,logical,lMatrix-method Logic,numeric,lMatrix-method
###   Ops,lMatrix,dMatrix-method Ops,lMatrix,lMatrix-method
###   Ops,lMatrix,nMatrix-method Ops,lMatrix,numeric-method
###   Ops,numeric,lMatrix-method coerce,matrix,lMatrix-method
###   coerce,vector,lMatrix-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
 showClass("dMatrix")

 set.seed(101)
 round(Matrix(rnorm(28), 4,7), 2)
 M <- Matrix(rlnorm(56, sd=10), 4,14)
 (M. <- zapsmall(M))
 table(as.logical(M. == 0))



cleanEx()
nameEx("ddenseMatrix-class")
### * ddenseMatrix-class

flush(stderr()); flush(stdout())

### Name: ddenseMatrix-class
### Title: Virtual Class "ddenseMatrix" of Numeric Dense Matrices
### Aliases: ddenseMatrix-class &,ddenseMatrix,ddiMatrix-method
###   &,ddenseMatrix,ldiMatrix-method &,ddenseMatrix,ndiMatrix-method
###   *,ddenseMatrix,ddiMatrix-method *,ddenseMatrix,ldiMatrix-method
###   *,ddenseMatrix,ndiMatrix-method Arith,ddenseMatrix,logical-method
###   Arith,ddenseMatrix,numeric-method
###   Arith,ddenseMatrix,sparseVector-method
###   Arith,logical,ddenseMatrix-method Arith,numeric,ddenseMatrix-method
###   ^,ddenseMatrix,ddiMatrix-method ^,ddenseMatrix,ldiMatrix-method
###   ^,ddenseMatrix,ndiMatrix-method coerce,matrix,ddenseMatrix-method
###   coerce,vector,ddenseMatrix-method
### Keywords: array classes

### ** Examples

showClass("ddenseMatrix")

showMethods(class = "ddenseMatrix", where = "package:Matrix")



cleanEx()
nameEx("ddiMatrix-class")
### * ddiMatrix-class

flush(stderr()); flush(stdout())

### Name: ddiMatrix-class
### Title: Class "ddiMatrix" of Diagonal Numeric Matrices
### Aliases: ddiMatrix-class %%,ddiMatrix,Matrix-method
###   %%,ddiMatrix,ddenseMatrix-method %%,ddiMatrix,ldenseMatrix-method
###   %%,ddiMatrix,ndenseMatrix-method %/%,ddiMatrix,Matrix-method
###   %/%,ddiMatrix,ddenseMatrix-method %/%,ddiMatrix,ldenseMatrix-method
###   %/%,ddiMatrix,ndenseMatrix-method &,ddiMatrix,Matrix-method
###   &,ddiMatrix,ddenseMatrix-method &,ddiMatrix,ldenseMatrix-method
###   &,ddiMatrix,ndenseMatrix-method *,ddiMatrix,Matrix-method
###   *,ddiMatrix,ddenseMatrix-method *,ddiMatrix,ldenseMatrix-method
###   *,ddiMatrix,ndenseMatrix-method /,ddiMatrix,Matrix-method
###   /,ddiMatrix,ddenseMatrix-method /,ddiMatrix,ldenseMatrix-method
###   /,ddiMatrix,ndenseMatrix-method Arith,ddiMatrix,logical-method
###   Arith,ddiMatrix,numeric-method Arith,logical,ddiMatrix-method
###   Arith,numeric,ddiMatrix-method Ops,ANY,ddiMatrix-method
###   Ops,ddiMatrix,ANY-method Ops,ddiMatrix,Matrix-method
###   Ops,ddiMatrix,dMatrix-method Ops,ddiMatrix,ddiMatrix-method
###   Ops,ddiMatrix,ldiMatrix-method Ops,ddiMatrix,ndiMatrix-method
###   Ops,ddiMatrix,logical-method Ops,ddiMatrix,numeric-method
###   Ops,ddiMatrix,sparseMatrix-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
(d2 <- Diagonal(x = c(10,1)))
str(d2)
## slightly larger in internal size:
str(as(d2, "sparseMatrix"))

M <- Matrix(cbind(1,2:4))
M %*% d2 #> `fast' multiplication

chol(d2) # trivial
stopifnot(is(cd2 <- chol(d2), "ddiMatrix"),
          all.equal(cd2@x, c(sqrt(10),1)))



cleanEx()
nameEx("denseMatrix-class")
### * denseMatrix-class

flush(stderr()); flush(stdout())

### Name: denseMatrix-class
### Title: Virtual Class "denseMatrix" of All Dense Matrices
### Aliases: denseMatrix-class -,denseMatrix,missing-method
###   Math,denseMatrix-method Summary,denseMatrix-method
###   coerce,ANY,denseMatrix-method coerce,matrix,denseMatrix-method
###   coerce,vector,denseMatrix-method diag,denseMatrix-method
###   diag<-,denseMatrix-method diff,denseMatrix-method
###   dim<-,denseMatrix-method log,denseMatrix-method
###   mean,denseMatrix-method rep,denseMatrix-method
###   show,denseMatrix-method t,denseMatrix-method
### Keywords: array classes

### ** Examples

showClass("denseMatrix")



cleanEx()
nameEx("dgCMatrix-class")
### * dgCMatrix-class

flush(stderr()); flush(stdout())

### Name: dgCMatrix-class
### Title: Compressed, sparse, column-oriented numeric matrices
### Aliases: dgCMatrix-class Arith,dgCMatrix,dgCMatrix-method
###   Arith,dgCMatrix,logical-method Arith,dgCMatrix,numeric-method
###   Arith,logical,dgCMatrix-method Arith,numeric,dgCMatrix-method
###   coerce,matrix,dgCMatrix-method determinant,dgCMatrix,logical-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
(m <- Matrix(c(0,0,2:0), 3,5))
str(m)
m[,1]
## Don't show: 
## regression test: this must give a validity-check error:
stopifnot(inherits(try(new("dgCMatrix", i = 0:1, p = 0:2,
                           x = c(2,3), Dim = 3:4)),
          "try-error"))
## End(Don't show)



cleanEx()
nameEx("dgTMatrix-class")
### * dgTMatrix-class

flush(stderr()); flush(stdout())

### Name: dgTMatrix-class
### Title: Sparse matrices in triplet form
### Aliases: dgTMatrix-class +,dgTMatrix,dgTMatrix-method
###   determinant,dgTMatrix,logical-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
m <- Matrix(0+1:28, nrow = 4)
m[-3,c(2,4:5,7)] <- m[ 3, 1:4] <- m[1:3, 6] <- 0
(mT <- as(m, "TsparseMatrix"))
str(mT)
mT[1,]
mT[4, drop = FALSE]
stopifnot(identical(mT[lower.tri(mT)],
                    m [lower.tri(m) ]))
mT[lower.tri(mT,diag=TRUE)] <- 0
mT

## Triplet representation with repeated (i,j) entries
## *adds* the corresponding x's:
T2 <- new("dgTMatrix",
          i = as.integer(c(1,1,0,3,3)),
          j = as.integer(c(2,2,4,0,0)), x=10*1:5, Dim=4:5)
str(T2) # contains (i,j,x) slots exactly as above, but
T2 ## has only three non-zero entries, as for repeated (i,j)'s,
   ## the corresponding x's are "implicitly" added
stopifnot(nnzero(T2) == 3)



cleanEx()
nameEx("diagU2N")
### * diagU2N

flush(stderr()); flush(stdout())

### Name: diagU2N
### Title: Transform Triangular Matrices from Unit Triangular to General
###   Triangular and Back
### Aliases: diagU2N diagN2U .diagU2N .diagN2U
### Keywords: array attribute utilities

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
(T <- Diagonal(7) + triu(Matrix(rpois(49, 1/4), 7, 7), k = 1))
(uT <- diagN2U(T)) # "unitriangular"
(t.u <- diagN2U(10*T))# changes the diagonal!
stopifnot(all(T == uT), diag(t.u) == 1,
          identical(T, diagU2N(uT)))
T[upper.tri(T)] <- 5 # still "dtC"
T <- diagN2U(as(T,"triangularMatrix"))
dT <- as(T, "denseMatrix") # (unitriangular)
dT.n <- diagU2N(dT, checkDense = TRUE)
sT.n <- diagU2N(dT)
stopifnot(is(dT.n, "denseMatrix"), is(sT.n, "sparseMatrix"),
          dT@diag == "U", dT.n@diag == "N", sT.n@diag == "N",
          all(dT == dT.n), all(dT == sT.n))



cleanEx()
nameEx("diagonalMatrix-class")
### * diagonalMatrix-class

flush(stderr()); flush(stdout())

### Name: diagonalMatrix-class
### Title: Class "diagonalMatrix" of Diagonal Matrices
### Aliases: diagonalMatrix-class -,diagonalMatrix,missing-method
###   Math,diagonalMatrix-method Ops,diagonalMatrix,triangularMatrix-method
###   Summary,diagonalMatrix-method
###   coerce,diagonalMatrix,symmetricMatrix-method
###   coerce,diagonalMatrix,triangularMatrix-method
###   coerce,matrix,diagonalMatrix-method
###   determinant,diagonalMatrix,logical-method diag,diagonalMatrix-method
###   diag<-,diagonalMatrix-method log,diagonalMatrix-method
###   print,diagonalMatrix-method show,diagonalMatrix-method
###   summary,diagonalMatrix-method t,diagonalMatrix-method
### Keywords: array classes

### ** Examples

I5 <- Diagonal(5)
D5 <- Diagonal(x = 10*(1:5))
## trivial (but explicitly defined) methods:
stopifnot(identical(crossprod(I5), I5),
          identical(tcrossprod(I5), I5),
          identical(crossprod(I5, D5), D5),
          identical(tcrossprod(D5, I5), D5),
          identical(solve(D5), solve(D5, I5)),
          all.equal(D5, solve(solve(D5)), tolerance = 1e-12)
          )
solve(D5)# efficient as is diagonal

# an unusual way to construct a band matrix:
rbind2(cbind2(I5, D5),
       cbind2(D5, I5))



cleanEx()
nameEx("dimScale")
### * dimScale

flush(stderr()); flush(stdout())

### Name: dimScale
### Title: Scale the Rows and Columns of a Matrix
### Aliases: dimScale rowScale colScale
### Keywords: algebra arith array utilities

### ** Examples

n <- 6L
(x <- forceSymmetric(matrix(1, n, n)))
dimnames(x) <- rep.int(list(letters[seq_len(n)]), 2L)

d <- seq_len(n)
(D <- Diagonal(x = d))

(scx <- dimScale(x, d)) # symmetry and 'dimnames' kept
(mmx <- D %*% x %*% D) # symmetry and 'dimnames' lost
stopifnot(identical(unname(as(scx, "generalMatrix")), mmx))

rowScale(x, d)
colScale(x, d)



cleanEx()
nameEx("dmperm")
### * dmperm

flush(stderr()); flush(stdout())

### Name: dmperm
### Title: Dulmage-Mendelsohn Permutation / Decomposition
### Aliases: dmperm
### Keywords: algebra array utilities

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
set.seed(17)
(S9 <- rsparsematrix(9, 9, nnz = 10, symmetric=TRUE)) # dsCMatrix
str( dm9 <- dmperm(S9) )
(S9p <- with(dm9, S9[p, q]))
## looks good, but *not* quite upper triangular; these, too:
str( dm9.0 <- dmperm(S9, seed=-1)) # non-random too.
str( dm9_1 <- dmperm(S9, seed= 1)) # a random one
## The last two permutations differ, but have the same effect!
(S9p0 <- with(dm9.0, S9[p, q])) # .. hmm ..
stopifnot(all.equal(S9p0, S9p))# same as as default, but different from the random one


set.seed(11)
(M <- triu(rsparsematrix(9,11, 1/4)))
dM <- dmperm(M); with(dM, M[p, q])
(Mp <- M[sample.int(nrow(M)), sample.int(ncol(M))])
dMp <- dmperm(Mp); with(dMp, Mp[p, q])


set.seed(7)
(n7 <- rsparsematrix(5, 12, nnz = 10, rand.x = NULL))
str( dm.7 <- dmperm(n7) )
stopifnot(exprs = {
  lengths(dm.7[1:2]) == dim(n7)
  identical(dm.7,      dmperm(as(n7, "dMatrix")))
  identical(dm.7[1:4], dmperm(n7, nAns=4))
  identical(dm.7[1:2], dmperm(n7, nAns=2))
})



cleanEx()
nameEx("dpoMatrix-class")
### * dpoMatrix-class

flush(stderr()); flush(stdout())

### Name: dpoMatrix-class
### Title: Positive Semi-definite Dense (Packed | Non-packed) Numeric
###   Matrices
### Aliases: dpoMatrix-class dppMatrix-class corMatrix-class
###   pcorMatrix-class Arith,dpoMatrix,logical-method
###   Arith,dpoMatrix,numeric-method Arith,logical,dpoMatrix-method
###   Arith,numeric,dpoMatrix-method Ops,dpoMatrix,logical-method
###   Ops,dpoMatrix,numeric-method Ops,logical,dpoMatrix-method
###   Ops,numeric,dpoMatrix-method coerce,dpoMatrix,corMatrix-method
###   coerce,dpoMatrix,dppMatrix-method coerce,matrix,dpoMatrix-method
###   determinant,dpoMatrix,logical-method Arith,dppMatrix,logical-method
###   Arith,dppMatrix,numeric-method Arith,logical,dppMatrix-method
###   Arith,numeric,dppMatrix-method Ops,dppMatrix,logical-method
###   Ops,dppMatrix,numeric-method Ops,logical,dppMatrix-method
###   Ops,numeric,dppMatrix-method coerce,dppMatrix,dpoMatrix-method
###   coerce,dppMatrix,pcorMatrix-method coerce,matrix,dppMatrix-method
###   determinant,dppMatrix,logical-method
###   coerce,corMatrix,pcorMatrix-method coerce,matrix,corMatrix-method
###   coerce,pcorMatrix,corMatrix-method coerce,matrix,pcorMatrix-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
h6 <- Hilbert(6)
rcond(h6)
str(h6)
h6 * 27720 # is ``integer''
solve(h6)
str(hp6 <- pack(h6))

### Note that  as(*, "corMatrix")  *scales* the matrix
(ch6 <- as(h6, "corMatrix"))
stopifnot(all.equal(as(h6 * 27720, "dsyMatrix"), round(27720 * h6),
                    tolerance = 1e-14),
          all.equal(ch6@sd^(-2), 2*(1:6)-1,
                    tolerance = 1e-12))
chch <- Cholesky(ch6, perm = FALSE)
stopifnot(identical(chch, ch6@factors$Cholesky),
          all(abs(crossprod(as(chch, "dtrMatrix")) - ch6) < 1e-10))



cleanEx()
nameEx("drop0")
### * drop0

flush(stderr()); flush(stdout())

### Name: drop0
### Title: Drop Non-Structural Zeros from a Sparse Matrix
### Aliases: drop0
### Keywords: array manip utilities

### ** Examples

(m <- sparseMatrix(i = 1:8, j = 2:9, x = c(0:2, 3:-1),
                   dims = c(10L, 20L)))
drop0(m)

## A larger example:
t5 <- new("dtCMatrix", Dim = c(5L, 5L), uplo = "L",
          x = c(10, 1, 3, 10, 1, 10, 1, 10, 10),
          i = c(0L,2L,4L, 1L, 3L,2L,4L, 3L, 4L),
          p = c(0L, 3L, 5L, 7:9))
TT <- kronecker(t5, kronecker(kronecker(t5, t5), t5))
IT <- solve(TT)
I. <- TT %*% IT ;  nnzero(I.) # 697 ( == 625 + 72 )
I.0 <- drop0(zapsmall(I.))
## which actually can be more efficiently achieved by
I.. <- drop0(I., tol = 1e-15)
stopifnot(all(I.0 == Diagonal(625)), nnzero(I..) == 625)



cleanEx()
nameEx("dsCMatrix-class")
### * dsCMatrix-class

flush(stderr()); flush(stdout())

### Name: dsCMatrix-class
### Title: Numeric Symmetric Sparse (column compressed) Matrices
### Aliases: dsCMatrix-class dsTMatrix-class
###   Arith,dsCMatrix,dsCMatrix-method determinant,dsCMatrix,logical-method
###   determinant,dsTMatrix,logical-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
mm <- Matrix(toeplitz(c(10, 0, 1, 0, 3)), sparse = TRUE)
mm # automatically dsCMatrix
str(mm)
mT <- as(as(mm, "generalMatrix"), "TsparseMatrix")

## Either
(symM <- as(mT, "symmetricMatrix")) # dsT
(symC <- as(symM, "CsparseMatrix")) # dsC
## or
sT <- Matrix(mT, sparse=TRUE, forceCheck=TRUE) # dsT

sym2 <- as(symC, "TsparseMatrix")
## --> the same as 'symM', a "dsTMatrix"
## Don't show: 
stopifnot(identical(sT, symM), identical(sym2, symM),
          class(sym2) == "dsTMatrix",
	  identical(sym2[1,], sT[1,]),
	  identical(sym2[,2], sT[,2]))
## End(Don't show)



cleanEx()
nameEx("dsRMatrix-class")
### * dsRMatrix-class

flush(stderr()); flush(stdout())

### Name: dsRMatrix-class
### Title: Symmetric Sparse Compressed Row Matrices
### Aliases: dsRMatrix-class determinant,dsRMatrix,logical-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
(m0 <- new("dsRMatrix"))
m2 <- new("dsRMatrix", Dim = c(2L,2L),
          x = c(3,1), j = c(1L,1L), p = 0:2)
m2
stopifnot(colSums(as(m2, "TsparseMatrix")) == 3:4)
str(m2)
(ds2 <- forceSymmetric(diag(2))) # dsy*
dR <- as(ds2, "RsparseMatrix")
dR # dsRMatrix



cleanEx()
nameEx("dsparseMatrix-class")
### * dsparseMatrix-class

flush(stderr()); flush(stdout())

### Name: dsparseMatrix-class
### Title: Virtual Class "dsparseMatrix" of Numeric Sparse Matrices
### Aliases: dsparseMatrix-class Arith,dsparseMatrix,logical-method
###   Arith,dsparseMatrix,numeric-method Arith,logical,dsparseMatrix-method
###   Arith,numeric,dsparseMatrix-method
###   Ops,dsparseMatrix,nsparseMatrix-method
###   coerce,matrix,dsparseMatrix-method coerce,vector,dsparseMatrix-method
### Keywords: array classes

### ** Examples

showClass("dsparseMatrix")



cleanEx()
nameEx("dsyMatrix-class")
### * dsyMatrix-class

flush(stderr()); flush(stdout())

### Name: dsyMatrix-class
### Title: Symmetric Dense (Packed or Unpacked) Numeric Matrices
### Aliases: dsyMatrix-class dspMatrix-class
###   coerce,dsyMatrix,corMatrix-method coerce,dsyMatrix,dpoMatrix-method
###   determinant,dsyMatrix,logical-method
###   coerce,dspMatrix,dppMatrix-method coerce,dspMatrix,pcorMatrix-method
###   determinant,dspMatrix,logical-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
## Only upper triangular part matters (when uplo == "U" as per default)
(sy2 <- new("dsyMatrix", Dim = as.integer(c(2,2)), x = c(14, NA,32,77)))
str(t(sy2)) # uplo = "L", and the lower tri. (i.e. NA is replaced).

chol(sy2) #-> "Cholesky" matrix
(sp2 <- pack(sy2)) # a "dspMatrix"

## Coercing to dpoMatrix gives invalid object:
sy3 <- new("dsyMatrix", Dim = as.integer(c(2,2)), x = c(14, -1, 2, -7))
try(as(sy3, "dpoMatrix")) # -> error: not positive definite
## Don't show: 
tr <- try(as(sy3, "dpoMatrix"), silent=TRUE)
stopifnot(1 == grep("not a positive definite matrix",
                    as.character(tr)),
	  is(sp2, "dspMatrix"))
## End(Don't show)

## 4x4 example
m <- matrix(0,4,4); m[upper.tri(m)] <- 1:6
(sym <- m+t(m)+diag(11:14, 4))
(S1 <- pack(sym))
(S2 <- t(S1))
stopifnot(all(S1 == S2)) # equal "seen as matrix", but differ internally :
str(S1)
S2@x



cleanEx()
nameEx("dtCMatrix-class")
### * dtCMatrix-class

flush(stderr()); flush(stdout())

### Name: dtCMatrix-class
### Title: Triangular, (compressed) sparse column matrices
### Aliases: dtCMatrix-class dtTMatrix-class
###   Arith,dtCMatrix,dtCMatrix-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("dtCMatrix")
showClass("dtTMatrix")
t1 <- new("dtTMatrix", x= c(3,7), i= 0:1, j=3:2, Dim= as.integer(c(4,4)))
t1
## from  0-diagonal to unit-diagonal {low-level step}:
tu <- t1 ; tu@diag <- "U"
tu
(cu <- as(tu, "CsparseMatrix"))
str(cu)# only two entries in @i and @x
stopifnot(cu@i == 1:0,
          all(2 * symmpart(cu) == Diagonal(4) + forceSymmetric(cu)))

t1[1,2:3] <- -1:-2
diag(t1) <- 10*c(1:2,3:2)
t1 # still triangular
(it1 <- solve(t1))
t1. <- solve(it1)
all(abs(t1 - t1.) < 10 * .Machine$double.eps)

## 2nd example
U5 <- new("dtCMatrix", i= c(1L, 0:3), p=c(0L,0L,0:2, 5L), Dim = c(5L, 5L),
          x = rep(1, 5), diag = "U")
U5
(iu <- solve(U5)) # contains one '0'
validObject(iu2 <- solve(U5, Diagonal(5)))# failed in earlier versions

I5 <- iu  %*% U5 # should equal the identity matrix
i5 <- iu2 %*% U5
m53 <- matrix(1:15, 5,3, dimnames=list(NULL,letters[1:3]))
asDiag <- function(M) as(drop0(M), "diagonalMatrix")
stopifnot(
   all.equal(Diagonal(5), asDiag(I5), tolerance=1e-14) ,
   all.equal(Diagonal(5), asDiag(i5), tolerance=1e-14) ,
   identical(list(NULL, dimnames(m53)[[2]]), dimnames(solve(U5, m53)))
)
## Don't show: 
i5. <- I5; colnames(i5.) <- LETTERS[11:15]
M53 <- as(m53, "denseMatrix")
stopifnot(
   identical((dns <- dimnames(solve(i5., M53))),
             dimnames(solve(as.matrix(i5.), as.matrix(M53)))) ,
   identical(dns, dimnames(solve(i5., as.matrix(M53))))
)
## End(Don't show)



cleanEx()
nameEx("dtRMatrix-class-def")
### * dtRMatrix-class-def

flush(stderr()); flush(stdout())

### Name: dtRMatrix-class
### Title: Triangular Sparse Compressed Row Matrices
### Aliases: dtRMatrix-class
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
(m0 <- new("dtRMatrix"))
(m2 <- new("dtRMatrix", Dim = c(2L,2L),
                        x = c(5, 1:2), p = c(0L,2:3), j= c(0:1,1L)))
str(m2)
(m3 <- as(Diagonal(2), "RsparseMatrix"))# --> dtRMatrix



cleanEx()
nameEx("dtpMatrix-class")
### * dtpMatrix-class

flush(stderr()); flush(stdout())

### Name: dtpMatrix-class
### Title: Packed Triangular Dense Matrices - "dtpMatrix"
### Aliases: dtpMatrix-class
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("dtrMatrix")

example("dtrMatrix-class", echo=FALSE)
(p1 <- pack(T2))
str(p1)
(pp <- pack(T))
ip1 <- solve(p1)
stopifnot(length(p1@x) == 3, length(pp@x) == 3,
          p1 @ uplo == T2 @ uplo, pp @ uplo == T @ uplo,
	  identical(t(pp), p1), identical(t(p1), pp),
	  all((l.d <- p1 - T2) == 0), is(l.d, "dtpMatrix"),
	  all((u.d <- pp - T ) == 0), is(u.d, "dtpMatrix"),
	  l.d@uplo == T2@uplo, u.d@uplo == T@uplo,
	  identical(t(ip1), solve(pp)), is(ip1, "dtpMatrix"),
	  all.equal(as(solve(p1,p1), "diagonalMatrix"), Diagonal(2)))



cleanEx()
nameEx("dtrMatrix-class")
### * dtrMatrix-class

flush(stderr()); flush(stdout())

### Name: dtrMatrix-class
### Title: Triangular, dense, numeric matrices
### Aliases: dtrMatrix-class
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
(m <- rbind(2:3, 0:-1))
(M <- as(m, "generalMatrix"))

(T <- as(M, "triangularMatrix")) # formally upper triangular
(T2 <- as(t(M), "triangularMatrix"))
stopifnot(T@uplo == "U", T2@uplo == "L", identical(T2, t(T)))

m <- matrix(0,4,4); m[upper.tri(m)] <- 1:6
(t1 <- Matrix(m+diag(,4)))
str(t1p <- pack(t1))
(t1pu <- diagN2U(t1p))
stopifnot(exprs = {
   inherits(t1 , "dtrMatrix"); validObject(t1)
   inherits(t1p, "dtpMatrix"); validObject(t1p)
   inherits(t1pu,"dtCMatrix"); validObject(t1pu)
   t1pu@x == 1:6
   all(t1pu == t1p)
   identical((t1pu - t1)@x, numeric())# sparse all-0
})



cleanEx()
nameEx("expand")
### * expand

flush(stderr()); flush(stdout())

### Name: expand-methods
### Title: Expand Matrix Factorizations
### Aliases: expand expand-methods expand1 expand1-methods expand2
###   expand2-methods expand,CHMfactor-method expand,denseLU-method
###   expand,sparseLU-method expand1,BunchKaufman-method
###   expand1,CHMsimpl-method expand1,CHMsuper-method
###   expand1,Cholesky-method expand1,Schur-method expand1,denseLU-method
###   expand1,pBunchKaufman-method expand1,pCholesky-method
###   expand1,sparseLU-method expand1,sparseQR-method
###   expand2,BunchKaufman-method expand2,CHMsimpl-method
###   expand2,CHMsuper-method expand2,Cholesky-method expand2,Schur-method
###   expand2,denseLU-method expand2,pBunchKaufman-method
###   expand2,pCholesky-method expand2,sparseLU-method
###   expand2,sparseQR-method
### Keywords: algebra array methods

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
showMethods("expand1", inherited = FALSE)
showMethods("expand2", inherited = FALSE)
set.seed(0)

(A <- Matrix(rnorm(9L, 0, 10), 3L, 3L))
(lu.A <- lu(A))
(e.lu.A <- expand2(lu.A))
stopifnot(exprs = {
    is.list(e.lu.A)
    identical(names(e.lu.A), c("P1.", "L", "U"))
    all(sapply(e.lu.A, is, "Matrix"))
    all.equal(as(A, "matrix"), as(Reduce(`%*%`, e.lu.A), "matrix"))
})

## 'expand1' and 'expand2' give equivalent results modulo
## dimnames and representation of permutation matrices;
## see also function 'alt' in example("Cholesky-methods")
(a1 <- sapply(names(e.lu.A), expand1, x = lu.A, simplify = FALSE))
all.equal(a1, e.lu.A)

## see help("denseLU-class") and others for more examples



cleanEx()
nameEx("expm")
### * expm

flush(stderr()); flush(stdout())

### Name: expm-methods
### Title: Matrix Exponential
### Aliases: expm expm-methods expm,Matrix-method expm,dMatrix-method
###   expm,ddiMatrix-method expm,dgeMatrix-method expm,dspMatrix-method
###   expm,dsparseMatrix-method expm,dsyMatrix-method expm,dtpMatrix-method
###   expm,dtrMatrix-method expm,matrix-method
### Keywords: array math methods

### ** Examples

(m1 <- Matrix(c(1,0,1,1), ncol = 2))
(e1 <- expm(m1)) ; e <- exp(1)
stopifnot(all.equal(e1@x, c(e,0,e,e), tolerance = 1e-15))
(m2 <- Matrix(c(-49, -64, 24, 31), ncol = 2))
(e2 <- expm(m2))
(m3 <- Matrix(cbind(0,rbind(6*diag(3),0))))# sparse!
(e3 <- expm(m3)) # upper triangular



cleanEx()
nameEx("externalFormats")
### * externalFormats

flush(stderr()); flush(stdout())

### Name: externalFormats
### Title: Read and write external matrix formats
### Aliases: readHB readMM writeMM writeMM,CsparseMatrix-method
###   writeMM,sparseMatrix-method
### Keywords: connection file methods utilities

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
str(pores <- readMM(system.file("external/pores_1.mtx", package = "Matrix")))
str(utm   <- readHB(system.file("external/utm300.rua" , package = "Matrix")))
str(lundA <- readMM(system.file("external/lund_a.mtx" , package = "Matrix")))
str(lundA <- readHB(system.file("external/lund_a.rsa" , package = "Matrix")))
## https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/counterx/counterx.htm
str(jgl   <- readMM(system.file("external/jgl009.mtx" , package = "Matrix")))

## NOTE: The following examples take quite some time
## ----  even on a fast internet connection:
if(FALSE) {
## The URL has been corrected, but we need an untar step:
u. <- url("https://www.cise.ufl.edu/research/sparse/RB/Boeing/msc00726.tar.gz")
str(sm <- readHB(gzcon(u.)))
}

data(KNex, package = "Matrix")
## Store as MatrixMarket (".mtx") file, here inside temporary dir./folder:
(MMfile <- file.path(tempdir(), "mmMM.mtx"))
writeMM(KNex$mm, file=MMfile)
file.info(MMfile)[,c("size", "ctime")] # (some confirmation of the file's)

## very simple export - in triplet format - to text file:
data(CAex, package = "Matrix")
s.CA <- summary(CAex)
s.CA # shows  (i, j, x)  [columns of a data frame]
message("writing to ", outf <- tempfile())
write.table(s.CA, file = outf, row.names=FALSE)
## and read it back -- showing off  sparseMatrix():
str(dd <- read.table(outf, header=TRUE))
## has columns (i, j, x) -> we can use via do.call() as arguments to sparseMatrix():
mm <- do.call(sparseMatrix, dd)
stopifnot(all.equal(mm, CAex, tolerance=1e-15))



cleanEx()
nameEx("facmul")
### * facmul

flush(stderr()); flush(stdout())

### Name: facmul-methods
### Title: Multiplication by Factors from Matrix Factorizations
### Aliases: facmul facmul-methods
### Keywords: arith array methods

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
## Conceptually, methods for 'facmul' _would_ behave as follows ...
## Not run: 
##D n <- 3L
##D x <- lu(Matrix(rnorm(n * n), n, n))
##D y <- rnorm(n)
##D L <- unname(expand2(x)[[nm <- "L"]])
##D stopifnot(exprs = {
##D     all.equal(facmul(x, nm, y, trans = FALSE, left =  TRUE), L %*% y)
##D     all.equal(facmul(x, nm, y, trans = FALSE, left = FALSE), y %*% L)
##D     all.equal(facmul(x, nm, y, trans =  TRUE, left =  TRUE),  crossprod(L, y))
##D     all.equal(facmul(x, nm, y, trans =  TRUE, left = FALSE), tcrossprod(y, L))
##D })
## End(Not run)



cleanEx()
nameEx("fastMisc")
### * fastMisc

flush(stderr()); flush(stdout())

### Name: fastMisc
### Title: "Low Level" Coercions and Methods
### Aliases: fastMisc .M2kind .M2gen .M2sym .M2tri .M2diag .M2v .M2m
###   .M2unpacked .M2packed .M2C .M2R .M2T .M2V .m2V .sparse2dense
###   .diag2dense .ind2dense .m2dense .dense2sparse .diag2sparse
###   .ind2sparse .m2sparse .tCRT .CR2RC .CR2T .T2CR .dense2g .dense2kind
###   .dense2m .dense2v .sparse2g .sparse2kind .sparse2m .sparse2v .tCR2RC
###   .diag.dsC .solve.dgC.lu .solve.dgC.qr .solve.dgC.chol
###   .updateCHMfactor
### Keywords: utilities

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
D. <- diag(x = c(1, 1, 2, 3, 5, 8))
D.0 <- Diagonal(x = c(0, 0, 0, 3, 5, 8))
S. <- toeplitz(as.double(1:6))
C. <- new("dgCMatrix", Dim = c(3L, 4L),
          p = c(0L, 1L, 1L, 1L, 3L), i = c(1L, 0L, 2L), x = c(-8, 2, 3))

stopifnot(exprs = {
    identical(.M2tri (D.), as(D., "triangularMatrix"))
    identical(.M2sym (D.), as(D.,  "symmetricMatrix"))
    identical(.M2diag(D.), as(D.,   "diagonalMatrix"))
    identical(.M2kind(C., "l"),
              as(C., "lMatrix"))
    identical(.M2kind(.sparse2dense(C.), "l"),
              as(as(C., "denseMatrix"), "lMatrix"))
    identical(.diag2sparse(D.0, ".", "t", "C"),
              .dense2sparse(.diag2dense(D.0, ".", "t", TRUE), "C"))
    identical(.M2gen(.diag2dense(D.0, ".", "s", FALSE)),
              .sparse2dense(.M2gen(.diag2sparse(D.0, ".", "s", "T"))))
    identical(S.,
              .M2m(.m2sparse(S., ".sR")))
    identical(S. * lower.tri(S.) + diag(1, 6L),
              .M2m(.m2dense (S., ".tr", "L", "U")))
    identical(.M2R(C.), .M2R(.M2T(C.)))
    identical(.tCRT(C.), .M2R(t(C.)))
})

A <- tcrossprod(C.)/6 + Diagonal(3, 1/3); A[1,2] <- 3; A
stopifnot(exprs = {
    is.numeric( x. <- c(2.2, 0, -1.2) )
    all.equal(x., .solve.dgC.lu(A, c(1,0,0), check=FALSE))
    all.equal(x., .solve.dgC.qr(A, c(1,0,0), check=FALSE))
})

## Solving sparse least squares:

X <- rbind(A, Diagonal(3)) # design matrix X (for L.S.)
Xt <- t(X)                 # *transposed*  X (for L.S.)
(y <- drop(crossprod(Xt, 1:3)) + c(-1,1)/1000) # small rand.err.
str(solveCh <- .solve.dgC.chol(Xt, y, check=FALSE)) # Xt *is* dgC..
stopifnot(exprs = {
    all.equal(solveCh$coef, 1:3, tol = 1e-3)# rel.err ~ 1e-4
    all.equal(solveCh$coef, drop(solve(tcrossprod(Xt), Xt %*% y)))
    all.equal(solveCh$coef, .solve.dgC.qr(X, y, check=FALSE))
})



cleanEx()
nameEx("forceSymmetric")
### * forceSymmetric

flush(stderr()); flush(stdout())

### Name: forceSymmetric-methods
### Title: Force a Matrix to 'symmetricMatrix' Without Symmetry Checks
### Aliases: forceSymmetric forceSymmetric-methods
###   forceSymmetric,CsparseMatrix,character-method
###   forceSymmetric,CsparseMatrix,missing-method
###   forceSymmetric,RsparseMatrix,character-method
###   forceSymmetric,RsparseMatrix,missing-method
###   forceSymmetric,TsparseMatrix,character-method
###   forceSymmetric,TsparseMatrix,missing-method
###   forceSymmetric,denseMatrix,character-method
###   forceSymmetric,denseMatrix,missing-method
###   forceSymmetric,diagonalMatrix,character-method
###   forceSymmetric,diagonalMatrix,missing-method
###   forceSymmetric,indMatrix,character-method
###   forceSymmetric,indMatrix,missing-method
###   forceSymmetric,matrix,character-method
###   forceSymmetric,matrix,missing-method
### Keywords: array methods

### ** Examples

 ## Hilbert matrix
 i <- 1:6
 h6 <- 1/outer(i - 1L, i, "+")
 sd <- sqrt(diag(h6))
 hh <- t(h6/sd)/sd # theoretically symmetric
 isSymmetric(hh, tol=0) # FALSE; hence
 try( as(hh, "symmetricMatrix") ) # fails, but this works fine:
 H6 <- forceSymmetric(hh)

 ## result can be pretty surprising:
 (M <- Matrix(1:36, 6))
 forceSymmetric(M) # symmetric, hence very different in lower triangle
 (tm <- tril(M))
 forceSymmetric(tm)



cleanEx()
nameEx("formatSparseM")
### * formatSparseM

flush(stderr()); flush(stdout())

### Name: formatSparseM
### Title: Formatting Sparse Numeric Matrices Utilities
### Aliases: formatSparseM .formatSparseSimple
### Keywords: character print utilities

### ** Examples

m <- suppressWarnings(matrix(c(0, 3.2, 0,0, 11,0,0,0,0,-7,0), 4,9))
fm <- formatSparseM(m)
noquote(fm)
## nice, but this is nicer {with "units" vertically aligned}:
print(fm, quote=FALSE, right=TRUE)
## and "the same" as :
Matrix(m)

## align = "right" is cheaper -->  the "." are not aligned:
noquote(f2 <- formatSparseM(m,align="r"))
stopifnot(f2 == fm   |   m == 0, dim(f2) == dim(m),
         (f2 == ".") == (m == 0))



cleanEx()
nameEx("graph2T")
### * graph2T

flush(stderr()); flush(stdout())

### Name: coerce-methods-graph
### Title: Conversions "graph" <-> (sparse) Matrix
### Aliases: coerce-methods-graph coerce,Matrix,graph-method
###   coerce,Matrix,graphNEL-method coerce,TsparseMatrix,graphNEL-method
###   coerce,graph,CsparseMatrix-method coerce,graph,Matrix-method
###   coerce,graph,RsparseMatrix-method coerce,graph,TsparseMatrix-method
###   coerce,graph,sparseMatrix-method coerce,graphAM,TsparseMatrix-method
###   coerce,graphNEL,TsparseMatrix-method T2graph graph2T
### Keywords: methods utilities

### ** Examples

if(requireNamespace("graph")) {
  n4 <- LETTERS[1:4]; dns <- list(n4,n4)
  show(a1 <- sparseMatrix(i= c(1:4),   j=c(2:4,1),   x = 2,    dimnames=dns))
  show(g1 <- as(a1, "graph")) # directed
  unlist(graph::edgeWeights(g1)) # all '2'

  show(a2 <- sparseMatrix(i= c(1:4,4), j=c(2:4,1:2), x = TRUE, dimnames=dns))
  show(g2 <- as(a2, "graph")) # directed
  # now if you want it undirected:
  show(g3  <- T2graph(as(a2,"TsparseMatrix"), edgemode="undirected"))
  show(m3 <- as(g3,"Matrix"))
  show( graph2T(g3) ) # a "pattern Matrix" (nsTMatrix)
## Don't show: 
  stopifnot(
   identical(as(g3,"Matrix"), as(as(a2 + t(a2), "nMatrix"),"symmetricMatrix"))
  ,
   identical(tg3 <- graph2T(g3), graph2T(g3, use.weights=FALSE))
  ,
   identical(as(m3,"TsparseMatrix"), asUniqueT(tg3))
  )
## End(Don't show)
  a. <- sparseMatrix(i=4:1, j=1:4, dimnames=list(n4, n4), repr="T") # no 'x'
  show(a.) # "ngTMatrix"
  show(g. <- as(a., "graph"))
## Don't show: 
  stopifnot(graph::edgemode(g.) == "undirected",
            graph::numEdges(g.) == 2,
            all.equal(as(g., "TsparseMatrix"),
                      as(a., "symmetricMatrix"))
)
## End(Don't show)
}



cleanEx()
nameEx("image-methods")
### * image-methods

flush(stderr()); flush(stdout())

### Name: image-methods
### Title: Methods for image() in Package 'Matrix'
### Aliases: image image-methods image,ANY-method image,CHMfactor-method
###   image,Matrix-method image,dgTMatrix-method
### Keywords: hplot methods

### ** Examples

## Don't show: 
 
library(grDevices, pos = "package:base", verbose = FALSE)
library(    utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showMethods(image)
## And if you want to see the method definitions:
showMethods(image, includeDefs = TRUE, inherited = FALSE)
## Don't show: 
 
op <- options(warn = 2)
## End(Don't show)
data(CAex, package = "Matrix")
image(CAex, main = "image(CAex)") -> imgC; imgC
stopifnot(!is.null(leg <- imgC$legend), is.list(leg$right)) # failed for 2 days ..
image(CAex, useAbs=TRUE, main = "image(CAex, useAbs=TRUE)")

cCA <- Cholesky(crossprod(CAex), Imult = .01)
## See  ?print.trellis --- place two image() plots side by side:
print(image(cCA, main="Cholesky(crossprod(CAex), Imult = .01)"),
      split=c(x=1,y=1,nx=2, ny=1), more=TRUE)
print(image(cCA, useAbs=TRUE),
      split=c(x=2,y=1,nx=2,ny=1))

data(USCounties, package = "Matrix")
image(USCounties)# huge
image(sign(USCounties))## just the pattern
    # how the result looks, may depend heavily on
    # the device, screen resolution, antialiasing etc
    # e.g. x11(type="Xlib") may show very differently than cairo-based

## Drawing borders around each rectangle;
    # again, viewing depends very much on the device:
image(USCounties[1:400,1:200], lwd=.1)
## Using (xlim,ylim) has advantage : matrix dimension and (col/row) indices:
image(USCounties, c(1,200), c(1,400), lwd=.1)
image(USCounties, c(1,300), c(1,200), lwd=.5 )
image(USCounties, c(1,300), c(1,200), lwd=.01)
## These 3 are all equivalent :
(I1 <- image(USCounties, c(1,100), c(1,100), useAbs=FALSE))
 I2 <- image(USCounties, c(1,100), c(1,100), useAbs=FALSE,        border.col=NA)
 I3 <- image(USCounties, c(1,100), c(1,100), useAbs=FALSE, lwd=2, border.col=NA)
stopifnot(all.equal(I1, I2, check.environment=FALSE),
          all.equal(I2, I3, check.environment=FALSE))
## using an opaque border color
image(USCounties, c(1,100), c(1,100), useAbs=FALSE, lwd=3, border.col = adjustcolor("skyblue", 1/2))
## Don't show: 
options(op)
## End(Don't show)
if(interactive() || nzchar(Sys.getenv("R_MATRIX_CHECK_EXTRA"))) {
## Using raster graphics: For PDF this would give a 77 MB file,
## however, for such a large matrix, this is typically considerably
## *slower* (than vector graphics rectangles) in most cases :
if(doPNG <- !dev.interactive())
   png("image-USCounties-raster.png", width=3200, height=3200)
image(USCounties, useRaster = TRUE) # should not suffer from anti-aliasing
if(doPNG)
   dev.off()
   ## and now look at the *.png image in a viewer you can easily zoom in and out
}#only if(doExtras)



cleanEx()
nameEx("indMatrix-class")
### * indMatrix-class

flush(stderr()); flush(stdout())

### Name: indMatrix-class
### Title: Index Matrices
### Aliases: indMatrix-class !,indMatrix-method -,indMatrix,missing-method
###   Math,indMatrix-method Summary,indMatrix-method
###   coerce,indMatrix,pMatrix-method coerce,list,indMatrix-method
###   coerce,matrix,indMatrix-method coerce,numeric,indMatrix-method
###   determinant,indMatrix,logical-method diag,indMatrix-method
###   diag<-,indMatrix-method log,indMatrix-method t,indMatrix-method
###   which,indMatrix-method
### Keywords: array classes

### ** Examples

p1 <- as(c(2,3,1), "pMatrix")
(sm1 <- as(rep(c(2,3,1), e=3), "indMatrix"))
stopifnot(all(sm1 == p1[rep(1:3, each=3),]))

## row-indexing of a <pMatrix> turns it into an <indMatrix>:
class(p1[rep(1:3, each=3),])

set.seed(12) # so we know '10' is in sample
## random index matrix for 30 observations and 10 unique values:
(s10 <- as(sample(10, 30, replace=TRUE),"indMatrix"))

## Sample rows of a numeric matrix :
(mm <- matrix(1:10, nrow=10, ncol=3))
s10 %*% mm

set.seed(27)
IM1 <- as(sample(1:20, 100, replace=TRUE), "indMatrix")
IM2 <- as(sample(1:18, 100, replace=TRUE), "indMatrix")
(c12 <- crossprod(IM1,IM2))
## same as cross-tabulation of the two index vectors:
stopifnot(all(c12 - unclass(table(IM1@perm, IM2@perm)) == 0))

# 3 observations, 4 implied values, first does not occur in sample:
as(2:4, "indMatrix")
# 3 observations, 5 values, first and last do not occur in sample:
as(list(2:4, 5), "indMatrix")

as(sm1, "nMatrix")
s10[1:7, 1:4] # gives an "ngTMatrix" (most economic!)
s10[1:4, ]  # preserves "indMatrix"-class

I1 <- as(c(5:1,6:4,7:3), "indMatrix")
I2 <- as(7:1, "pMatrix")
(I12 <- rbind(I1, I2))
stopifnot(is(I12, "indMatrix"),
          identical(I12, rbind(I1, I2)),
	  colSums(I12) == c(2L,2:4,4:2))



cleanEx()
nameEx("index-class")
### * index-class

flush(stderr()); flush(stdout())

### Name: index-class
### Title: Virtual Class "index" - Simple Class for Matrix Indices
### Aliases: index-class
### Keywords: classes

### ** Examples

showClass("index")



cleanEx()
nameEx("invPerm")
### * invPerm

flush(stderr()); flush(stdout())

### Name: invertPerm
### Title: Utilities for Permutation Vectors
### Aliases: invertPerm signPerm isPerm asPerm invPerm
### Keywords: utilities

### ** Examples

p <- sample(10L) # a random permutation vector
ip <- invertPerm(p)
s <- signPerm(p)

## 'p' and 'ip' are indeed inverses:
stopifnot(exprs = {
    isPerm(p)
    isPerm(ip)
    identical(s, 1L) || identical(s, -1L)
    identical(s, signPerm(ip))
    identical(p[ip], 1:10)
    identical(ip[p], 1:10)
    identical(invertPerm(ip), p)
})

## Product of transpositions (1 2)(2 1)(4 3)(6 8)(10 1) = (3 4)(6 8)(1 10)
pivot <- c(2L, 1L, 3L, 3L, 5L, 8L, 7L, 8L, 9L, 1L)
q <- asPerm(pivot)
stopifnot(exprs = {
    identical(q, c(10L, 2L, 4L, 3L, 5L, 8L, 7L, 6L, 9L, 1L))
    identical(q[q], seq_len(10L)) # because the permutation is odd:
    signPerm(q) == -1L
})

invPerm # a less general version of 'invertPerm'
## Don't show: 
stopifnot(exprs = {
    identical(isPerm(0L), FALSE)
    identical(signPerm(1:2),  1L)
    identical(signPerm(2:1), -1L)
    identical(invertPerm(c(3, 1:2)), c(2:3, 1L)) # 'p' of type "double",
    tryCatch(invPerm(NA), error = function(e) TRUE) # was a segfault
})
## End(Don't show)



cleanEx()
nameEx("is.na-methods")
### * is.na-methods

flush(stderr()); flush(stdout())

### Name: is.na-methods
### Title: is.na(), is.finite() Methods for 'Matrix' Objects
### Aliases: anyNA anyNA-methods is.na is.na-methods is.nan is.nan-methods
###   is.infinite is.infinite-methods is.finite is.finite-methods
###   anyNA,denseMatrix-method anyNA,diagonalMatrix-method
###   anyNA,indMatrix-method anyNA,sparseMatrix-method
###   anyNA,sparseVector-method is.na,abIndex-method
###   is.na,denseMatrix-method is.na,diagonalMatrix-method
###   is.na,indMatrix-method is.na,sparseMatrix-method
###   is.na,sparseVector-method is.nan,denseMatrix-method
###   is.nan,diagonalMatrix-method is.nan,indMatrix-method
###   is.nan,sparseMatrix-method is.nan,sparseVector-method
###   is.infinite,abIndex-method is.infinite,denseMatrix-method
###   is.infinite,diagonalMatrix-method is.infinite,indMatrix-method
###   is.infinite,sparseMatrix-method is.infinite,sparseVector-method
###   is.finite,abIndex-method is.finite,denseMatrix-method
###   is.finite,diagonalMatrix-method is.finite,indMatrix-method
###   is.finite,sparseMatrix-method is.finite,sparseVector-method
### Keywords: NA math programming methods

### ** Examples

(M <- Matrix(1:6, nrow = 4, ncol = 3,
             dimnames = list(letters[1:4], LETTERS[1:3])))
stopifnot(!anyNA(M), !any(is.na(M)))

M[2:3, 2] <- NA
(inM <- is.na(M))
stopifnot(anyNA(M), sum(inM) == 2)

(A <- spMatrix(nrow = 10, ncol = 20,
               i = c(1, 3:8), j = c(2, 9, 6:10), x = 7 * (1:7)))
stopifnot(!anyNA(A), !any(is.na(A)))

A[2, 3] <- A[1, 2] <- A[5, 5:9] <- NA
(inA <- is.na(A))
stopifnot(anyNA(A), sum(inA) == 1 + 1 + 5)



cleanEx()
nameEx("is.null.DN")
### * is.null.DN

flush(stderr()); flush(stdout())

### Name: is.null.DN
### Title: Are the Dimnames 'dn' NULL-like ?
### Aliases: is.null.DN
### Keywords: array attribute programming utilities

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
m1 <- m2 <- m3 <- m4 <- m <-
    matrix(round(100 * rnorm(6)), 2, 3)
dimnames(m1) <- list(NULL, NULL)
dimnames(m2) <- list(NULL, character())
dimnames(m3) <- rev(dimnames(m2))
dimnames(m4) <- rep(list(character()),2)

m4 # prints absolutely identically to m

c.o <- capture.output
cm <- c.o(m)
stopifnot(exprs = {
    m == m1; m == m2; m == m3; m == m4
	identical(cm, c.o(m1));	identical(cm, c.o(m2))
	identical(cm, c.o(m3)); identical(cm, c.o(m4))
})

hasNoDimnames <- function(.) is.null.DN(dimnames(.))
stopifnot(exprs = {
    hasNoDimnames(m)
    hasNoDimnames(m1); hasNoDimnames(m2)
    hasNoDimnames(m3); hasNoDimnames(m4)
    hasNoDimnames(Matrix(m) -> M)
    hasNoDimnames(as(M, "sparseMatrix"))
})



cleanEx()
nameEx("isSymmetric-methods")
### * isSymmetric-methods

flush(stderr()); flush(stdout())

### Name: isSymmetric-methods
### Title: Methods for Function 'isSymmetric' in Package 'Matrix'
### Aliases: isSymmetric isSymmetric-methods
###   isSymmetric,CsparseMatrix-method isSymmetric,RsparseMatrix-method
###   isSymmetric,TsparseMatrix-method isSymmetric,denseMatrix-method
###   isSymmetric,diagonalMatrix-method isSymmetric,indMatrix-method
###   isSymmetric,dgCMatrix-method isSymmetric,dgRMatrix-method
###   isSymmetric,dgTMatrix-method isSymmetric,dgeMatrix-method
###   isSymmetric,dtCMatrix-method isSymmetric,dtRMatrix-method
###   isSymmetric,dtTMatrix-method isSymmetric,dtpMatrix-method
###   isSymmetric,dtrMatrix-method
### Keywords: array programming methods

### ** Examples

isSymmetric(Diagonal(4)) # TRUE of course
M <- Matrix(c(1,2,2,1), 2,2)
isSymmetric(M) # TRUE (*and* of formal class "dsyMatrix")
isSymmetric(as(M, "generalMatrix")) # still symmetric, even if not "formally"
isSymmetric(triu(M)) # FALSE

## Look at implementations:
showMethods("isSymmetric", includeDefs = TRUE) # includes S3 generic from base



cleanEx()
nameEx("isTriangular")
### * isTriangular

flush(stderr()); flush(stdout())

### Name: isTriangular-methods
### Title: Test whether a Matrix is Triangular or Diagonal
### Aliases: isTriangular isTriangular-methods isDiagonal
###   isDiagonal-methods isTriangular,CsparseMatrix-method
###   isTriangular,RsparseMatrix-method isTriangular,TsparseMatrix-method
###   isTriangular,denseMatrix-method isTriangular,diagonalMatrix-method
###   isTriangular,indMatrix-method isTriangular,matrix-method
###   isDiagonal,CsparseMatrix-method isDiagonal,RsparseMatrix-method
###   isDiagonal,TsparseMatrix-method isDiagonal,denseMatrix-method
###   isDiagonal,diagonalMatrix-method isDiagonal,indMatrix-method
###   isDiagonal,matrix-method
### Keywords: array programming methods

### ** Examples

isTriangular(Diagonal(4))
## is TRUE: a diagonal matrix is also (both upper and lower) triangular
(M <- Matrix(c(1,2,0,1), 2,2))
isTriangular(M) # TRUE (*and* of formal class "dtrMatrix")
isTriangular(as(M, "generalMatrix")) # still triangular, even if not "formally"
isTriangular(crossprod(M)) # FALSE

isDiagonal(matrix(c(2,0,0,1), 2,2)) # TRUE

## Look at implementations:
showMethods("isTriangular", includeDefs = TRUE)
showMethods("isDiagonal", includeDefs = TRUE)



cleanEx()
nameEx("kronecker-methods")
### * kronecker-methods

flush(stderr()); flush(stdout())

### Name: kronecker-methods
### Title: Methods for Function 'kronecker()' in Package 'Matrix'
### Aliases: kronecker kronecker-methods
###   kronecker,CsparseMatrix,CsparseMatrix-method
###   kronecker,CsparseMatrix,Matrix-method
###   kronecker,CsparseMatrix,diagonalMatrix-method
###   kronecker,Matrix,matrix-method kronecker,Matrix,vector-method
###   kronecker,RsparseMatrix,Matrix-method
###   kronecker,RsparseMatrix,RsparseMatrix-method
###   kronecker,RsparseMatrix,diagonalMatrix-method
###   kronecker,TsparseMatrix,Matrix-method
###   kronecker,TsparseMatrix,TsparseMatrix-method
###   kronecker,TsparseMatrix,diagonalMatrix-method
###   kronecker,denseMatrix,Matrix-method
###   kronecker,denseMatrix,denseMatrix-method
###   kronecker,diagonalMatrix,CsparseMatrix-method
###   kronecker,diagonalMatrix,Matrix-method
###   kronecker,diagonalMatrix,RsparseMatrix-method
###   kronecker,diagonalMatrix,TsparseMatrix-method
###   kronecker,diagonalMatrix,diagonalMatrix-method
###   kronecker,diagonalMatrix,indMatrix-method
###   kronecker,indMatrix,Matrix-method
###   kronecker,indMatrix,diagonalMatrix-method
###   kronecker,indMatrix,indMatrix-method kronecker,matrix,Matrix-method
###   kronecker,vector,Matrix-method
### Keywords: algebra arith array methods methods array

### ** Examples

(t1 <- spMatrix(5,4, x= c(3,2,-7,11), i= 1:4, j=4:1)) #  5 x  4
(t2 <- kronecker(Diagonal(3, 2:4), t1))               # 15 x 12

## should also work with special-cased logical matrices
l3 <- upper.tri(matrix(,3,3))
M <- Matrix(l3)
(N <- as(M, "nsparseMatrix")) # "ntCMatrix" (upper triangular)
N2 <- as(N, "generalMatrix")  # (lost "t"riangularity)
MM <- kronecker(M,M)
NN <- kronecker(N,N) # "dtTMatrix" i.e. did keep
NN2 <- kronecker(N2,N2)
stopifnot(identical(NN,MM),
          is(NN2, "sparseMatrix"), all(NN2 == NN),
          is(NN, "triangularMatrix"))



cleanEx()
nameEx("ldenseMatrix-class")
### * ldenseMatrix-class

flush(stderr()); flush(stdout())

### Name: ldenseMatrix-class
### Title: Virtual Class "ldenseMatrix" of Dense Logical Matrices
### Aliases: ldenseMatrix-class !,ldenseMatrix-method
###   &,ldenseMatrix,ddiMatrix-method &,ldenseMatrix,ldiMatrix-method
###   &,ldenseMatrix,ndiMatrix-method *,ldenseMatrix,ddiMatrix-method
###   *,ldenseMatrix,ldiMatrix-method *,ldenseMatrix,ndiMatrix-method
###   Logic,ldenseMatrix,lsparseMatrix-method
###   Ops,ldenseMatrix,ldenseMatrix-method ^,ldenseMatrix,ddiMatrix-method
###   ^,ldenseMatrix,ldiMatrix-method ^,ldenseMatrix,ndiMatrix-method
###   coerce,matrix,ldenseMatrix-method coerce,vector,ldenseMatrix-method
###   which,ldenseMatrix-method
### Keywords: array classes

### ** Examples

showClass("ldenseMatrix")

as(diag(3) > 0, "ldenseMatrix")



cleanEx()
nameEx("ldiMatrix-class")
### * ldiMatrix-class

flush(stderr()); flush(stdout())

### Name: ldiMatrix-class
### Title: Class "ldiMatrix" of Diagonal Logical Matrices
### Aliases: ldiMatrix-class ndiMatrix-class !,ldiMatrix-method
###   %%,ldiMatrix,Matrix-method %%,ldiMatrix,ddenseMatrix-method
###   %%,ldiMatrix,ldenseMatrix-method %%,ldiMatrix,ndenseMatrix-method
###   %/%,ldiMatrix,Matrix-method %/%,ldiMatrix,ddenseMatrix-method
###   %/%,ldiMatrix,ldenseMatrix-method %/%,ldiMatrix,ndenseMatrix-method
###   &,ldiMatrix,Matrix-method &,ldiMatrix,ddenseMatrix-method
###   &,ldiMatrix,ldenseMatrix-method &,ldiMatrix,ndenseMatrix-method
###   *,ldiMatrix,Matrix-method *,ldiMatrix,ddenseMatrix-method
###   *,ldiMatrix,ldenseMatrix-method *,ldiMatrix,ndenseMatrix-method
###   /,ldiMatrix,Matrix-method /,ldiMatrix,ddenseMatrix-method
###   /,ldiMatrix,ldenseMatrix-method /,ldiMatrix,ndenseMatrix-method
###   Arith,ldiMatrix,logical-method Arith,ldiMatrix,numeric-method
###   Arith,logical,ldiMatrix-method Arith,numeric,ldiMatrix-method
###   Ops,ANY,ldiMatrix-method Ops,ldiMatrix,ANY-method
###   Ops,ldiMatrix,Matrix-method Ops,ldiMatrix,dMatrix-method
###   Ops,ldiMatrix,ddiMatrix-method Ops,ldiMatrix,ldiMatrix-method
###   Ops,ldiMatrix,ndiMatrix-method Ops,ldiMatrix,logical-method
###   Ops,ldiMatrix,numeric-method Ops,ldiMatrix,sparseMatrix-method
###   which,ldiMatrix-method !,ndiMatrix-method %%,ndiMatrix,Matrix-method
###   %%,ndiMatrix,ddenseMatrix-method %%,ndiMatrix,ldenseMatrix-method
###   %%,ndiMatrix,ndenseMatrix-method %/%,ndiMatrix,Matrix-method
###   %/%,ndiMatrix,ddenseMatrix-method %/%,ndiMatrix,ldenseMatrix-method
###   %/%,ndiMatrix,ndenseMatrix-method &,ndiMatrix,Matrix-method
###   &,ndiMatrix,ddenseMatrix-method &,ndiMatrix,ldenseMatrix-method
###   &,ndiMatrix,ndenseMatrix-method *,ndiMatrix,Matrix-method
###   *,ndiMatrix,Matrix-method *,ndiMatrix,ddenseMatrix-method
###   *,ndiMatrix,ldenseMatrix-method *,ndiMatrix,ndenseMatrix-method
###   /,ndiMatrix,Matrix-method /,ndiMatrix,ddenseMatrix-method
###   /,ndiMatrix,ldenseMatrix-method /,ndiMatrix,ndenseMatrix-method
###   Ops,ndiMatrix,ddiMatrix-method Ops,ndiMatrix,ldiMatrix-method
###   Ops,ndiMatrix,ndiMatrix-method which,ndiMatrix-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
(lM <- Diagonal(x = c(TRUE,FALSE,FALSE)))
str(lM)#> gory details (slots)

crossprod(lM) # numeric
(nM <- as(lM, "nMatrix"))
crossprod(nM) # pattern sparse



cleanEx()
nameEx("lgeMatrix-class")
### * lgeMatrix-class

flush(stderr()); flush(stdout())

### Name: lgeMatrix-class
### Title: Class "lgeMatrix" of General Dense Logical Matrices
### Aliases: lgeMatrix-class Arith,lgeMatrix,lgeMatrix-method
###   Compare,lgeMatrix,lgeMatrix-method Logic,lgeMatrix,lgeMatrix-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("lgeMatrix")
str(new("lgeMatrix"))
set.seed(1)
(lM <- Matrix(matrix(rnorm(28), 4,7) > 0))# a simple random lgeMatrix
set.seed(11)
(lC <- Matrix(matrix(rnorm(28), 4,7) > 0))# a simple random lgCMatrix
as(lM, "CsparseMatrix")



cleanEx()
nameEx("lsparseMatrix-classes")
### * lsparseMatrix-classes

flush(stderr()); flush(stdout())

### Name: lsparseMatrix-classes
### Title: Sparse logical matrices
### Aliases: lsparseMatrix-class lgCMatrix-class lgRMatrix-class
###   lgTMatrix-class ltCMatrix-class ltRMatrix-class ltTMatrix-class
###   lsCMatrix-class lsRMatrix-class lsTMatrix-class
###   !,lsparseMatrix-method Arith,lsparseMatrix,Matrix-method
###   Logic,lsparseMatrix,ldenseMatrix-method
###   Logic,lsparseMatrix,lsparseMatrix-method
###   Ops,lsparseMatrix,lsparseMatrix-method
###   Ops,lsparseMatrix,nsparseMatrix-method
###   coerce,matrix,lsparseMatrix-method coerce,vector,lsparseMatrix-method
###   which,lsparseMatrix-method Arith,lgCMatrix,lgCMatrix-method
###   Logic,lgCMatrix,lgCMatrix-method Arith,lgTMatrix,lgTMatrix-method
###   Logic,lgTMatrix,lgTMatrix-method Logic,ltCMatrix,ltCMatrix-method
###   Logic,lsCMatrix,lsCMatrix-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
(m <- Matrix(c(0,0,2:0), 3,5, dimnames=list(LETTERS[1:3],NULL)))
(lm <- (m > 1)) # lgC
!lm     # no longer sparse
stopifnot(is(lm,"lsparseMatrix"),
          identical(!lm, m <= 1))

data(KNex, package = "Matrix")
str(mmG.1 <- (KNex $ mm) > 0.1)# "lgC..."
table(mmG.1@x)# however with many ``non-structural zeros''
## from logical to nz_pattern -- okay when there are no NA's :
nmG.1 <- as(mmG.1, "nMatrix") # <<< has "TRUE" also where mmG.1 had FALSE
## from logical to "double"
dmG.1 <- as(mmG.1, "dMatrix") # has '0' and back:
lmG.1 <- as(dmG.1, "lMatrix")
stopifnot(identical(nmG.1, as((KNex $ mm) != 0,"nMatrix")),
          validObject(lmG.1),
          identical(lmG.1, mmG.1))

class(xnx <- crossprod(nmG.1))# "nsC.."
class(xlx <- crossprod(mmG.1))# "dsC.." : numeric
is0 <- (xlx == 0)
mean(as.vector(is0))# 99.3% zeros: quite sparse, but
table(xlx@x == 0)# more than half of the entries are (non-structural!) 0
stopifnot(isSymmetric(xlx), isSymmetric(xnx),
          ## compare xnx and xlx : have the *same* non-structural 0s :
          sapply(slotNames(xnx),
                 function(n) identical(slot(xnx, n), slot(xlx, n))))



cleanEx()
nameEx("lsyMatrix-class")
### * lsyMatrix-class

flush(stderr()); flush(stdout())

### Name: lsyMatrix-class
### Title: Symmetric Dense Logical Matrices
### Aliases: lsyMatrix-class lspMatrix-class
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
(M2 <- Matrix(c(TRUE, NA, FALSE, FALSE), 2, 2)) # logical dense (ltr)
str(M2)
# can
(sM <- M2 | t(M2)) # "lge"
as(sM, "symmetricMatrix")
str(sM <- as(sM, "packedMatrix")) # packed symmetric



cleanEx()
nameEx("ltrMatrix-class")
### * ltrMatrix-class

flush(stderr()); flush(stdout())

### Name: ltrMatrix-class
### Title: Triangular Dense Logical Matrices
### Aliases: ltrMatrix-class ltpMatrix-class
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("ltrMatrix")

str(new("ltpMatrix"))
(lutr <- as(upper.tri(matrix(, 4, 4)), "ldenseMatrix"))
str(lutp <- pack(lutr)) # packed matrix: only 10 = 4*(4+1)/2 entries
!lutp # the logical negation (is *not* logical triangular !)
## but this one is:
stopifnot(all.equal(lutp, pack(!!lutp)))



cleanEx()
nameEx("lu")
### * lu

flush(stderr()); flush(stdout())

### Name: lu-methods
### Title: Methods for LU Factorization
### Aliases: lu lu-methods lu,denseMatrix-method lu,diagonalMatrix-method
###   lu,dgCMatrix-method lu,dgRMatrix-method lu,dgTMatrix-method
###   lu,dgeMatrix-method lu,dsCMatrix-method lu,dsRMatrix-method
###   lu,dsTMatrix-method lu,dspMatrix-method lu,dsyMatrix-method
###   lu,dtCMatrix-method lu,dtRMatrix-method lu,dtTMatrix-method
###   lu,dtpMatrix-method lu,dtrMatrix-method lu,matrix-method
###   lu,sparseMatrix-method
### Keywords: algebra array methods

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showMethods("lu", inherited = FALSE)
set.seed(0)

## ---- Dense ----------------------------------------------------------

(A1 <- Matrix(rnorm(9L), 3L, 3L))
(lu.A1 <- lu(A1))

(A2 <- round(10 * A1[, -3L]))
(lu.A2 <- lu(A2))

## A ~ P1' L U in floating point
str(e.lu.A2 <- expand2(lu.A2), max.level = 2L)
stopifnot(all.equal(A2, Reduce(`%*%`, e.lu.A2)))

## ---- Sparse ---------------------------------------------------------

A3 <- as(readMM(system.file("external/pores_1.mtx", package = "Matrix")),
         "CsparseMatrix")
(lu.A3 <- lu(A3))

## A ~ P1' L U P2' in floating point
str(e.lu.A3 <- expand2(lu.A3), max.level = 2L)
stopifnot(all.equal(A3, Reduce(`%*%`, e.lu.A3)))



cleanEx()
nameEx("mat2triplet")
### * mat2triplet

flush(stderr()); flush(stdout())

### Name: mat2triplet
### Title: Map Matrix to its Triplet Representation
### Aliases: mat2triplet
### Keywords: array utilities

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
mat2triplet # simple definition

i <- c(1,3:8); j <- c(2,9,6:10); x <- 7 * (1:7)
(Ax <- sparseMatrix(i, j, x = x)) ##  8 x 10 "dgCMatrix"
str(trA <- mat2triplet(Ax))
stopifnot(i == sort(trA$i),  sort(j) == trA$j,  x == sort(trA$x))

D <- Diagonal(x=4:2)
summary(D)
str(mat2triplet(D))



cleanEx()
nameEx("matrix-products")
### * matrix-products

flush(stderr()); flush(stdout())

### Name: matmult-methods
### Title: Matrix (Cross) Products (of Transpose)
### Aliases: %*% %*%-methods crossprod crossprod-methods tcrossprod
###   tcrossprod-methods matmult-methods %*%,ANY,Matrix-method
###   %*%,ANY,sparseVector-method %*%,CsparseMatrix,CsparseMatrix-method
###   %*%,CsparseMatrix,RsparseMatrix-method
###   %*%,CsparseMatrix,TsparseMatrix-method
###   %*%,CsparseMatrix,denseMatrix-method
###   %*%,CsparseMatrix,diagonalMatrix-method
###   %*%,CsparseMatrix,matrix-method %*%,CsparseMatrix,vector-method
###   %*%,Matrix,ANY-method %*%,Matrix,indMatrix-method
###   %*%,Matrix,pMatrix-method %*%,Matrix,sparseVector-method
###   %*%,RsparseMatrix,CsparseMatrix-method
###   %*%,RsparseMatrix,RsparseMatrix-method
###   %*%,RsparseMatrix,TsparseMatrix-method
###   %*%,RsparseMatrix,denseMatrix-method
###   %*%,RsparseMatrix,diagonalMatrix-method
###   %*%,RsparseMatrix,matrix-method %*%,RsparseMatrix,vector-method
###   %*%,TsparseMatrix,CsparseMatrix-method
###   %*%,TsparseMatrix,RsparseMatrix-method
###   %*%,TsparseMatrix,TsparseMatrix-method
###   %*%,TsparseMatrix,denseMatrix-method
###   %*%,TsparseMatrix,diagonalMatrix-method
###   %*%,TsparseMatrix,matrix-method %*%,TsparseMatrix,vector-method
###   %*%,denseMatrix,CsparseMatrix-method
###   %*%,denseMatrix,RsparseMatrix-method
###   %*%,denseMatrix,TsparseMatrix-method
###   %*%,denseMatrix,denseMatrix-method
###   %*%,denseMatrix,diagonalMatrix-method %*%,denseMatrix,matrix-method
###   %*%,denseMatrix,vector-method %*%,diagonalMatrix,CsparseMatrix-method
###   %*%,diagonalMatrix,RsparseMatrix-method
###   %*%,diagonalMatrix,TsparseMatrix-method
###   %*%,diagonalMatrix,denseMatrix-method
###   %*%,diagonalMatrix,diagonalMatrix-method
###   %*%,diagonalMatrix,matrix-method %*%,diagonalMatrix,vector-method
###   %*%,indMatrix,Matrix-method %*%,indMatrix,indMatrix-method
###   %*%,indMatrix,matrix-method %*%,indMatrix,pMatrix-method
###   %*%,indMatrix,vector-method %*%,matrix,CsparseMatrix-method
###   %*%,matrix,RsparseMatrix-method %*%,matrix,TsparseMatrix-method
###   %*%,matrix,denseMatrix-method %*%,matrix,diagonalMatrix-method
###   %*%,matrix,indMatrix-method %*%,matrix,pMatrix-method
###   %*%,matrix,sparseVector-method %*%,pMatrix,Matrix-method
###   %*%,pMatrix,indMatrix-method %*%,pMatrix,matrix-method
###   %*%,pMatrix,pMatrix-method %*%,pMatrix,vector-method
###   %*%,sparseVector,ANY-method %*%,sparseVector,Matrix-method
###   %*%,sparseVector,matrix-method %*%,sparseVector,sparseVector-method
###   %*%,sparseVector,vector-method %*%,vector,CsparseMatrix-method
###   %*%,vector,RsparseMatrix-method %*%,vector,TsparseMatrix-method
###   %*%,vector,denseMatrix-method %*%,vector,diagonalMatrix-method
###   %*%,vector,indMatrix-method %*%,vector,pMatrix-method
###   %*%,vector,sparseVector-method crossprod,ANY,Matrix-method
###   crossprod,ANY,sparseVector-method
###   crossprod,CsparseMatrix,CsparseMatrix-method
###   crossprod,CsparseMatrix,RsparseMatrix-method
###   crossprod,CsparseMatrix,TsparseMatrix-method
###   crossprod,CsparseMatrix,denseMatrix-method
###   crossprod,CsparseMatrix,diagonalMatrix-method
###   crossprod,CsparseMatrix,matrix-method
###   crossprod,CsparseMatrix,missing-method
###   crossprod,CsparseMatrix,vector-method crossprod,Matrix,ANY-method
###   crossprod,Matrix,indMatrix-method crossprod,Matrix,pMatrix-method
###   crossprod,Matrix,sparseVector-method
###   crossprod,RsparseMatrix,CsparseMatrix-method
###   crossprod,RsparseMatrix,RsparseMatrix-method
###   crossprod,RsparseMatrix,TsparseMatrix-method
###   crossprod,RsparseMatrix,denseMatrix-method
###   crossprod,RsparseMatrix,diagonalMatrix-method
###   crossprod,RsparseMatrix,matrix-method
###   crossprod,RsparseMatrix,missing-method
###   crossprod,RsparseMatrix,vector-method
###   crossprod,TsparseMatrix,CsparseMatrix-method
###   crossprod,TsparseMatrix,RsparseMatrix-method
###   crossprod,TsparseMatrix,TsparseMatrix-method
###   crossprod,TsparseMatrix,denseMatrix-method
###   crossprod,TsparseMatrix,diagonalMatrix-method
###   crossprod,TsparseMatrix,matrix-method
###   crossprod,TsparseMatrix,missing-method
###   crossprod,TsparseMatrix,vector-method
###   crossprod,denseMatrix,CsparseMatrix-method
###   crossprod,denseMatrix,RsparseMatrix-method
###   crossprod,denseMatrix,TsparseMatrix-method
###   crossprod,denseMatrix,denseMatrix-method
###   crossprod,denseMatrix,diagonalMatrix-method
###   crossprod,denseMatrix,matrix-method
###   crossprod,denseMatrix,missing-method
###   crossprod,denseMatrix,vector-method
###   crossprod,diagonalMatrix,CsparseMatrix-method
###   crossprod,diagonalMatrix,RsparseMatrix-method
###   crossprod,diagonalMatrix,TsparseMatrix-method
###   crossprod,diagonalMatrix,denseMatrix-method
###   crossprod,diagonalMatrix,diagonalMatrix-method
###   crossprod,diagonalMatrix,matrix-method
###   crossprod,diagonalMatrix,missing-method
###   crossprod,diagonalMatrix,vector-method
###   crossprod,indMatrix,Matrix-method crossprod,indMatrix,matrix-method
###   crossprod,indMatrix,missing-method crossprod,indMatrix,vector-method
###   crossprod,matrix,CsparseMatrix-method
###   crossprod,matrix,RsparseMatrix-method
###   crossprod,matrix,TsparseMatrix-method
###   crossprod,matrix,denseMatrix-method
###   crossprod,matrix,diagonalMatrix-method
###   crossprod,matrix,indMatrix-method crossprod,matrix,pMatrix-method
###   crossprod,matrix,sparseVector-method crossprod,pMatrix,missing-method
###   crossprod,pMatrix,pMatrix-method crossprod,sparseVector,ANY-method
###   crossprod,sparseVector,Matrix-method
###   crossprod,sparseVector,matrix-method
###   crossprod,sparseVector,missing-method
###   crossprod,sparseVector,sparseVector-method
###   crossprod,sparseVector,vector-method
###   crossprod,vector,CsparseMatrix-method
###   crossprod,vector,RsparseMatrix-method
###   crossprod,vector,TsparseMatrix-method
###   crossprod,vector,denseMatrix-method
###   crossprod,vector,diagonalMatrix-method
###   crossprod,vector,indMatrix-method crossprod,vector,pMatrix-method
###   crossprod,vector,sparseVector-method tcrossprod,ANY,Matrix-method
###   tcrossprod,ANY,sparseVector-method
###   tcrossprod,CsparseMatrix,CsparseMatrix-method
###   tcrossprod,CsparseMatrix,RsparseMatrix-method
###   tcrossprod,CsparseMatrix,TsparseMatrix-method
###   tcrossprod,CsparseMatrix,denseMatrix-method
###   tcrossprod,CsparseMatrix,diagonalMatrix-method
###   tcrossprod,CsparseMatrix,matrix-method
###   tcrossprod,CsparseMatrix,missing-method
###   tcrossprod,CsparseMatrix,vector-method tcrossprod,Matrix,ANY-method
###   tcrossprod,Matrix,indMatrix-method
###   tcrossprod,Matrix,sparseVector-method
###   tcrossprod,RsparseMatrix,CsparseMatrix-method
###   tcrossprod,RsparseMatrix,RsparseMatrix-method
###   tcrossprod,RsparseMatrix,TsparseMatrix-method
###   tcrossprod,RsparseMatrix,denseMatrix-method
###   tcrossprod,RsparseMatrix,diagonalMatrix-method
###   tcrossprod,RsparseMatrix,matrix-method
###   tcrossprod,RsparseMatrix,missing-method
###   tcrossprod,RsparseMatrix,vector-method
###   tcrossprod,TsparseMatrix,CsparseMatrix-method
###   tcrossprod,TsparseMatrix,RsparseMatrix-method
###   tcrossprod,TsparseMatrix,TsparseMatrix-method
###   tcrossprod,TsparseMatrix,denseMatrix-method
###   tcrossprod,TsparseMatrix,diagonalMatrix-method
###   tcrossprod,TsparseMatrix,matrix-method
###   tcrossprod,TsparseMatrix,missing-method
###   tcrossprod,TsparseMatrix,vector-method
###   tcrossprod,denseMatrix,CsparseMatrix-method
###   tcrossprod,denseMatrix,RsparseMatrix-method
###   tcrossprod,denseMatrix,TsparseMatrix-method
###   tcrossprod,denseMatrix,denseMatrix-method
###   tcrossprod,denseMatrix,diagonalMatrix-method
###   tcrossprod,denseMatrix,matrix-method
###   tcrossprod,denseMatrix,missing-method
###   tcrossprod,denseMatrix,vector-method
###   tcrossprod,diagonalMatrix,CsparseMatrix-method
###   tcrossprod,diagonalMatrix,RsparseMatrix-method
###   tcrossprod,diagonalMatrix,TsparseMatrix-method
###   tcrossprod,diagonalMatrix,denseMatrix-method
###   tcrossprod,diagonalMatrix,diagonalMatrix-method
###   tcrossprod,diagonalMatrix,matrix-method
###   tcrossprod,diagonalMatrix,missing-method
###   tcrossprod,diagonalMatrix,vector-method
###   tcrossprod,indMatrix,Matrix-method tcrossprod,indMatrix,matrix-method
###   tcrossprod,indMatrix,missing-method
###   tcrossprod,indMatrix,vector-method
###   tcrossprod,matrix,CsparseMatrix-method
###   tcrossprod,matrix,RsparseMatrix-method
###   tcrossprod,matrix,TsparseMatrix-method
###   tcrossprod,matrix,denseMatrix-method
###   tcrossprod,matrix,diagonalMatrix-method
###   tcrossprod,matrix,indMatrix-method
###   tcrossprod,matrix,sparseVector-method
###   tcrossprod,pMatrix,Matrix-method tcrossprod,pMatrix,matrix-method
###   tcrossprod,pMatrix,missing-method tcrossprod,pMatrix,pMatrix-method
###   tcrossprod,pMatrix,vector-method tcrossprod,sparseVector,ANY-method
###   tcrossprod,sparseVector,Matrix-method
###   tcrossprod,sparseVector,matrix-method
###   tcrossprod,sparseVector,missing-method
###   tcrossprod,sparseVector,sparseVector-method
###   tcrossprod,sparseVector,vector-method
###   tcrossprod,vector,CsparseMatrix-method
###   tcrossprod,vector,RsparseMatrix-method
###   tcrossprod,vector,TsparseMatrix-method
###   tcrossprod,vector,denseMatrix-method
###   tcrossprod,vector,diagonalMatrix-method
###   tcrossprod,vector,indMatrix-method
###   tcrossprod,vector,sparseVector-method
### Keywords: algebra arith array

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
 ## A random sparse "incidence" matrix :
 m <- matrix(0, 400, 500)
 set.seed(12)
 m[runif(314, 0, length(m))] <- 1
 mm <- as(m, "CsparseMatrix")
 object.size(m) / object.size(mm) # smaller by a factor of > 200

 ## tcrossprod() is very fast:
 system.time(tCmm <- tcrossprod(mm))# 0   (PIII, 933 MHz)
 system.time(cm <- crossprod(t(m))) # 0.16
 system.time(cm. <- tcrossprod(m))  # 0.02

 stopifnot(cm == as(tCmm, "matrix"))

 ## show sparse sub matrix
 tCmm[1:16, 1:30]



cleanEx()
nameEx("nMatrix-class")
### * nMatrix-class

flush(stderr()); flush(stdout())

### Name: nMatrix-class
### Title: Class "nMatrix" of Non-zero Pattern Matrices
### Aliases: nMatrix-class Arith,logical,nMatrix-method
###   Arith,nMatrix,logical-method Arith,nMatrix,numeric-method
###   Arith,numeric,nMatrix-method Compare,logical,nMatrix-method
###   Compare,nMatrix,logical-method Compare,nMatrix,nMatrix-method
###   Compare,nMatrix,numeric-method Compare,numeric,nMatrix-method
###   Logic,logical,nMatrix-method Logic,nMatrix,Matrix-method
###   Logic,nMatrix,logical-method Logic,nMatrix,nMatrix-method
###   Logic,nMatrix,numeric-method Logic,nMatrix,sparseVector-method
###   Logic,numeric,nMatrix-method Ops,nMatrix,dMatrix-method
###   Ops,nMatrix,lMatrix-method Ops,nMatrix,nMatrix-method
###   Ops,nMatrix,numeric-method Ops,numeric,nMatrix-method
###   coerce,matrix,nMatrix-method coerce,vector,nMatrix-method
### Keywords: array classes

### ** Examples

getClass("nMatrix")

L3 <- Matrix(upper.tri(diag(3)))
L3 # an "ltCMatrix"
as(L3, "nMatrix") # -> ntC*

## similar, not using Matrix()
as(upper.tri(diag(3)), "nMatrix")# currently "ngTMatrix"



cleanEx()
nameEx("ndenseMatrix-class")
### * ndenseMatrix-class

flush(stderr()); flush(stdout())

### Name: ndenseMatrix-class
### Title: Virtual Class "ndenseMatrix" of Dense Logical Matrices
### Aliases: ndenseMatrix-class !,ndenseMatrix-method
###   &,ndenseMatrix,ddiMatrix-method &,ndenseMatrix,ldiMatrix-method
###   &,ndenseMatrix,ndiMatrix-method *,ndenseMatrix,ddiMatrix-method
###   *,ndenseMatrix,ldiMatrix-method *,ndenseMatrix,ndiMatrix-method
###   Ops,ndenseMatrix,ndenseMatrix-method ^,ndenseMatrix,ddiMatrix-method
###   ^,ndenseMatrix,ldiMatrix-method ^,ndenseMatrix,ndiMatrix-method
###   coerce,matrix,ndenseMatrix-method coerce,vector,ndenseMatrix-method
###   which,ndenseMatrix-method
### Keywords: array classes

### ** Examples

showClass("ndenseMatrix")

as(diag(3) > 0, "ndenseMatrix")# -> "nge"



cleanEx()
nameEx("nearPD")
### * nearPD

flush(stderr()); flush(stdout())

### Name: nearPD
### Title: Nearest Positive Definite Matrix
### Aliases: nearPD
### Keywords: algebra array utilities

### ** Examples

## Don't show: 
 
library(    stats, pos = "package:base", verbose = FALSE)
library( graphics, pos = "package:base", verbose = FALSE)
library(grDevices, pos = "package:base", verbose = FALSE)
library(    utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
 ## Higham(2002), p.334f - simple example
 A <- matrix(1, 3,3); A[1,3] <- A[3,1] <- 0
 n.A <- nearPD(A, corr=TRUE, do2eigen=FALSE)
 n.A[c("mat", "normF")]
 n.A.m <- nearPD(A, corr=TRUE, do2eigen=FALSE, base.matrix=TRUE)$mat
 stopifnot(exprs = {                           #=--------------
   all.equal(n.A$mat[1,2], 0.760689917)
   all.equal(n.A$normF, 0.52779033, tolerance=1e-9)
   all.equal(n.A.m, unname(as.matrix(n.A$mat)), tolerance = 1e-15)# seen rel.d.= 1.46e-16
 })
 set.seed(27)
 m <- matrix(round(rnorm(25),2), 5, 5)
 m <- m + t(m)
 diag(m) <- pmax(0, diag(m)) + 1
 (m <- round(cov2cor(m), 2))

 str(near.m <- nearPD(m, trace = TRUE))
 round(near.m$mat, 2)
 norm(m - near.m$mat) # 1.102 / 1.08

 if(requireNamespace("sfsmisc")) {
    m2 <- sfsmisc::posdefify(m) # a simpler approach
    norm(m - m2)  # 1.185, i.e., slightly "less near"
 }

 round(nearPD(m, only.values=TRUE), 9)

## A longer example, extended from Jens' original,
## showing the effects of some of the options:

pr <- Matrix(c(1,     0.477, 0.644, 0.478, 0.651, 0.826,
               0.477, 1,     0.516, 0.233, 0.682, 0.75,
               0.644, 0.516, 1,     0.599, 0.581, 0.742,
               0.478, 0.233, 0.599, 1,     0.741, 0.8,
               0.651, 0.682, 0.581, 0.741, 1,     0.798,
               0.826, 0.75,  0.742, 0.8,   0.798, 1),
             nrow = 6, ncol = 6)

nc.  <- nearPD(pr, conv.tol = 1e-7) # default
nc.$iterations  # 2
nc.1 <- nearPD(pr, conv.tol = 1e-7, corr = TRUE)
nc.1$iterations # 11 / 12 (!)
ncr   <- nearPD(pr, conv.tol = 1e-15)
str(ncr)# still 2 iterations
ncr.1 <- nearPD(pr, conv.tol = 1e-15, corr = TRUE)
ncr.1 $ iterations # 27 / 30 !

ncF <- nearPD(pr, conv.tol = 1e-15, conv.norm = "F")
stopifnot(all.equal(ncr, ncF))# norm type does not matter at all in this example

## But indeed, the 'corr = TRUE' constraint did ensure a better solution;
## cov2cor() does not just fix it up equivalently :
norm(pr - cov2cor(ncr$mat)) # = 0.09994
norm(pr -       ncr.1$mat)  # = 0.08746 / 0.08805

### 3) a real data example from a 'systemfit' model (3 eq.):
(load(system.file("external", "symW.rda", package="Matrix"))) # "symW"
dim(symW) #  24 x 24
class(symW)# "dsCMatrix": sparse symmetric
if(dev.interactive())  image(symW)
EV <- eigen(symW, only=TRUE)$values
summary(EV) ## looking more closely {EV sorted decreasingly}:
tail(EV)# all 6 are negative
EV2 <- eigen(sWpos <- nearPD(symW)$mat, only=TRUE)$values
stopifnot(EV2 > 0)
if(requireNamespace("sfsmisc")) {
    plot(pmax(1e-3,EV), EV2, type="o", log="xy", xaxt="n", yaxt="n")
    for(side in 1:2) sfsmisc::eaxis(side)
} else
    plot(pmax(1e-3,EV), EV2, type="o", log="xy")
abline(0, 1, col="red3", lty=2)



cleanEx()
nameEx("ngeMatrix-class")
### * ngeMatrix-class

flush(stderr()); flush(stdout())

### Name: ngeMatrix-class
### Title: Class "ngeMatrix" of General Dense Nonzero-pattern Matrices
### Aliases: ngeMatrix-class Arith,ngeMatrix,ngeMatrix-method
###   Compare,ngeMatrix,ngeMatrix-method Logic,ngeMatrix,ngeMatrix-method
### Keywords: array classes

### ** Examples

showClass("ngeMatrix")
## "lgeMatrix" is really more relevant



cleanEx()
nameEx("nnzero")
### * nnzero

flush(stderr()); flush(stdout())

### Name: nnzero-methods
### Title: The Number of Non-Zero Values of a Matrix
### Aliases: nnzero nnzero-methods nnzero,ANY-method
###   nnzero,CHMfactor-method nnzero,denseMatrix-method
###   nnzero,diagonalMatrix-method nnzero,indMatrix-method
###   nnzero,sparseMatrix-method nnzero,vector-method
### Keywords: array logic methods

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
m <- Matrix(0+1:28, nrow = 4)
m[-3,c(2,4:5,7)] <- m[ 3, 1:4] <- m[1:3, 6] <- 0
(mT <- as(m, "TsparseMatrix"))
nnzero(mT)
(S <- crossprod(mT))
nnzero(S)
str(S) # slots are smaller than nnzero()
stopifnot(nnzero(S) == sum(as.matrix(S) != 0))# failed earlier

data(KNex, package = "Matrix")
M <- KNex$mm
class(M)
dim(M)
length(M); stopifnot(length(M) == prod(dim(M)))
nnzero(M) # more relevant than length
## the above are also visible from
str(M)



cleanEx()
nameEx("norm")
### * norm

flush(stderr()); flush(stdout())

### Name: norm-methods
### Title: Matrix Norms
### Aliases: norm norm-methods norm,ANY,missing-method
###   norm,denseMatrix,character-method
###   norm,diagonalMatrix,character-method norm,indMatrix,character-method
###   norm,pMatrix,character-method norm,sparseMatrix,character-method
### Keywords: algebra math methods

### ** Examples

x <- Hilbert(9)
norm(x)# = "O" = "1"
stopifnot(identical(norm(x), norm(x, "1")))
norm(x, "I")# the same, because 'x' is symmetric

allnorms <- function(x) {
    ## norm(NA, "2") did not work until R 4.0.0
    do2 <- getRversion() >= "4.0.0" || !anyNA(x)
    vapply(c("1", "I", "F", "M", if(do2) "2"), norm, 0, x = x)
}
allnorms(x)
allnorms(Hilbert(10))

i <- c(1,3:8); j <- c(2,9,6:10); x <- 7 * (1:7)
A <- sparseMatrix(i, j, x = x)                      ##  8 x 10 "dgCMatrix"
(sA <- sparseMatrix(i, j, x = x, symmetric = TRUE)) ## 10 x 10 "dsCMatrix"
(tA <- sparseMatrix(i, j, x = x, triangular= TRUE)) ## 10 x 10 "dtCMatrix"
(allnorms(A) -> nA)
allnorms(sA)
allnorms(tA)
stopifnot(all.equal(nA, allnorms(as(A, "matrix"))),
	  all.equal(nA, allnorms(tA))) # because tA == rbind(A, 0, 0)
A. <- A; A.[1,3] <- NA
stopifnot(is.na(allnorms(A.))) # gave error



cleanEx()
nameEx("nsparseMatrix-classes")
### * nsparseMatrix-classes

flush(stderr()); flush(stdout())

### Name: nsparseMatrix-classes
### Title: Sparse "pattern" Matrices
### Aliases: nsparseMatrix-class ngCMatrix-class ngRMatrix-class
###   ngTMatrix-class ntCMatrix-class ntRMatrix-class ntTMatrix-class
###   nsCMatrix-class nsRMatrix-class nsTMatrix-class
###   !,nsparseMatrix-method -,nsparseMatrix,missing-method
###   Arith,nsparseMatrix,Matrix-method
###   Arith,dsparseMatrix,nsparseMatrix-method
###   Arith,lsparseMatrix,nsparseMatrix-method
###   Arith,nsparseMatrix,dsparseMatrix-method
###   Arith,nsparseMatrix,lsparseMatrix-method
###   Ops,nsparseMatrix,dsparseMatrix-method
###   Ops,nsparseMatrix,lsparseMatrix-method
###   Ops,nsparseMatrix,sparseMatrix-method
###   coerce,matrix,nsparseMatrix-method
###   coerce,nsparseMatrix,indMatrix-method
###   coerce,nsparseMatrix,pMatrix-method
###   coerce,vector,nsparseMatrix-method which,nsparseMatrix-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
(m <- Matrix(c(0,0,2:0), 3,5, dimnames=list(LETTERS[1:3],NULL)))
## ``extract the nonzero-pattern of (m) into an nMatrix'':
nm <- as(m, "nsparseMatrix") ## -> will be a "ngCMatrix"
str(nm) # no 'x' slot
nnm <- !nm # no longer sparse
## consistency check:
stopifnot(xor(as( nm, "matrix"),
              as(nnm, "matrix")))

## low-level way of adding "non-structural zeros" :
nnm <- as(nnm, "lsparseMatrix") # "lgCMatrix"
nnm@x[2:4] <- c(FALSE, NA, NA)
nnm
as(nnm, "nMatrix") # NAs *and* non-structural 0  |--->  'TRUE'

data(KNex, package = "Matrix")
nmm <- as(KNex $ mm, "nMatrix")
str(xlx <- crossprod(nmm))# "nsCMatrix"
stopifnot(isSymmetric(xlx))
image(xlx, main=paste("crossprod(nmm) : Sparse", class(xlx)))



cleanEx()
nameEx("nsyMatrix-class")
### * nsyMatrix-class

flush(stderr()); flush(stdout())

### Name: nsyMatrix-class
### Title: Symmetric Dense Nonzero-Pattern Matrices
### Aliases: nsyMatrix-class nspMatrix-class
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
(s0 <- new("nsyMatrix"))

(M2 <- Matrix(c(TRUE, NA, FALSE, FALSE), 2, 2)) # logical dense (ltr)
(sM <- M2 & t(M2))                       # -> "lge"
class(sM <- as(sM, "nMatrix"))           # -> "nge"
     (sM <- as(sM, "symmetricMatrix"))   # -> "nsy"
str(sM <- as(sM, "packedMatrix")) # -> "nsp", i.e., packed symmetric



cleanEx()
nameEx("ntrMatrix-class")
### * ntrMatrix-class

flush(stderr()); flush(stdout())

### Name: ntrMatrix-class
### Title: Triangular Dense Logical Matrices
### Aliases: ntrMatrix-class ntpMatrix-class
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("ntrMatrix")

str(new("ntpMatrix"))
(nutr <- as(upper.tri(matrix(, 4, 4)), "ndenseMatrix"))
str(nutp <- pack(nutr)) # packed matrix: only 10 = 4*(4+1)/2 entries
!nutp # the logical negation (is *not* logical triangular !)
## but this one is:
stopifnot(all.equal(nutp, pack(!!nutp)))



cleanEx()
nameEx("number-class")
### * number-class

flush(stderr()); flush(stdout())

### Name: number-class
### Title: Class "number" of Possibly Complex Numbers
### Aliases: number-class
### Keywords: classes

### ** Examples

showClass("number")
stopifnot( is(1i, "number"), is(pi, "number"), is(1:3, "number") )



cleanEx()
nameEx("pMatrix-class")
### * pMatrix-class

flush(stderr()); flush(stdout())

### Name: pMatrix-class
### Title: Permutation matrices
### Aliases: pMatrix-class coerce,matrix,pMatrix-method
###   coerce,numeric,pMatrix-method determinant,pMatrix,logical-method
###   t,pMatrix-method
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
(pm1 <- as(as.integer(c(2,3,1)), "pMatrix"))
t(pm1) # is the same as
solve(pm1)
pm1 %*% t(pm1) # check that the transpose is the inverse
stopifnot(all(diag(3) == as(pm1 %*% t(pm1), "matrix")),
          is.logical(as(pm1, "matrix")))

set.seed(11)
## random permutation matrix :
(p10 <- as(sample(10),"pMatrix"))

## Permute rows / columns of a numeric matrix :
(mm <- round(array(rnorm(3 * 3), c(3, 3)), 2))
mm %*% pm1
pm1 %*% mm
try(as(as.integer(c(3,3,1)), "pMatrix"))# Error: not a permutation

as(pm1, "TsparseMatrix")
p10[1:7, 1:4] # gives an "ngTMatrix" (most economic!)

## row-indexing of a <pMatrix> keeps it as an <indMatrix>:
p10[1:3, ]



cleanEx()
nameEx("packedMatrix-class")
### * packedMatrix-class

flush(stderr()); flush(stdout())

### Name: packedMatrix-class
### Title: Virtual Class '"packedMatrix"' of Packed Dense Matrices
### Aliases: packedMatrix-class coerce,matrix,packedMatrix-method
###   cov2cor,packedMatrix-method
### Keywords: array classes

### ** Examples

showClass("packedMatrix")
showMethods(classes = "packedMatrix")



cleanEx()
nameEx("printSpMatrix")
### * printSpMatrix

flush(stderr()); flush(stdout())

### Name: printSpMatrix
### Title: Format and Print Sparse Matrices Flexibly
### Aliases: formatSpMatrix printSpMatrix printSpMatrix2
### Keywords: character print utilities

### ** Examples

f1 <- gl(5, 3, labels = LETTERS[1:5])
X <- as(f1, "sparseMatrix")
X ## <==>  show(X)  <==>  print(X)
t(X) ## shows column names, since only 5 columns

X2 <- as(gl(12, 3, labels = paste(LETTERS[1:12],"c",sep=".")),
         "sparseMatrix")
X2
## less nice, but possible:
print(X2, col.names = TRUE) # use [,1] [,2] .. => does not fit

## Possibilities with column names printing:
      t(X2) # suppressing column names
print(t(X2), col.names=TRUE)
print(t(X2), zero.print = "", col.names="abbr. 1")
print(t(X2), zero.print = "-", col.names="substring 2")

## Don't show: 
op <- options(max.print = 25000, width = 80)
sink(print(tempfile()))
M <- Matrix(0, 10000, 100)
M[1,1] <- M[2,3] <- 3.14
st <- system.time(show(M))
sink()
st

if(interactive() || nzchar(Sys.getenv("R_MATRIX_CHECK_EXTRA")))
## valgrind (2023-07-26) gave 10.5 sec!
stopifnot(st[1] < 1.0) # only 0.09 on cmath-3
options(op)
## End(Don't show)



cleanEx()
nameEx("qr-methods")
### * qr-methods

flush(stderr()); flush(stdout())

### Name: qr-methods
### Title: Methods for QR Factorization
### Aliases: qr qr-methods qr,dgCMatrix-method qr,sparseMatrix-method
### Keywords: algebra array methods

### ** Examples

showMethods("qr", inherited = FALSE)

## Rank deficient: columns 3 {b2} and 6 {c3} are "extra"
M <- as(cbind(a1 = 1,
              b1 = rep(c(1, 0), each = 3L),
              b2 = rep(c(0, 1), each = 3L),
              c1 = rep(c(1, 0, 0), 2L),
              c2 = rep(c(0, 1, 0), 2L),
              c3 = rep(c(0, 0, 1), 2L)),
        "CsparseMatrix")
rownames(M) <- paste0("r", seq_len(nrow(M)))
b <- 1:6
eps <- .Machine$double.eps

## .... [1] full rank ..................................................
## ===> a least squares solution of A x = b exists
##      and is unique _in exact arithmetic_

(A1 <- M[, -c(3L, 6L)])
(qr.A1 <- qr(A1))

stopifnot(exprs = {
    rankMatrix(A1) == ncol(A1)
    { d1 <- abs(diag(qr.A1@R)); sum(d1 < max(d1) * eps) == 0L }
    rcond(crossprod(A1)) >= eps
    all.equal(qr.coef(qr.A1, b), drop(solve(crossprod(A1), crossprod(A1, b))))
    all.equal(qr.fitted(qr.A1, b) + qr.resid(qr.A1, b), b)
})

## .... [2] numerically rank deficient with full structural rank .......
## ===> a least squares solution of A x = b does not
##      exist or is not unique _in exact arithmetic_

(A2 <- M)
(qr.A2 <- qr(A2))

stopifnot(exprs = {
    rankMatrix(A2) == ncol(A2) - 2L
    { d2 <- abs(diag(qr.A2@R)); sum(d2 < max(d2) * eps) == 2L }
    rcond(crossprod(A2)) < eps

    ## 'qr.coef' computes unique least squares solution of "nearby" problem
    ## Z x = b for some full rank Z ~ A, currently without warning {FIXME} !
    tryCatch({ qr.coef(qr.A2, b); TRUE }, condition = function(x) FALSE)

    all.equal(qr.fitted(qr.A2, b) + qr.resid(qr.A2, b), b)
})

## .... [3] numerically and structurally rank deficient ................
## ===> factorization of _augmented_ matrix with
##      full structural rank proceeds as in [2]

##  NB: implementation details are subject to change; see (*) below

A3 <- M
A3[, c(3L, 6L)] <- 0
A3
(qr.A3 <- qr(A3)) # with a warning ... "additional 2 row(s) of zeros"

stopifnot(exprs = {
    ## sparseQR object preserves the unaugmented dimensions (*)
    dim(qr.A3  ) == dim(A3)
    dim(qr.A3@V) == dim(A3) + c(2L, 0L)
    dim(qr.A3@R) == dim(A3) + c(2L, 0L)

    ## The augmented matrix remains numerically rank deficient
    rankMatrix(A3) == ncol(A3) - 2L
    { d3 <- abs(diag(qr.A3@R)); sum(d3 < max(d3) * eps) == 2L }
    rcond(crossprod(A3)) < eps
})

## Auxiliary functions accept and return a vector or matrix
## with dimensions corresponding to the unaugmented matrix (*),
## in all cases with a warning
qr.coef  (qr.A3, b)
qr.fitted(qr.A3, b)
qr.resid (qr.A3, b)

## .... [4] yet more examples ..........................................

## By disabling column pivoting, one gets the "vanilla" factorization
## A = Q~ R, where Q~ := P1' Q is orthogonal because P1 and Q are

(qr.A1.pp <- qr(A1, order = 0L)) # partial pivoting

ae1 <- function(a, b, ...) all.equal(as(a, "matrix"), as(b, "matrix"), ...)
ae2 <- function(a, b, ...) ae1(unname(a), unname(b), ...)

stopifnot(exprs = {
    length(qr.A1   @q) == ncol(A1)
    length(qr.A1.pp@q) == 0L # indicating no column pivoting
    ae2(A1[, qr.A1@q + 1L], qr.Q(qr.A1   ) %*% qr.R(qr.A1   ))
    ae2(A1                , qr.Q(qr.A1.pp) %*% qr.R(qr.A1.pp))
})



cleanEx()
nameEx("rankMatrix")
### * rankMatrix

flush(stderr()); flush(stdout())

### Name: rankMatrix
### Title: Rank of a Matrix
### Aliases: rankMatrix qr2rankMatrix
### Keywords: algebra utilities

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
rankMatrix(cbind(1, 0, 1:3)) # 2

(meths <- eval(formals(rankMatrix)$method))

## a "border" case:
H12 <- Hilbert(12)
rankMatrix(H12, tol = 1e-20) # 12;  but  11  with default method & tol.
sapply(meths, function(.m.) rankMatrix(H12, method = .m.))
## tolNorm2   qr.R  qrLINPACK   qr  useGrad maybeGrad
##       11     11         12   12       11        11
## The meaning of 'tol' for method="qrLINPACK" and *dense* x is not entirely "scale free"
rMQL <- function(ex, M) rankMatrix(M, method="qrLINPACK",tol = 10^-ex)
rMQR <- function(ex, M) rankMatrix(M, method="qr.R",     tol = 10^-ex)
sapply(5:15, rMQL, M = H12) # result is platform dependent
##  7  7  8 10 10 11 11 11 12 12 12  {x86_64}
sapply(5:15, rMQL, M = 1000 * H12) # not identical unfortunately
##  7  7  8 10 11 11 12 12 12 12 12
sapply(5:15, rMQR, M = H12)
##  5  6  7  8  8  9  9 10 10 11 11
sapply(5:15, rMQR, M = 1000 * H12) # the *same*
## Don't show: 
  (r12 <- sapply(5:15, rMQR, M = H12))
  stopifnot(identical(r12, sapply(5:15, rMQR, M = H12 / 100)),
            identical(r12, sapply(5:15, rMQR, M = H12 * 1e5)))

  rM1 <- function(ex, M) rankMatrix(M, tol = 10^-ex)
  (r12 <- sapply(5:15, rM1, M = H12))
  stopifnot(identical(r12, sapply(5:15, rM1, M = H12 / 100)),
            identical(r12, sapply(5:15, rM1, M = H12 * 1e5)))
## End(Don't show)

## "sparse" case:
M15 <- kronecker(diag(x=c(100,1,10)), Hilbert(5))
sapply(meths, function(.m.) rankMatrix(M15, method = .m.))
#--> all 15, but 'useGrad' has 14.
sapply(meths, function(.m.) rankMatrix(M15, method = .m., tol = 1e-7)) # all 14

## "large" sparse
n <- 250000; p <- 33; nnz <- 10000
L <- sparseMatrix(i = sample.int(n, nnz, replace=TRUE),
                  j = sample.int(p, nnz, replace=TRUE),
                  x = rnorm(nnz))
(st1 <- system.time(r1 <- rankMatrix(L)))                # warning+ ~1.5 sec (2013)
(st2 <- system.time(r2 <- rankMatrix(L, method = "qr"))) # considerably faster!
r1[[1]] == print(r2[[1]]) ## -->  ( 33  TRUE )
## Don't show: 
stopifnot(r1[[1]] == 33, 33 == r2[[1]])
if(interactive() || nzchar(Sys.getenv("R_MATRIX_CHECK_EXTRA")))
    stopifnot(st2[[1]] < 0.2) # seeing 0.03 (on ~ 2010-hardware; R 3.0.2)
## End(Don't show)
## another sparse-"qr" one, which ``failed'' till 2013-11-23:
set.seed(42)
f1 <- factor(sample(50, 1000, replace=TRUE))
f2 <- factor(sample(50, 1000, replace=TRUE))
f3 <- factor(sample(50, 1000, replace=TRUE))
D <- t(do.call(rbind, lapply(list(f1,f2,f3), as, 'sparseMatrix')))
dim(D); nnzero(D) ## 1000 x 150 // 3000 non-zeros (= 2%)
stopifnot(rankMatrix(D,           method='qr') == 148,
	  rankMatrix(crossprod(D),method='qr') == 148)

## zero matrix has rank 0 :
stopifnot(sapply(meths, function(.m.)
                        rankMatrix(matrix(0, 2, 2), method = .m.)) == 0)



cleanEx()
nameEx("rcond")
### * rcond

flush(stderr()); flush(stdout())

### Name: rcond-methods
### Title: Estimate the Reciprocal Condition Number
### Aliases: rcond rcond-methods rcond,ANY,missing-method
###   rcond,denseMatrix,character-method
###   rcond,diagonalMatrix,character-method
###   rcond,indMatrix,character-method rcond,pMatrix,character-method
###   rcond,sparseMatrix,character-method
### Keywords: algebra math methods

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
x <- Matrix(rnorm(9), 3, 3)
rcond(x)
## typically "the same" (with more computational effort):
1 / (norm(x) * norm(solve(x)))
rcond(Hilbert(9))  # should be about 9.1e-13

## For non-square matrices:
rcond(x1 <- cbind(1,1:10))# 0.05278
rcond(x2 <- cbind(x1, 2:11))# practically 0, since x2 does not have full rank

## sparse
(S1 <- Matrix(rbind(0:1,0, diag(3:-2))))
rcond(S1)
m1 <- as(S1, "denseMatrix")
all.equal(rcond(S1), rcond(m1))

## wide and sparse
rcond(Matrix(cbind(0, diag(2:-1))))

## Large sparse example ----------
m <- Matrix(c(3,0:2), 2,2)
M <- bdiag(kronecker(Diagonal(2), m), kronecker(m,m))
36*(iM <- solve(M)) # still sparse
MM <- kronecker(Diagonal(10), kronecker(Diagonal(5),kronecker(m,M)))
dim(M3 <- kronecker(bdiag(M,M),MM)) # 12'800 ^ 2
if(interactive()) ## takes about 2 seconds if you have >= 8 GB RAM
  system.time(r <- rcond(M3))
## whereas this is *fast* even though it computes  solve(M3)
system.time(r. <- rcond(M3, useInv=TRUE))
if(interactive()) ## the values are not the same
  c(r, r.)  # 0.05555 0.013888
## for all 4 norms available for sparseMatrix :
cbind(rr <- sapply(c("1","I","F","M"),
             function(N) rcond(M3, norm=N, useInv=TRUE)))
## Don't show: 
stopifnot(all.equal(r., 1/72, tolerance=1e-12))
## End(Don't show)



cleanEx()
nameEx("rep2abI")
### * rep2abI

flush(stderr()); flush(stdout())

### Name: rep2abI
### Title: Replicate Vectors into 'abIndex' Result
### Aliases: rep2abI
### Keywords: manip utilities

### ** Examples

(ab <- rep2abI(2:7, 4))
stopifnot(identical(as(ab, "numeric"),
	   rep(2:7, 4)))



cleanEx()
nameEx("replValue-class")
### * replValue-class

flush(stderr()); flush(stdout())

### Name: replValue-class
### Title: Virtual Class "replValue" - Simple Class for Subassignment
###   Values
### Aliases: replValue-class
### Keywords: classes

### ** Examples

showClass("replValue")



cleanEx()
nameEx("rleDiff-class")
### * rleDiff-class

flush(stderr()); flush(stdout())

### Name: rleDiff-class
### Title: Class "rleDiff" of rle(diff(.)) Stored Vectors
### Aliases: rleDiff-class show,rleDiff-method
### Keywords: classes

### ** Examples

showClass("rleDiff")

ab <- c(abIseq(2, 100), abIseq(20, -2))
ab@rleD  # is "rleDiff"



cleanEx()
nameEx("rsparsematrix")
### * rsparsematrix

flush(stderr()); flush(stdout())

### Name: rsparsematrix
### Title: Random Sparse Matrix
### Aliases: rsparsematrix
### Keywords: array distribution utilities

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
set.seed(17)# to be reproducible
M <- rsparsematrix(8, 12, nnz = 30) # small example, not very sparse
M
M1 <- rsparsematrix(1000, 20,  nnz = 123,  rand.x = runif)
summary(M1)

## a random *symmetric* Matrix
(S9 <- rsparsematrix(9, 9, nnz = 10, symmetric=TRUE)) # dsCMatrix
nnzero(S9)# ~ 20: as 'nnz' only counts one "triangle"

## a random patter*n* aka boolean Matrix (no 'x' slot):
(n7 <- rsparsematrix(5, 12, nnz = 10, rand.x = NULL))

## a [T]riplet representation sparseMatrix:
T2 <- rsparsematrix(40, 12, nnz = 99, repr = "T")
head(T2)



cleanEx()
nameEx("solve-methods")
### * solve-methods

flush(stderr()); flush(stdout())

### Name: solve-methods
### Title: Methods in Package 'Matrix' for Function 'solve'
### Aliases: solve solve-methods solve,ANY,ANY-method
###   solve,BunchKaufman,missing-method solve,BunchKaufman,dgeMatrix-method
###   solve,CHMfactor,missing-method solve,CHMfactor,dgeMatrix-method
###   solve,CHMfactor,dgCMatrix-method solve,Cholesky,missing-method
###   solve,Cholesky,dgeMatrix-method solve,CsparseMatrix,ANY-method
###   solve,Matrix,sparseVector-method
###   solve,MatrixFactorization,CsparseMatrix-method
###   solve,MatrixFactorization,RsparseMatrix-method
###   solve,MatrixFactorization,TsparseMatrix-method
###   solve,MatrixFactorization,denseMatrix-method
###   solve,MatrixFactorization,dgCMatrix-method
###   solve,MatrixFactorization,dgeMatrix-method
###   solve,MatrixFactorization,diagonalMatrix-method
###   solve,MatrixFactorization,indMatrix-method
###   solve,MatrixFactorization,matrix-method
###   solve,MatrixFactorization,sparseVector-method
###   solve,MatrixFactorization,vector-method
###   solve,RsparseMatrix,ANY-method solve,Schur,ANY-method
###   solve,TsparseMatrix,ANY-method solve,ddiMatrix,Matrix-method
###   solve,ddiMatrix,matrix-method solve,ddiMatrix,missing-method
###   solve,ddiMatrix,vector-method solve,denseLU,missing-method
###   solve,denseLU,dgeMatrix-method solve,denseMatrix,ANY-method
###   solve,dgCMatrix,denseMatrix-method solve,dgCMatrix,matrix-method
###   solve,dgCMatrix,missing-method solve,dgCMatrix,sparseMatrix-method
###   solve,dgCMatrix,vector-method solve,dgeMatrix,ANY-method
###   solve,diagonalMatrix,ANY-method solve,dpoMatrix,ANY-method
###   solve,dppMatrix,ANY-method solve,dsCMatrix,denseMatrix-method
###   solve,dsCMatrix,matrix-method solve,dsCMatrix,missing-method
###   solve,dsCMatrix,sparseMatrix-method solve,dsCMatrix,vector-method
###   solve,dspMatrix,ANY-method solve,dsyMatrix,ANY-method
###   solve,dtCMatrix,dgCMatrix-method solve,dtCMatrix,dgeMatrix-method
###   solve,dtCMatrix,missing-method
###   solve,dtCMatrix,triangularMatrix-method
###   solve,dtpMatrix,dgeMatrix-method solve,dtpMatrix,missing-method
###   solve,dtpMatrix,triangularMatrix-method
###   solve,dtrMatrix,dgeMatrix-method solve,dtrMatrix,missing-method
###   solve,dtrMatrix,triangularMatrix-method solve,indMatrix,ANY-method
###   solve,matrix,Matrix-method solve,matrix,sparseVector-method
###   solve,pBunchKaufman,missing-method
###   solve,pBunchKaufman,dgeMatrix-method solve,pCholesky,missing-method
###   solve,pCholesky,dgeMatrix-method solve,pMatrix,Matrix-method
###   solve,pMatrix,matrix-method solve,pMatrix,missing-method
###   solve,pMatrix,vector-method solve,sparseLU,missing-method
###   solve,sparseLU,dgeMatrix-method solve,sparseLU,dgCMatrix-method
###   solve,sparseQR,missing-method solve,sparseQR,dgeMatrix-method
###   solve,sparseQR,dgCMatrix-method
###   solve,triangularMatrix,CsparseMatrix-method
###   solve,triangularMatrix,RsparseMatrix-method
###   solve,triangularMatrix,TsparseMatrix-method
###   solve,triangularMatrix,denseMatrix-method
###   solve,triangularMatrix,dgCMatrix-method
###   solve,triangularMatrix,dgeMatrix-method
###   solve,triangularMatrix,diagonalMatrix-method
###   solve,triangularMatrix,indMatrix-method
###   solve,triangularMatrix,matrix-method
###   solve,triangularMatrix,vector-method
### Keywords: algebra array methods

### ** Examples

## A close to symmetric example with "quite sparse" inverse:
n1 <- 7; n2 <- 3
dd <- data.frame(a = gl(n1,n2), b = gl(n2,1,n1*n2))# balanced 2-way
X <- sparse.model.matrix(~ -1+ a + b, dd)# no intercept --> even sparser
XXt <- tcrossprod(X)
diag(XXt) <- rep(c(0,0,1,0), length.out = nrow(XXt))

n <- nrow(ZZ <- kronecker(XXt, Diagonal(x=c(4,1))))
image(a <- 2*Diagonal(n) + ZZ %*% Diagonal(x=c(10, rep(1, n-1))))
isSymmetric(a) # FALSE
image(drop0(skewpart(a)))
image(ia0 <- solve(a, tol = 0)) # checker board, dense [but really, a is singular!]
try(solve(a, sparse=TRUE))##-> error [ TODO: assertError ]
ia. <- solve(a, sparse=TRUE, tol = 1e-19)##-> *no* error
if(R.version$arch == "x86_64")
  ## Fails on 32-bit [Fedora 19, R 3.0.2] from Matrix 1.1-0 on [FIXME ??] only
  stopifnot(all.equal(as.matrix(ia.), as.matrix(ia0)))
a <- a + Diagonal(n)
iad <- solve(a)
ias <- solve(a, sparse=FALSE)
stopifnot(all.equal(as(iad,"denseMatrix"), ias, tolerance=1e-14))
I. <- iad %*% a          ; image(I.)
I0 <- drop0(zapsmall(I.)); image(I0)
.I <- a %*% iad
.I0 <- drop0(zapsmall(.I))
stopifnot( all.equal(as(I0, "diagonalMatrix"), Diagonal(n)),
           all.equal(as(.I0,"diagonalMatrix"), Diagonal(n)) )




cleanEx()
nameEx("spMatrix")
### * spMatrix

flush(stderr()); flush(stdout())

### Name: spMatrix
### Title: Sparse Matrix Constructor From Triplet
### Aliases: spMatrix
### Keywords: array utilities

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
## simple example
A <- spMatrix(10,20, i = c(1,3:8),
                     j = c(2,9,6:10),
                     x = 7 * (1:7))
A # a "dgTMatrix"
summary(A)
str(A) # note that *internally* 0-based indices (i,j) are used

L <- spMatrix(9, 30, i = rep(1:9, 3), 1:27,
              (1:27) %% 4 != 1)
L # an "lgTMatrix"


## A simplified predecessor of  Matrix'  rsparsematrix() function :

 rSpMatrix <- function(nrow, ncol, nnz,
                       rand.x = function(n) round(rnorm(nnz), 2))
 {
     ## Purpose: random sparse matrix
     ## --------------------------------------------------------------
     ## Arguments: (nrow,ncol): dimension
     ##          nnz  :  number of non-zero entries
     ##         rand.x:  random number generator for 'x' slot
     ## --------------------------------------------------------------
     ## Author: Martin Maechler, Date: 14.-16. May 2007
     stopifnot((nnz <- as.integer(nnz)) >= 0,
               nrow >= 0, ncol >= 0, nnz <= nrow * ncol)
     spMatrix(nrow, ncol,
              i = sample(nrow, nnz, replace = TRUE),
              j = sample(ncol, nnz, replace = TRUE),
              x = rand.x(nnz))
 }

 M1 <- rSpMatrix(100000, 20, nnz = 200)
 summary(M1)



cleanEx()
nameEx("sparse.model.matrix")
### * sparse.model.matrix

flush(stderr()); flush(stdout())

### Name: sparse.model.matrix
### Title: Construct Sparse Design / Model Matrices
### Aliases: sparse.model.matrix fac2sparse fac2Sparse
### Keywords: array models utilities

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
## End(Don't show)
dd <- data.frame(a = gl(3,4), b = gl(4,1,12))# balanced 2-way
options("contrasts") # the default:  "contr.treatment"
sparse.model.matrix(~ a + b, dd)
sparse.model.matrix(~ -1+ a + b, dd)# no intercept --> even sparser
sparse.model.matrix(~ a + b, dd, contrasts = list(a="contr.sum"))
sparse.model.matrix(~ a + b, dd, contrasts = list(b="contr.SAS"))

## Sparse method is equivalent to the traditional one :
stopifnot(all(sparse.model.matrix(~    a + b, dd) ==
	          Matrix(model.matrix(~    a + b, dd), sparse=TRUE)),
	      all(sparse.model.matrix(~0 + a + b, dd) ==
	          Matrix(model.matrix(~0 + a + b, dd), sparse=TRUE)))

(ff <- gl(3,4,, c("X","Y", "Z")))
fac2sparse(ff) #  3 x 12 sparse Matrix of class "dgCMatrix"
##
##  X  1 1 1 1 . . . . . . . .
##  Y  . . . . 1 1 1 1 . . . .
##  Z  . . . . . . . . 1 1 1 1

## can also be computed via sparse.model.matrix():
f30 <- gl(3,0    )
f12 <- gl(3,0, 12)
stopifnot(
  all.equal(t( fac2sparse(ff) ),
	    sparse.model.matrix(~ 0+ff),
	    tolerance = 0, check.attributes=FALSE),
  is(M <- fac2sparse(f30, drop= TRUE),"CsparseMatrix"), dim(M) == c(0, 0),
  is(M <- fac2sparse(f30, drop=FALSE),"CsparseMatrix"), dim(M) == c(3, 0),
  is(M <- fac2sparse(f12, drop= TRUE),"CsparseMatrix"), dim(M) == c(0,12),
  is(M <- fac2sparse(f12, drop=FALSE),"CsparseMatrix"), dim(M) == c(3,12)
 )



cleanEx()
nameEx("sparseLU-class")
### * sparseLU-class

flush(stderr()); flush(stdout())

### Name: sparseLU-class
### Title: Sparse LU Factorizations
### Aliases: sparseLU-class determinant,sparseLU,logical-method
### Keywords: algebra array classes

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("sparseLU")
set.seed(2)

A <- as(readMM(system.file("external", "pores_1.mtx", package = "Matrix")),
        "CsparseMatrix")
(n <- A@Dim[1L])

## With dimnames, to see that they are propagated :
dimnames(A) <- dn <- list(paste0("r", seq_len(n)),
                          paste0("c", seq_len(n)))

(lu.A <- lu(A))
str(e.lu.A <- expand2(lu.A), max.level = 2L)

ae1 <- function(a, b, ...) all.equal(as(a, "matrix"), as(b, "matrix"), ...)
ae2 <- function(a, b, ...) ae1(unname(a), unname(b), ...)

## A ~ P1' L U P2' in floating point
stopifnot(exprs = {
    identical(names(e.lu.A), c("P1.", "L", "U", "P2."))
    identical(e.lu.A[["P1."]],
              new("pMatrix", Dim = c(n, n), Dimnames = c(dn[1L], list(NULL)),
                  margin = 1L, perm = invertPerm(lu.A@p, 0L, 1L)))
    identical(e.lu.A[["P2."]],
              new("pMatrix", Dim = c(n, n), Dimnames = c(list(NULL), dn[2L]),
                  margin = 2L, perm = invertPerm(lu.A@q, 0L, 1L)))
    identical(e.lu.A[["L"]], lu.A@L)
    identical(e.lu.A[["U"]], lu.A@U)
    ae1(A, with(e.lu.A, P1. %*% L %*% U %*% P2.))
    ae2(A[lu.A@p + 1L, lu.A@q + 1L], with(e.lu.A, L %*% U))
})

## Factorization handled as factorized matrix
b <- rnorm(n)
stopifnot(identical(det(A), det(lu.A)),
          identical(solve(A, b), solve(lu.A, b)))



cleanEx()
nameEx("sparseMatrix-class")
### * sparseMatrix-class

flush(stderr()); flush(stdout())

### Name: sparseMatrix-class
### Title: Virtual Class "sparseMatrix" - Mother of Sparse Matrices
### Aliases: sparseMatrix-class -,sparseMatrix,missing-method
###   Math,sparseMatrix-method Ops,numeric,sparseMatrix-method
###   Ops,sparseMatrix,ddiMatrix-method Ops,sparseMatrix,ldiMatrix-method
###   Ops,sparseMatrix,nsparseMatrix-method Ops,sparseMatrix,numeric-method
###   Ops,sparseMatrix,sparseMatrix-method Summary,sparseMatrix-method
###   coerce,ANY,sparseMatrix-method coerce,factor,sparseMatrix-method
###   coerce,matrix,sparseMatrix-method coerce,vector,sparseMatrix-method
###   cov2cor,sparseMatrix-method diff,sparseMatrix-method
###   dim<-,sparseMatrix-method format,sparseMatrix-method
###   log,sparseMatrix-method mean,sparseMatrix-method
###   print,sparseMatrix-method rep,sparseMatrix-method
###   show,sparseMatrix-method summary,sparseMatrix-method
###   print.sparseMatrix
### Keywords: array classes

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("sparseMatrix") ## and look at the help() of its subclasses
M <- Matrix(0, 10000, 100)
M[1,1] <- M[2,3] <- 3.14
M  ## show(.) method suppresses printing of the majority of rows

data(CAex, package = "Matrix")
dim(CAex) # 72 x 72 matrix
determinant(CAex) # works via sparse lu(.)

## factor -> t( <sparse design matrix> ) :
(fact <- gl(5, 3, 30, labels = LETTERS[1:5]))
(Xt <- as(fact, "sparseMatrix"))  # indicator rows

## missing values --> all-0 columns:
f.mis <- fact
i.mis <- c(3:5, 17)
is.na(f.mis) <- i.mis
Xt != (X. <- as(f.mis, "sparseMatrix")) # differ only in columns 3:5,17
stopifnot(all(X.[,i.mis] == 0), all(Xt[,-i.mis] == X.[,-i.mis]))



cleanEx()
nameEx("sparseMatrix")
### * sparseMatrix

flush(stderr()); flush(stdout())

### Name: sparseMatrix
### Title: General Sparse Matrix Construction from Nonzero Entries
### Aliases: sparseMatrix
### Keywords: array utilities

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
## simple example
i <- c(1,3:8); j <- c(2,9,6:10); x <- 7 * (1:7)
(A <- sparseMatrix(i, j, x = x))                    ##  8 x 10 "dgCMatrix"
summary(A)
str(A) # note that *internally* 0-based row indices are used

(sA <- sparseMatrix(i, j, x = x, symmetric = TRUE)) ## 10 x 10 "dsCMatrix"
(tA <- sparseMatrix(i, j, x = x, triangular= TRUE)) ## 10 x 10 "dtCMatrix"
stopifnot( all(sA == tA + t(tA)) ,
           identical(sA, as(tA + t(tA), "symmetricMatrix")))

## dims can be larger than the maximum row or column indices
(AA <- sparseMatrix(c(1,3:8), c(2,9,6:10), x = 7 * (1:7), dims = c(10,20)))
summary(AA)

## i, j and x can be in an arbitrary order, as long as they are consistent
set.seed(1); (perm <- sample(1:7))
(A1 <- sparseMatrix(i[perm], j[perm], x = x[perm]))
stopifnot(identical(A, A1))

## The slots are 0-index based, so
try( sparseMatrix(i=A@i, p=A@p, x= seq_along(A@x)) )
## fails and you should say so: 1-indexing is FALSE:
     sparseMatrix(i=A@i, p=A@p, x= seq_along(A@x), index1 = FALSE)

## the (i,j) pairs can be repeated, in which case the x's are summed
(args <- data.frame(i = c(i, 1), j = c(j, 2), x = c(x, 2)))
(Aa <- do.call(sparseMatrix, args))
## explicitly ask for elimination of such duplicates, so
## that the last one is used:
(A. <- do.call(sparseMatrix, c(args, list(use.last.ij = TRUE))))
stopifnot(Aa[1,2] == 9, # 2+7 == 9
          A.[1,2] == 2) # 2 was *after* 7

## for a pattern matrix, of course there is no "summing":
(nA <- do.call(sparseMatrix, args[c("i","j")]))

dn <- list(LETTERS[1:3], letters[1:5])
## pointer vectors can be used, and the (i,x) slots are sorted if necessary:
m <- sparseMatrix(i = c(3,1, 3:2, 2:1), p= c(0:2, 4,4,6), x = 1:6, dimnames = dn)
m
str(m)
stopifnot(identical(dimnames(m), dn))

sparseMatrix(x = 2.72, i=1:3, j=2:4) # recycling x
sparseMatrix(x = TRUE, i=1:3, j=2:4) # recycling x, |--> "lgCMatrix"

## no 'x' --> patter*n* matrix:
(n <- sparseMatrix(i=1:6, j=rev(2:7)))# -> ngCMatrix

## an empty sparse matrix:
(e <- sparseMatrix(dims = c(4,6), i={}, j={}))

## a symmetric one:
(sy <- sparseMatrix(i= c(2,4,3:5), j= c(4,7:5,5), x = 1:5,
                    dims = c(7,7), symmetric=TRUE))
stopifnot(isSymmetric(sy),
          identical(sy, ## switch i <-> j {and transpose }
    t( sparseMatrix(j= c(2,4,3:5), i= c(4,7:5,5), x = 1:5,
                    dims = c(7,7), symmetric=TRUE))))

## rsparsematrix() calls sparseMatrix() :
M1 <- rsparsematrix(1000, 20, nnz = 200)
summary(M1)

## pointers example in converting from other sparse matrix representations.
if(requireNamespace("SparseM") &&
   packageVersion("SparseM") >= "0.87" &&
   nzchar(dfil <- system.file("extdata", "rua_32_ax.rua", package = "SparseM"))) {
  X <- SparseM::model.matrix(SparseM::read.matrix.hb(dfil))
  XX <- sparseMatrix(j = X@ja, p = X@ia - 1L, x = X@ra, dims = X@dimension)
  validObject(XX)

  ## Alternatively, and even more user friendly :
  X. <- as(X, "Matrix")  # or also
  X2 <- as(X, "sparseMatrix")
  stopifnot(identical(XX, X.), identical(X., X2))
}



cleanEx()
nameEx("sparseQR-class")
### * sparseQR-class

flush(stderr()); flush(stdout())

### Name: sparseQR-class
### Title: Sparse QR Factorizations
### Aliases: sparseQR-class determinant,sparseQR,logical-method
###   qr.Q,sparseQR-method qr.R,sparseQR-method qr.X,sparseQR-method
###   qr.coef,sparseQR,Matrix-method qr.coef,sparseQR,dgeMatrix-method
###   qr.coef,sparseQR,matrix-method qr.coef,sparseQR,vector-method
###   qr.fitted,sparseQR,Matrix-method qr.fitted,sparseQR,dgeMatrix-method
###   qr.fitted,sparseQR,matrix-method qr.fitted,sparseQR,vector-method
###   qr.qty,sparseQR,Matrix-method qr.qty,sparseQR,dgeMatrix-method
###   qr.qty,sparseQR,matrix-method qr.qty,sparseQR,vector-method
###   qr.qy,sparseQR,Matrix-method qr.qy,sparseQR,dgeMatrix-method
###   qr.qy,sparseQR,matrix-method qr.qy,sparseQR,vector-method
###   qr.resid,sparseQR,Matrix-method qr.resid,sparseQR,dgeMatrix-method
###   qr.resid,sparseQR,matrix-method qr.resid,sparseQR,vector-method qrR
### Keywords: algebra array classes utilities

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showClass("sparseQR")
set.seed(2)

m <- 300L
n <- 60L
A <- rsparsematrix(m, n, 0.05)

## With dimnames, to see that they are propagated :
dimnames(A) <- dn <- list(paste0("r", seq_len(m)),
                          paste0("c", seq_len(n)))

(qr.A <- qr(A))
str(e.qr.A <- expand2(qr.A, complete = FALSE), max.level = 2L)
str(E.qr.A <- expand2(qr.A, complete =  TRUE), max.level = 2L)

t(sapply(e.qr.A, dim))
t(sapply(E.qr.A, dim))

## Horribly inefficient, but instructive :
slowQ <- function(V, beta) {
    d <- dim(V)
    Q <- diag(d[1L])
    if(d[2L] > 0L) {
        for(j in d[2L]:1L) {
            cat(j, "\n", sep = "")
            Q <- Q - (beta[j] * tcrossprod(V[, j])) %*% Q
        }
    }
    Q
}

ae1 <- function(a, b, ...) all.equal(as(a, "matrix"), as(b, "matrix"), ...)
ae2 <- function(a, b, ...) ae1(unname(a), unname(b), ...)

## A ~ P1' Q R P2' ~ P1' Q1 R1 P2' in floating point
stopifnot(exprs = {
    identical(names(e.qr.A), c("P1.", "Q1", "R1", "P2."))
    identical(names(E.qr.A), c("P1.", "Q" , "R" , "P2."))
    identical(e.qr.A[["P1."]],
              new("pMatrix", Dim = c(m, m), Dimnames = c(dn[1L], list(NULL)),
                  margin = 1L, perm = invertPerm(qr.A@p, 0L, 1L)))
    identical(e.qr.A[["P2."]],
              new("pMatrix", Dim = c(n, n), Dimnames = c(list(NULL), dn[2L]),
                  margin = 2L, perm = invertPerm(qr.A@q, 0L, 1L)))
    identical(e.qr.A[["R1"]], triu(E.qr.A[["R"]][seq_len(n), ]))
    identical(e.qr.A[["Q1"]],      E.qr.A[["Q"]][, seq_len(n)] )
    identical(E.qr.A[["R"]], qr.A@R)
 ## ae1(E.qr.A[["Q"]], slowQ(qr.A@V, qr.A@beta))
    ae1(crossprod(E.qr.A[["Q"]]), diag(m))
    ae1(A, with(e.qr.A, P1. %*% Q1 %*% R1 %*% P2.))
    ae1(A, with(E.qr.A, P1. %*% Q  %*% R  %*% P2.))
    ae2(A.perm <- A[qr.A@p + 1L, qr.A@q + 1L], with(e.qr.A, Q1 %*% R1))
    ae2(A.perm                               , with(E.qr.A, Q  %*% R ))
})

## More identities
b <- rnorm(m)
stopifnot(exprs = {
    ae1(qrX <- qr.X     (qr.A   ), A)
    ae2(qrQ <- qr.Q     (qr.A   ), with(e.qr.A, P1. %*% Q1))
    ae2(       qr.R     (qr.A   ), with(e.qr.A, R1))
    ae2(qrc <- qr.coef  (qr.A, b), with(e.qr.A, solve(R1 %*% P2., t(qrQ)) %*% b))
    ae2(qrf <- qr.fitted(qr.A, b), with(e.qr.A, tcrossprod(qrQ) %*% b))
    ae2(qrr <- qr.resid (qr.A, b), b - qrf)
    ae2(qrq <- qr.qy    (qr.A, b), with(E.qr.A, P1. %*% Q %*% b))
    ae2(qr.qty(qr.A, qrq), b)
})

## Sparse and dense computations should agree here
qr.Am <- qr(as(A, "matrix")) # <=> qr.default(A)
stopifnot(exprs = {
    ae2(qrX, qr.X     (qr.Am   ))
    ae2(qrc, qr.coef  (qr.Am, b))
    ae2(qrf, qr.fitted(qr.Am, b))
    ae2(qrr, qr.resid (qr.Am, b))
})



cleanEx()
nameEx("sparseVector-class")
### * sparseVector-class

flush(stderr()); flush(stdout())

### Name: sparseVector-class
### Title: Sparse Vector Classes
### Aliases: sparseVector-class nsparseVector-class lsparseVector-class
###   isparseVector-class dsparseVector-class zsparseVector-class
###   !,sparseVector-method Arith,sparseVector,ddenseMatrix-method
###   Arith,sparseVector,dgeMatrix-method
###   Arith,sparseVector,sparseVector-method
###   Logic,sparseVector,dMatrix-method Logic,sparseVector,lMatrix-method
###   Logic,sparseVector,nMatrix-method
###   Logic,sparseVector,sparseVector-method Math,sparseVector-method
###   Math2,sparseVector-method Ops,ANY,sparseVector-method
###   Ops,sparseVector,ANY-method Ops,sparseVector,Matrix-method
###   Ops,sparseVector,atomicVector-method
###   Ops,sparseVector,sparseVector-method Summary,sparseVector-method
###   as.array,sparseVector-method as.complex,sparseVector-method
###   as.integer,sparseVector-method as.logical,sparseVector-method
###   as.matrix,sparseVector-method as.numeric,sparseVector-method
###   as.vector,sparseVector-method coerce,ANY,sparseVector-method
###   coerce,matrix,sparseVector-method
###   coerce,sparseVector,CsparseMatrix-method
###   coerce,sparseVector,Matrix-method
###   coerce,sparseVector,RsparseMatrix-method
###   coerce,sparseVector,TsparseMatrix-method
###   coerce,sparseVector,denseMatrix-method
###   coerce,sparseVector,dsparseVector-method
###   coerce,sparseVector,generalMatrix-method
###   coerce,sparseVector,isparseVector-method
###   coerce,sparseVector,lsparseVector-method
###   coerce,sparseVector,nsparseVector-method
###   coerce,sparseVector,sparseMatrix-method
###   coerce,sparseVector,unpackedMatrix-method
###   coerce,sparseVector,zsparseVector-method
###   coerce,vector,dsparseVector-method coerce,vector,isparseVector-method
###   coerce,vector,lsparseVector-method coerce,vector,nsparseVector-method
###   coerce,vector,sparseVector-method coerce,vector,zsparseVector-method
###   diff,sparseVector-method dim<-,sparseVector-method
###   head,sparseVector-method initialize,sparseVector-method
###   length,sparseVector-method log,sparseVector-method
###   mean,sparseVector-method rep,sparseVector-method
###   show,sparseVector-method sort,sparseVector-method
###   t,sparseVector-method tail,sparseVector-method
###   toeplitz,sparseVector-method zapsmall,sparseVector-method
###   !,nsparseVector-method which,nsparseVector-method
###   !,lsparseVector-method Logic,lsparseVector,lsparseVector-method
###   which,lsparseVector-method -,dsparseVector,missing-method
###   Arith,dsparseVector,dsparseVector-method c.sparseVector
### Keywords: classes manip

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
getClass("sparseVector")
getClass("dsparseVector")

sx <- c(0,0,3, 3.2, 0,0,0,-3:1,0,0,2,0,0,5,0,0)
(ss <- as(sx, "sparseVector"))

ix <- as.integer(round(sx))
(is <- as(ix, "sparseVector")) ## an "isparseVector" (!)
(ns <- sparseVector(i= c(7, 3, 2), length = 10)) # "nsparseVector"
## rep() works too:
(ri <- rep(is, length.out= 25))

## Using `dim<-`  as in base R :
r <- ss
dim(r) <- c(4,5) # becomes a sparse Matrix:
r
## or coercion (as as.matrix() in base R):
as(ss, "Matrix")
stopifnot(all(ss == print(as(ss, "CsparseMatrix"))))

## currently has "non-structural" FALSE -- printing as ":"
(lis <- is & FALSE)
(nn <- is[is == 0]) # all "structural" FALSE

## NA-case
sN <- sx; sN[4] <- NA
(svN <- as(sN, "sparseVector"))

v <- as(c(0,0,3, 3.2, rep(0,9),-3,0,-1, rep(0,20),5,0),
         "sparseVector")
v <- rep(rep(v, 50), 5000)
set.seed(1); v[sample(v@i, 1e6)] <- 0
str(v)
system.time(for(i in 1:4) hv <- head(v, 1e6))
##   user  system elapsed
##  0.033   0.000   0.032
system.time(for(i in 1:4) h2 <- v[1:1e6])
##   user  system elapsed
##  1.317   0.000   1.319

stopifnot(identical(hv, h2),
          identical(is | FALSE, is != 0),
          validObject(svN), validObject(lis), as.logical(is.na(svN[4])),
          identical(is^2 > 0, is & TRUE),
          all(!lis), !any(lis), length(nn@i) == 0, !any(nn), all(!nn),
          sum(lis) == 0, !prod(lis), range(lis) == c(0,0))

## create and use the t(.) method:
t(x20 <- sparseVector(c(9,3:1), i=c(1:2,4,7), length=20))
(T20 <- toeplitz(x20))
stopifnot(is(T20, "symmetricMatrix"), is(T20, "sparseMatrix"),
          identical(unname(as.matrix(T20)),
                    toeplitz(as.vector(x20))))

## c() method for "sparseVector" - also available as regular function
(c1 <- c(x20, 0,0,0, -10*x20))
(c2 <- c(ns, is, FALSE))
(c3 <- c(ns, !ns, TRUE, NA, FALSE))
(c4 <- c(ns, rev(ns)))
## here, c() would produce a list {not dispatching to c.sparseVector()}
(c5 <- c.sparseVector(0,0, x20))

## checking (consistency)
.v <- as.vector
.s <- function(v) as(v, "sparseVector")
stopifnot(exprs = {
    all.equal(c1, .s(c(.v(x20), 0,0,0, -10*.v(x20))),      tol = 0)
    all.equal(c2, .s(c(.v(ns), .v(is), FALSE)),            tol = 0)
    all.equal(c3, .s(c(.v(ns), !.v(ns), TRUE, NA, FALSE)), tol = 0)
    all.equal(c4, .s(c(.v(ns), rev(.v(ns)))),              tol = 0,
              check.class = FALSE)
    all.equal(c5, .s(c(0,0, .v(x20))),                     tol = 0)
})



cleanEx()
nameEx("sparseVector")
### * sparseVector

flush(stderr()); flush(stdout())

### Name: sparseVector
### Title: Sparse Vector Construction from Nonzero Entries
### Aliases: sparseVector
### Keywords: utilities

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
str(sv <- sparseVector(x = 1:10, i = sample(999, 10), length=1000))

sx <- c(0,0,3, 3.2, 0,0,0,-3:1,0,0,2,0,0,5,0,0)
ss <- as(sx, "sparseVector")
stopifnot(identical(ss,
   sparseVector(x = c(2, -1, -2, 3, 1, -3, 5, 3.2),
                i = c(15L, 10:9, 3L,12L,8L,18L, 4L), length = 20L)))

(ns <- sparseVector(i= c(7, 3, 2), length = 10))
stopifnot(identical(ns,
      new("nsparseVector", length = 10, i = c(2, 3, 7))))



cleanEx()
nameEx("symmetricMatrix-class")
### * symmetricMatrix-class

flush(stderr()); flush(stdout())

### Name: symmetricMatrix-class
### Title: Virtual Class of Symmetric Matrices in Package Matrix
### Aliases: symmetricMatrix-class coerce,matrix,symmetricMatrix-method
###   dimnames,symmetricMatrix-method
### Keywords: array classes

### ** Examples

## An example about the symmetric Dimnames:
sy <- sparseMatrix(i= c(2,4,3:5), j= c(4,7:5,5), x = 1:5, dims = c(7,7),
                   symmetric=TRUE, dimnames = list(NULL, letters[1:7]))
sy # shows symmetrical dimnames
sy@Dimnames  # internally only one part is stored
dimnames(sy) # both parts - as sy *is* symmetrical
## Don't show: 
local({ nm <- letters[1:7]
  stopifnot(identical(dimnames(sy), list(  nm, nm)),
	    identical(sy@Dimnames , list(NULL, nm)))
})
## End(Don't show)
showClass("symmetricMatrix")

## The names of direct subclasses:
scl <- getClass("symmetricMatrix")@subclasses
directly <- sapply(lapply(scl, slot, "by"), length) == 0
names(scl)[directly]

## Methods -- applicaple to all subclasses above:
showMethods(classes = "symmetricMatrix")



cleanEx()
nameEx("symmpart")
### * symmpart

flush(stderr()); flush(stdout())

### Name: symmpart-methods
### Title: Symmetric Part and Skew(symmetric) Part of a Matrix
### Aliases: symmpart symmpart-methods skewpart skewpart-methods
###   symmpart,CsparseMatrix-method symmpart,RsparseMatrix-method
###   symmpart,TsparseMatrix-method symmpart,denseMatrix-method
###   symmpart,diagonalMatrix-method symmpart,indMatrix-method
###   symmpart,matrix-method skewpart,CsparseMatrix-method
###   skewpart,RsparseMatrix-method skewpart,TsparseMatrix-method
###   skewpart,denseMatrix-method skewpart,diagonalMatrix-method
###   skewpart,indMatrix-method skewpart,matrix-method
### Keywords: algebra arith array methods

### ** Examples

m <- Matrix(1:4, 2,2)
symmpart(m)
skewpart(m)

stopifnot(all(m == symmpart(m) + skewpart(m)))

dn <- dimnames(m) <- list(row = c("r1", "r2"), col = c("var.1", "var.2"))
stopifnot(all(m == symmpart(m) + skewpart(m)))
colnames(m) <- NULL
stopifnot(all(m == symmpart(m) + skewpart(m)))
dimnames(m) <- unname(dn)
stopifnot(all(m == symmpart(m) + skewpart(m)))


## investigate the current methods:
showMethods(skewpart, include = TRUE)



cleanEx()
nameEx("triangularMatrix-class")
### * triangularMatrix-class

flush(stderr()); flush(stdout())

### Name: triangularMatrix-class
### Title: Virtual Class of Triangular Matrices in Package Matrix
### Aliases: triangularMatrix-class
###   Arith,triangularMatrix,diagonalMatrix-method
###   Compare,triangularMatrix,diagonalMatrix-method
###   Logic,triangularMatrix,diagonalMatrix-method
###   coerce,matrix,triangularMatrix-method
###   determinant,triangularMatrix,logical-method
### Keywords: array classes

### ** Examples

showClass("triangularMatrix")

## The names of direct subclasses:
scl <- getClass("triangularMatrix")@subclasses
directly <- sapply(lapply(scl, slot, "by"), length) == 0
names(scl)[directly]

(m <- matrix(c(5,1,0,3), 2))
as(m, "triangularMatrix")



cleanEx()
nameEx("uniqTsparse")
### * uniqTsparse

flush(stderr()); flush(stdout())

### Name: asUniqueT
### Title: Standardize a Sparse Matrix in Triplet Format
### Aliases: anyDuplicatedT isUniqueT asUniqueT aggregateT uniqTsparse
### Keywords: array logic manip utilities

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
example("dgTMatrix-class", echo=FALSE)
## -> 'T2'  with (i,j,x) slots of length 5 each
T2u <- asUniqueT(T2)
stopifnot(## They "are" the same (and print the same):
          all.equal(T2, T2u, tol=0),
          ## but not internally:
          anyDuplicatedT(T2)  == 2,
          anyDuplicatedT(T2u) == 0,
          length(T2 @x) == 5,
          length(T2u@x) == 3)

isUniqueT(T2 ) # FALSE
isUniqueT(T2u) # TRUE

T3 <- T2u
T3[1, c(1,3)] <- 10; T3[2, c(1,5)] <- 20
T3u <- asUniqueT(T3)
str(T3u) # sorted in 'j', and within j, sorted in i
stopifnot(isUniqueT(T3u))

## Logical l.TMatrix and n.TMatrix :
(L2 <- T2 > 0)
validObject(L2u <- asUniqueT(L2))
(N2 <- as(L2, "nMatrix"))
validObject(N2u <- asUniqueT(N2))
stopifnot(N2u@i == L2u@i, L2u@i == T2u@i,  N2@i == L2@i, L2@i == T2@i,
          N2u@j == L2u@j, L2u@j == T2u@j,  N2@j == L2@j, L2@j == T2@j)
# now with a nasty NA  [partly failed in Matrix 1.1-5]:
L.0N <- L.1N <- L2
L.0N@x[1:2] <- c(FALSE, NA)
L.1N@x[1:2] <- c(TRUE, NA)
validObject(L.0N)
validObject(L.1N)
(m.0N <- as.matrix(L.0N))
(m.1N <- as.matrix(L.1N))
stopifnot(identical(10L, which(is.na(m.0N))), !anyNA(m.1N))
symnum(m.0N)
symnum(m.1N)



cleanEx()
nameEx("unpack")
### * unpack

flush(stderr()); flush(stdout())

### Name: pack
### Title: Representation of Packed and Unpacked Dense Matrices
### Aliases: pack pack-methods unpack unpack-methods pack,dgeMatrix-method
###   pack,lgeMatrix-method pack,matrix-method pack,ngeMatrix-method
###   pack,packedMatrix-method pack,sparseMatrix-method
###   pack,unpackedMatrix-method unpack,matrix-method
###   unpack,packedMatrix-method unpack,sparseMatrix-method
###   unpack,unpackedMatrix-method
### Keywords: array methods

### ** Examples

## Don't show: 
 
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
showMethods("pack")
(s <- crossprod(matrix(sample(15), 5,3))) # traditional symmetric matrix
(sp <- pack(s))
mt <- as.matrix(tt <- tril(s))
(pt <- pack(mt))
stopifnot(identical(pt, pack(tt)),
	  dim(s ) == dim(sp), all(s  == sp),
	  dim(mt) == dim(pt), all(mt == pt), all(mt == tt))

showMethods("unpack")
(cp4 <- chol(Hilbert(4))) # is triangular
tp4 <- pack(cp4) # [t]riangular [p]acked
str(tp4)
(unpack(tp4))
stopifnot(identical(tp4, pack(unpack(tp4))))

z1 <- new("dsyMatrix", Dim = c(2L, 2L), x = as.double(1:4), uplo = "U")
z2 <- unpack(pack(z1))
stopifnot(!identical(z1, z2), # _not_ identical
          all(z1 == z2)) # but mathematically equal
cbind(z1@x, z2@x) # (unused!) lower triangle is "lost" in translation



cleanEx()
nameEx("unpackedMatrix-class")
### * unpackedMatrix-class

flush(stderr()); flush(stdout())

### Name: unpackedMatrix-class
### Title: Virtual Class '"unpackedMatrix"' of Unpacked Dense Matrices
### Aliases: unpackedMatrix-class coerce,matrix,unpackedMatrix-method
###   coerce,vector,unpackedMatrix-method cov2cor,unpackedMatrix-method
### Keywords: array classes

### ** Examples

showClass("unpackedMatrix")
showMethods(classes = "unpackedMatrix")



cleanEx()
nameEx("unused-classes")
### * unused-classes

flush(stderr()); flush(stdout())

### Name: Matrix-notyet
### Title: Virtual Classes Not Yet Really Implemented and Used
### Aliases: Matrix-notyet iMatrix-class zMatrix-class
### Keywords: array classes

### ** Examples

showClass("iMatrix")
showClass("zMatrix")



cleanEx()
nameEx("updown")
### * updown

flush(stderr()); flush(stdout())

### Name: updown-methods
### Title: Updating and Downdating Sparse Cholesky Factorizations
### Aliases: updown updown-methods updown,character,ANY,ANY-method
###   updown,logical,Matrix,CHMfactor-method
###   updown,logical,dgCMatrix,CHMfactor-method
###   updown,logical,dsCMatrix,CHMfactor-method
###   updown,logical,dtCMatrix,CHMfactor-method
###   updown,logical,matrix,CHMfactor-method
### Keywords: algebra array methods

### ** Examples

m <- sparseMatrix(i = c(3, 1, 3:2, 2:1), p = c(0:2, 4, 4, 6), x = 1:6,
                  dimnames = list(LETTERS[1:3], letters[1:5]))
uc0 <- Cholesky(A <- crossprod(m) + Diagonal(5))
uc1 <- updown("+", Diagonal(5, 1), uc0)
uc2 <- updown("-", Diagonal(5, 1), uc1)
stopifnot(all.equal(uc0, uc2))
## Don't show: 
if(FALSE) {
## Hmm: this loses positive definiteness:
uc2 <- updown("-", Diagonal(5, 2), uc0)
image(show(as(uc0, "CsparseMatrix")))
image(show(as(uc2, "CsparseMatrix"))) # severely negative entries
}
## End(Don't show)



cleanEx()
nameEx("wrld_1deg")
### * wrld_1deg

flush(stderr()); flush(stdout())

### Name: wrld_1deg
### Title: Contiguity Matrix of World One-Degree Grid Cells
### Aliases: wrld_1deg
### Keywords: datasets

### ** Examples

## Don't show: 
 
library(stats, pos = "package:base", verbose = FALSE)
library(utils, pos = "package:base", verbose = FALSE)
## End(Don't show)
data(wrld_1deg, package = "Matrix")
(n <- ncol(wrld_1deg))
I <- .symDiagonal(n)

doExtras <- interactive() || nzchar(Sys.getenv("R_MATRIX_CHECK_EXTRA"))
set.seed(1)
r <- if(doExtras) 20L else 3L
rho <- 1 / runif(r, 0, 0.5)

system.time(MJ0 <- sapply(rho, function(mult)
    determinant(wrld_1deg + mult * I, logarithm = TRUE)$modulus))

## Can be done faster by updating the Cholesky factor:

C1 <- Cholesky(wrld_1deg, Imult = 2)
system.time(MJ1 <- sapply(rho, function(mult)
    determinant(update(C1, wrld_1deg, mult), sqrt = FALSE)$modulus))
stopifnot(all.equal(MJ0, MJ1))

C2 <- Cholesky(wrld_1deg, super = TRUE, Imult = 2)
system.time(MJ2 <- sapply(rho, function(mult)
    determinant(update(C2, wrld_1deg, mult), sqrt = FALSE)$modulus))
stopifnot(all.equal(MJ0, MJ2))



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
