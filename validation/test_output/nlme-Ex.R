pkgname <- "nlme"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('nlme')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("ACF")
### * ACF

flush(stderr()); flush(stdout())

### Name: ACF
### Title: Autocorrelation Function
### Aliases: ACF
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("ACF.gls")
### * ACF.gls

flush(stderr()); flush(stdout())

### Name: ACF.gls
### Title: Autocorrelation Function for gls Residuals
### Aliases: ACF.gls
### Keywords: models

### ** Examples

fm1 <- gls(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary)
ACF(fm1, form = ~ 1 | Mare)

# Pinheiro and Bates, p. 255-257
fm1Dial.gls <- gls(rate ~
  (pressure+I(pressure^2)+I(pressure^3)+I(pressure^4))*QB,
                   Dialyzer)

fm2Dial.gls <- update(fm1Dial.gls,
                 weights = varPower(form = ~ pressure))

ACF(fm2Dial.gls, form = ~ 1 | Subject)



cleanEx()
nameEx("ACF.lme")
### * ACF.lme

flush(stderr()); flush(stdout())

### Name: ACF.lme
### Title: Autocorrelation Function for lme Residuals
### Aliases: ACF.lme
### Keywords: models

### ** Examples

fm1 <- lme(follicles ~ sin(2*pi*Time) + cos(2*pi*Time),
           Ovary, random = ~ sin(2*pi*Time) | Mare)
ACF(fm1, maxLag = 11)

# Pinheiro and Bates, p240-241
fm1Over.lme <- lme(follicles  ~ sin(2*pi*Time) +
           cos(2*pi*Time), data=Ovary,
     random=pdDiag(~sin(2*pi*Time)) )
(ACF.fm1Over <- ACF(fm1Over.lme, maxLag=10))
plot(ACF.fm1Over, alpha=0.01) 



cleanEx()
nameEx("Cefamandole")
### * Cefamandole

flush(stderr()); flush(stdout())

### Name: Cefamandole
### Title: Pharmacokinetics of Cefamandole
### Aliases: Cefamandole
### Keywords: datasets

### ** Examples

plot(Cefamandole)
fm1 <- nlsList(SSbiexp, data = Cefamandole)
summary(fm1)



cleanEx()
nameEx("Covariate")
### * Covariate

flush(stderr()); flush(stdout())

### Name: Covariate
### Title: Assign Covariate Values
### Aliases: covariate<-
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("Covariate.varFunc")
### * Covariate.varFunc

flush(stderr()); flush(stdout())

### Name: Covariate.varFunc
### Title: Assign varFunc Covariate
### Aliases: covariate<-.varFunc
### Keywords: models

### ** Examples

vf1 <- varPower(1.1, form = ~age)
covariate(vf1) <- Orthodont[["age"]]



cleanEx()
nameEx("Dim")
### * Dim

flush(stderr()); flush(stdout())

### Name: Dim
### Title: Extract Dimensions from an Object
### Aliases: Dim Dim.default
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("Dim.corSpatial")
### * Dim.corSpatial

flush(stderr()); flush(stdout())

### Name: Dim.corSpatial
### Title: Dimensions of a corSpatial Object
### Aliases: Dim.corSpatial
### Keywords: models

### ** Examples

Dim(corGaus(), getGroups(Orthodont))

cs1ARMA <- corARMA(0.4, form = ~ 1 | Subject, q = 1)
cs1ARMA <- Initialize(cs1ARMA, data = Orthodont)
Dim(cs1ARMA)



cleanEx()
nameEx("Dim.corStruct")
### * Dim.corStruct

flush(stderr()); flush(stdout())

### Name: Dim.corStruct
### Title: Dimensions of a corStruct Object
### Aliases: Dim.corStruct
### Keywords: models

### ** Examples

Dim(corAR1(), getGroups(Orthodont))



cleanEx()
nameEx("Dim.pdMat")
### * Dim.pdMat

flush(stderr()); flush(stdout())

### Name: Dim.pdMat
### Title: Dimensions of a pdMat Object
### Aliases: Dim.pdMat Dim.pdCompSymm Dim.pdDiag Dim.pdIdent Dim.pdNatural
###   Dim.pdSymm
### Keywords: models

### ** Examples

Dim(pdSymm(diag(3)))



cleanEx()
nameEx("Extract.pdMat")
### * Extract.pdMat

flush(stderr()); flush(stdout())

### Name: [.pdMat
### Title: Subscript a pdMat Object
### Aliases: [.pdMat [.pdBlocked [<-.pdMat
### Keywords: models

### ** Examples

pd1 <- pdSymm(diag(3))
pd1[1, , drop = FALSE]
pd1[1:2, 1:2] <- 3 * diag(2)



cleanEx()
nameEx("Initialize")
### * Initialize

flush(stderr()); flush(stdout())

### Name: Initialize
### Title: Initialize Object
### Aliases: Initialize
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("Initialize.corStruct")
### * Initialize.corStruct

flush(stderr()); flush(stdout())

### Name: Initialize.corStruct
### Title: Initialize corStruct Object
### Aliases: Initialize.corStruct Initialize.corAR1 Initialize.corARMA
###   Initialize.corCAR1 Initialize.corCompSymm Initialize.corHF
###   Initialize.corLin Initialize.corNatural Initialize.corSpatial
###   Initialize.corSpher Initialize.corSymm
### Keywords: models

### ** Examples

cs1 <- corAR1(form = ~ 1 | Subject)
cs1 <- Initialize(cs1, data = Orthodont)



cleanEx()
nameEx("Initialize.varFunc")
### * Initialize.varFunc

flush(stderr()); flush(stdout())

### Name: Initialize.varFunc
### Title: Initialize varFunc Object
### Aliases: Initialize.varFunc Initialize.varComb Initialize.varConstPower
###   Initialize.varConstProp Initialize.varExp Initialize.varFixed
###   Initialize.varIdent Initialize.varPower
### Keywords: models

### ** Examples

vf1 <- varPower( form = ~ age | Sex )
vf1 <- Initialize( vf1, Orthodont )



cleanEx()
nameEx("LDEsysMat")
### * LDEsysMat

flush(stderr()); flush(stdout())

### Name: LDEsysMat
### Title: Generate system matrix for LDEs
### Aliases: LDEsysMat
### Keywords: models

### ** Examples

# incidence matrix for a two compartment open system
incidence <-
  matrix(c(1,1,2,2,2,1,3,2,0), ncol = 3, byrow = TRUE,
   dimnames = list(NULL, c("Par", "From", "To")))
incidence
LDEsysMat(c(1.2, 0.3, 0.4), incidence)



cleanEx()
nameEx("MathAchieve")
### * MathAchieve

flush(stderr()); flush(stdout())

### Name: MathAchieve
### Title: Mathematics achievement scores
### Aliases: MathAchieve
### Keywords: datasets

### ** Examples

summary(MathAchieve)



cleanEx()
nameEx("Matrix.pdMat")
### * Matrix.pdMat

flush(stderr()); flush(stdout())

### Name: Matrix.pdMat
### Title: Assign Matrix to a pdMat or pdBlocked Object
### Aliases: matrix<-.pdMat matrix<-.pdBlocked
### Keywords: models

### ** Examples

class(pd1 <- pdSymm(diag(3))) # "pdSymm" "pdMat"
matrix(pd1) <- diag(1:3)
pd1



cleanEx()
nameEx("Matrix.reStruct")
### * Matrix.reStruct

flush(stderr()); flush(stdout())

### Name: Matrix.reStruct
### Title: Assign reStruct Matrices
### Aliases: matrix<-.reStruct
### Keywords: models

### ** Examples

rs1 <- reStruct(list(Dog = ~day, Side = ~1), data = Pixel)
matrix(rs1) <- list(diag(2), 3)



cleanEx()
nameEx("Names")
### * Names

flush(stderr()); flush(stdout())

### Name: Names
### Title: Names Associated with an Object
### Aliases: Names Names<-
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("Names.formula")
### * Names.formula

flush(stderr()); flush(stdout())

### Name: Names.formula
### Title: Extract Names from a formula
### Aliases: Names.formula Names.listForm
### Keywords: models

### ** Examples

Names(distance ~ Sex * age, data = Orthodont)



cleanEx()
nameEx("Names.pdBlocked")
### * Names.pdBlocked

flush(stderr()); flush(stdout())

### Name: Names.pdBlocked
### Title: Names of a pdBlocked Object
### Aliases: Names.pdBlocked Names<-.pdBlocked
### Keywords: models

### ** Examples

pd1 <- pdBlocked(list(~Sex - 1, ~age - 1), data = Orthodont)
Names(pd1)



cleanEx()
nameEx("Names.pdMat")
### * Names.pdMat

flush(stderr()); flush(stdout())

### Name: Names.pdMat
### Title: Names of a pdMat Object
### Aliases: Names.pdMat Names<-.pdMat
### Keywords: models

### ** Examples

pd1 <- pdSymm(~age, data = Orthodont)
Names(pd1)



cleanEx()
nameEx("Names.reStruct")
### * Names.reStruct

flush(stderr()); flush(stdout())

### Name: Names.reStruct
### Title: Names of an reStruct Object
### Aliases: Names.reStruct Names<-.reStruct
### Keywords: models

### ** Examples

rs1 <- reStruct(list(Dog = ~day, Side = ~1), data = Pixel)
Names(rs1)



cleanEx()
nameEx("Orthodont")
### * Orthodont

flush(stderr()); flush(stdout())

### Name: Orthodont
### Title: Growth curve data on an orthdontic measurement
### Aliases: Orthodont
### Keywords: datasets

### ** Examples

formula(Orthodont)
plot(Orthodont)



cleanEx()
nameEx("Pixel")
### * Pixel

flush(stderr()); flush(stdout())

### Name: Pixel
### Title: X-ray pixel intensities over time
### Aliases: Pixel
### Keywords: datasets

### ** Examples

fm1 <- lme(pixel ~ day + I(day^2), data = Pixel,
           random = list(Dog = ~ day, Side = ~ 1))
summary(fm1)
VarCorr(fm1)



cleanEx()
nameEx("Remifentanil")
### * Remifentanil

flush(stderr()); flush(stdout())

### Name: Remifentanil
### Title: Pharmacokinetics of Remifentanil
### Aliases: Remifentanil
### Keywords: datasets

### ** Examples

plot(Remifentanil, type = "l", lwd = 2) # shows the 65 patients' remi profiles

## The same on  log-log  scale  (*more* sensible for modeling ?):
plot(Remifentanil, type = "l", lwd = 2, scales = list(log=TRUE))

str(Remifentanil)
summary(Remifentanil)

plot(xtabs(~Subject, Remifentanil))
summary(unclass(table(Remifentanil$Subject)))
## between 20 and 54 measurements per patient (median: 24; mean: 32.42)

## Only first measurement of each patient :
dim(Remi.1 <- Remifentanil[!duplicated(Remifentanil[,"ID"]),]) #  65 x 12

LBMfn <- function(Wt, Ht, Sex) ifelse(Sex == "Female",
                                        1.07 * Wt - 148*(Wt/Ht)^2,
                                        1.1  * Wt - 128*(Wt/Ht)^2)
with(Remi.1,
    stopifnot(all.equal(BSA, Wt^{0.425} * Ht^{0.725} * 0.007184, tol = 1.5e-5),
              all.equal(LBM, LBMfn(Wt, Ht, Sex),                 tol = 7e-7)
))

## Rate: typically  3 µg / kg body weight, but :
sunflowerplot(Rate ~ Wt, Remifentanil)
abline(0,3, lty=2, col=adjustcolor("black", 0.5))



cleanEx()
nameEx("Soybean")
### * Soybean

flush(stderr()); flush(stdout())

### Name: Soybean
### Title: Growth of soybean plants
### Aliases: Soybean
### Keywords: datasets

### ** Examples

summary(fm1 <- nlsList(SSlogis, data = Soybean))



cleanEx()
nameEx("VarCorr")
### * VarCorr

flush(stderr()); flush(stdout())

### Name: VarCorr
### Title: Extract variance and correlation components
### Aliases: VarCorr VarCorr.lme VarCorr.pdMat VarCorr.pdBlocked
###   print.VarCorr.lme
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age, data = Orthodont, random = ~age)
VarCorr(fm1)



cleanEx()
nameEx("Variogram")
### * Variogram

flush(stderr()); flush(stdout())

### Name: Variogram
### Title: Calculate Semi-variogram
### Aliases: Variogram
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("Variogram.corExp")
### * Variogram.corExp

flush(stderr()); flush(stdout())

### Name: Variogram.corExp
### Title: Calculate Semi-variogram for a corExp Object
### Aliases: Variogram.corExp
### Keywords: models

### ** Examples

stopifnot(require("stats", quietly = TRUE))
cs1 <- corExp(3, form = ~ Time | Rat)
cs1 <- Initialize(cs1, BodyWeight)
Variogram(cs1)[1:10,]



cleanEx()
nameEx("Variogram.corGaus")
### * Variogram.corGaus

flush(stderr()); flush(stdout())

### Name: Variogram.corGaus
### Title: Calculate Semi-variogram for a corGaus Object
### Aliases: Variogram.corGaus
### Keywords: models

### ** Examples

cs1 <- corGaus(3, form = ~ Time | Rat)
cs1 <- Initialize(cs1, BodyWeight)
Variogram(cs1)[1:10,]



cleanEx()
nameEx("Variogram.corLin")
### * Variogram.corLin

flush(stderr()); flush(stdout())

### Name: Variogram.corLin
### Title: Calculate Semi-variogram for a corLin Object
### Aliases: Variogram.corLin
### Keywords: models

### ** Examples

cs1 <- corLin(15, form = ~ Time | Rat)
cs1 <- Initialize(cs1, BodyWeight)
Variogram(cs1)[1:10,]



cleanEx()
nameEx("Variogram.corRatio")
### * Variogram.corRatio

flush(stderr()); flush(stdout())

### Name: Variogram.corRatio
### Title: Calculate Semi-variogram for a corRatio Object
### Aliases: Variogram.corRatio
### Keywords: models

### ** Examples

cs1 <- corRatio(7, form = ~ Time | Rat)
cs1 <- Initialize(cs1, BodyWeight)
Variogram(cs1)[1:10,]



cleanEx()
nameEx("Variogram.corSpatial")
### * Variogram.corSpatial

flush(stderr()); flush(stdout())

### Name: Variogram.corSpatial
### Title: Calculate Semi-variogram for a corSpatial Object
### Aliases: Variogram.corSpatial
### Keywords: models

### ** Examples

cs1 <- corExp(3, form = ~ Time | Rat)
cs1 <- Initialize(cs1, BodyWeight)
Variogram(cs1, FUN = function(x, y) (1 - exp(-x/y)))[1:10,]



cleanEx()
nameEx("Variogram.corSpher")
### * Variogram.corSpher

flush(stderr()); flush(stdout())

### Name: Variogram.corSpher
### Title: Calculate Semi-variogram for a corSpher Object
### Aliases: Variogram.corSpher
### Keywords: models

### ** Examples

cs1 <- corSpher(15, form = ~ Time | Rat)
cs1 <- Initialize(cs1, BodyWeight)
Variogram(cs1)[1:10,]



cleanEx()
nameEx("Variogram.default")
### * Variogram.default

flush(stderr()); flush(stdout())

### Name: Variogram.default
### Title: Calculate Semi-variogram
### Aliases: Variogram.default
### Keywords: models

### ** Examples

fm1 <- lm(follicles ~ sin(2 * pi * Time) + cos(2 * pi * Time), Ovary,
          subset = Mare == 1)
Variogram(resid(fm1), dist(1:29))[1:10,]



cleanEx()
nameEx("Variogram.gls")
### * Variogram.gls

flush(stderr()); flush(stdout())

### Name: Variogram.gls
### Title: Calculate Semi-variogram for Residuals from a gls Object
### Aliases: Variogram.gls
### Keywords: models

### ** Examples

fm1 <- gls(weight ~ Time * Diet, BodyWeight)
Vm1 <- Variogram(fm1, form = ~ Time | Rat)
print(head(Vm1), digits = 3)



cleanEx()
nameEx("Variogram.lme")
### * Variogram.lme

flush(stderr()); flush(stdout())

### Name: Variogram.lme
### Title: Calculate Semi-variogram for Residuals from an lme Object
### Aliases: Variogram.lme
### Keywords: models

### ** Examples

fm1 <- lme(weight ~ Time * Diet, data=BodyWeight, ~ Time | Rat)
Variogram(fm1, form = ~ Time | Rat, nint = 10, robust = TRUE)



cleanEx()
nameEx("allCoef")
### * allCoef

flush(stderr()); flush(stdout())

### Name: allCoef
### Title: Extract Coefficients from a Set of Objects
### Aliases: allCoef
### Keywords: models

### ** Examples

cs1 <- corAR1(0.1)
vf1 <- varPower(0.5)
allCoef(cs1, vf1)



cleanEx()
nameEx("anova.gls")
### * anova.gls

flush(stderr()); flush(stdout())

### Name: anova.gls
### Title: Compare Likelihoods of Fitted Objects
### Aliases: anova.gls
### Keywords: models

### ** Examples

# AR(1) errors within each Mare
fm1 <- gls(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary,
           correlation = corAR1(form = ~ 1 | Mare))
anova(fm1)
# variance changes with a power of the absolute fitted values?
fm2 <- update(fm1, weights = varPower())
anova(fm1, fm2)

# Pinheiro and Bates, p. 251-252
fm1Orth.gls <- gls(distance ~ Sex * I(age - 11), Orthodont,
                correlation = corSymm(form = ~ 1 | Subject),
                weights = varIdent(form = ~ 1 | age))
fm2Orth.gls <- update(fm1Orth.gls,
                corr = corCompSymm(form = ~ 1 | Subject))
anova(fm1Orth.gls, fm2Orth.gls)

# Pinheiro and Bates, pp. 215-215, 255-260
#p. 215
fm1Dial.lme <-
  lme(rate ~(pressure + I(pressure^2) + I(pressure^3) + I(pressure^4))*QB,
      Dialyzer, ~ pressure + I(pressure^2))
# p. 216
fm2Dial.lme <- update(fm1Dial.lme,
                  weights = varPower(form = ~ pressure))
# p. 255
fm1Dial.gls <- gls(rate ~ (pressure +
     I(pressure^2) + I(pressure^3) + I(pressure^4))*QB,
        Dialyzer)
fm2Dial.gls <- update(fm1Dial.gls,
                 weights = varPower(form = ~ pressure))
anova(fm1Dial.gls, fm2Dial.gls)
fm3Dial.gls <- update(fm2Dial.gls,
                    corr = corAR1(0.771, form = ~ 1 | Subject))
anova(fm2Dial.gls, fm3Dial.gls)
# anova.gls to compare a gls and an lme fit 
anova(fm3Dial.gls, fm2Dial.lme, test = FALSE)

# Pinheiro and Bates, pp. 261-266
fm1Wheat2 <- gls(yield ~ variety - 1, Wheat2)
fm3Wheat2 <- update(fm1Wheat2,
      corr = corRatio(c(12.5, 0.2),
        form = ~ latitude + longitude, nugget = TRUE))
# Test a specific contrast 
anova(fm3Wheat2, L = c(-1, 0, 1))




cleanEx()
nameEx("anova.lme")
### * anova.lme

flush(stderr()); flush(stdout())

### Name: anova.lme
### Title: Compare Likelihoods of Fitted Objects
### Aliases: anova.lme print.anova.lme
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age, Orthodont, random = ~ age | Subject)
anova(fm1)
fm2 <- update(fm1, random = pdDiag(~age))
anova(fm1, fm2)

## Pinheiro and Bates, pp. 251-254 ------------------------------------------
fm1Orth.gls <- gls(distance ~ Sex * I(age - 11), Orthodont,
		   correlation = corSymm(form = ~ 1 | Subject),
		   weights = varIdent(form = ~ 1 | age))
fm2Orth.gls <- update(fm1Orth.gls,
		      corr = corCompSymm(form = ~ 1 | Subject))
## anova.gls examples:
anova(fm1Orth.gls, fm2Orth.gls)
fm3Orth.gls <- update(fm2Orth.gls, weights = NULL)
anova(fm2Orth.gls, fm3Orth.gls)
fm4Orth.gls <- update(fm3Orth.gls, weights = varIdent(form = ~ 1 | Sex))
anova(fm3Orth.gls, fm4Orth.gls)
# not in book but needed for the following command
fm3Orth.lme <- lme(distance ~ Sex*I(age-11), data = Orthodont,
                   random = ~ I(age-11) | Subject,
                   weights = varIdent(form = ~ 1 | Sex))
# Compare an "lme" object with a "gls" object (test would be non-sensical!)
anova(fm3Orth.lme, fm4Orth.gls, test = FALSE)

## Pinheiro and Bates, pp. 222-225 ------------------------------------------
op <- options(contrasts = c("contr.treatment", "contr.poly"))
fm1BW.lme <- lme(weight ~ Time * Diet, BodyWeight, random = ~ Time)
fm2BW.lme <- update(fm1BW.lme, weights = varPower())
# Test a specific contrast
anova(fm2BW.lme, L = c("Time:Diet2" = 1, "Time:Diet3" = -1))

## Pinheiro and Bates, pp. 352-365 ------------------------------------------
fm1Theo.lis <- nlsList(
     conc ~ SSfol(Dose, Time, lKe, lKa, lCl), data=Theoph)
fm1Theo.lis
fm1Theo.nlme <- nlme(fm1Theo.lis)
fm2Theo.nlme <- update(fm1Theo.nlme, random= pdDiag(lKe+lKa+lCl~1) )
fm3Theo.nlme <- update(fm2Theo.nlme, random= pdDiag(    lKa+lCl~1) )

# Comparing the 3 nlme models
anova(fm1Theo.nlme, fm3Theo.nlme, fm2Theo.nlme)

options(op) # (set back to previous state)



base::options(contrasts = c(unordered = "contr.treatment",ordered = "contr.poly"))
cleanEx()
nameEx("as.matrix.corStruct")
### * as.matrix.corStruct

flush(stderr()); flush(stdout())

### Name: as.matrix.corStruct
### Title: Matrix of a corStruct Object
### Aliases: as.matrix.corStruct
### Keywords: models

### ** Examples

cst1 <- corAR1(form = ~1|Subject)
cst1 <- Initialize(cst1, data = Orthodont)
as.matrix(cst1)



cleanEx()
nameEx("as.matrix.pdMat")
### * as.matrix.pdMat

flush(stderr()); flush(stdout())

### Name: as.matrix.pdMat
### Title: Matrix of a pdMat Object
### Aliases: as.matrix.pdMat
### Keywords: models

### ** Examples

as.matrix(pdSymm(diag(4)))



cleanEx()
nameEx("as.matrix.reStruct")
### * as.matrix.reStruct

flush(stderr()); flush(stdout())

### Name: as.matrix.reStruct
### Title: Matrices of an reStruct Object
### Aliases: as.matrix.reStruct
### Keywords: models

### ** Examples

rs1 <- reStruct(pdSymm(diag(3), ~age+Sex, data = Orthodont))
as.matrix(rs1)



cleanEx()
nameEx("asOneFormula")
### * asOneFormula

flush(stderr()); flush(stdout())

### Name: asOneFormula
### Title: Combine Formulas of a Set of Objects
### Aliases: asOneFormula
### Keywords: models

### ** Examples

asOneFormula(y ~ x + z | g, list(~ w, ~ t * sin(2 * pi)))



cleanEx()
nameEx("asTable")
### * asTable

flush(stderr()); flush(stdout())

### Name: asTable
### Title: Convert groupedData to a matrix
### Aliases: asTable asTable.groupedData
### Keywords: manip

### ** Examples

asTable(Orthodont)

# Pinheiro and Bates, p. 109
ergoStool.mat <- asTable(ergoStool)



cleanEx()
nameEx("augPred")
### * augPred

flush(stderr()); flush(stdout())

### Name: augPred
### Title: Augmented Predictions
### Aliases: augPred augPred.gls augPred.lme augPred.lmList
### Keywords: models

### ** Examples

fm1 <- lme(Orthodont, random = ~1)
augPred(fm1, length.out = 2, level = c(0,1))



cleanEx()
nameEx("balancedGrouped")
### * balancedGrouped

flush(stderr()); flush(stdout())

### Name: balancedGrouped
### Title: Create a groupedData object from a matrix
### Aliases: balancedGrouped
### Keywords: data

### ** Examples

OrthoMat <- asTable( Orthodont )
Orth2 <- balancedGrouped(distance ~ age | Subject, data = OrthoMat,
    labels = list(x = "Age",
                  y = "Distance from pituitary to pterygomaxillary fissure"),
    units = list(x = "(yr)", y = "(mm)"))
Orth2[ 1:10, ]        ## check the first few entries

# Pinheiro and Bates, p. 109
ergoStool.mat <- asTable(ergoStool)
balancedGrouped(effort~Type|Subject,
                data=ergoStool.mat)



cleanEx()
nameEx("bdf")
### * bdf

flush(stderr()); flush(stdout())

### Name: bdf
### Title: Language scores
### Aliases: bdf
### Keywords: datasets

### ** Examples

summary(bdf)

## More examples, including lme() fits  reproducing parts in the above
## book, are available in the R script files
system.file("mlbook", "ch04.R", package ="nlme") # and
system.file("mlbook", "ch05.R", package ="nlme")



cleanEx()
nameEx("coef.corStruct")
### * coef.corStruct

flush(stderr()); flush(stdout())

### Name: coef.corStruct
### Title: Coefficients of a corStruct Object
### Aliases: coef.corStruct coef.corAR1 coef.corARMAd coef.corCAR1
###   coef.corCompSymm coef.corHF coef.corLin coef.corNatural
###   coef.corSpatial coef.corSpher coef.corSymm coef<-.corStruct
###   coef<-.corAR1 coef<-.corARMA coef<-.corCAR1 coef<-.corCompSymm
###   coef<-.corNatural coef<-.corHF coef<-.corLin coef<-.corSpatial
###   coef<-.corSpher coef<-.corSymm coef.summary.nlsList
### Keywords: models

### ** Examples

cst1 <- corARMA(p = 1, q = 1)
coef(cst1)



cleanEx()
nameEx("coef.gnls")
### * coef.gnls

flush(stderr()); flush(stdout())

### Name: coef.gnls
### Title: Extract gnls Coefficients
### Aliases: coef.gnls
### Keywords: models

### ** Examples

fm1 <- gnls(weight ~ SSlogis(Time, Asym, xmid, scal), Soybean,
            weights = varPower())
coef(fm1)



cleanEx()
nameEx("coef.lmList")
### * coef.lmList

flush(stderr()); flush(stdout())

### Name: coef.lmList
### Title: Extract lmList Coefficients
### Aliases: coef.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age|Subject, data = Orthodont)
coef(fm1)
coef(fm1, augFrame = TRUE)



cleanEx()
nameEx("coef.lme")
### * coef.lme

flush(stderr()); flush(stdout())

### Name: coef.lme
### Title: Extract lme Coefficients
### Aliases: coef.lme
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age, Orthodont, random = ~ age | Subject)
coef(fm1)
coef(fm1, augFrame = TRUE)



cleanEx()
nameEx("coef.modelStruct")
### * coef.modelStruct

flush(stderr()); flush(stdout())

### Name: coef.modelStruct
### Title: Extract modelStruct Object Coefficients
### Aliases: coef.modelStruct coef<-.modelStruct
### Keywords: models

### ** Examples

lms1 <- lmeStruct(reStruct = reStruct(pdDiag(diag(2), ~age)),
   corStruct = corAR1(0.3))
coef(lms1)



cleanEx()
nameEx("coef.pdMat")
### * coef.pdMat

flush(stderr()); flush(stdout())

### Name: coef.pdMat
### Title: pdMat Object Coefficients
### Aliases: coef.pdMat coef.pdBlocked coef.pdCompSymm coef.pdDiag
###   coef.pdIdent coef.pdNatural coef.pdSymm coef<-.pdMat coef<-.pdBlocked
### Keywords: models

### ** Examples

coef(pdSymm(diag(3)))



cleanEx()
nameEx("coef.reStruct")
### * coef.reStruct

flush(stderr()); flush(stdout())

### Name: coef.reStruct
### Title: reStruct Object Coefficients
### Aliases: coef.reStruct coef<-.reStruct
### Keywords: models

### ** Examples

rs1 <- reStruct(list(A = pdSymm(diag(1:3), form = ~Score),
  B = pdDiag(2 * diag(4), form = ~Educ)))
coef(rs1)



cleanEx()
nameEx("coef.varFunc")
### * coef.varFunc

flush(stderr()); flush(stdout())

### Name: coef.varFunc
### Title: varFunc Object Coefficients
### Aliases: coef.varFunc coef.varComb coef.varConstPower coef.varConstProp
###   coef.varExp coef.varFixed coef.varIdent coef.varPower coef<-.varComb
###   coef<-.varConstPower coef<-.varConstProp coef<-.varExp
###   coef<-.varFixed coef<-.varIdent coef<-.varPower
### Keywords: models

### ** Examples

vf1 <- varPower(1)
coef(vf1)
coef(vf1) <- 2



cleanEx()
nameEx("collapse")
### * collapse

flush(stderr()); flush(stdout())

### Name: collapse
### Title: Collapse According to Groups
### Aliases: collapse
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("collapse.groupedData")
### * collapse.groupedData

flush(stderr()); flush(stdout())

### Name: collapse.groupedData
### Title: Collapse a groupedData Object
### Aliases: collapse.groupedData
### Keywords: models

### ** Examples

# collapsing by Dog
collapse(Pixel, collapse = 1)  # same as collapse(Pixel, collapse = "Dog")



cleanEx()
nameEx("compareFits")
### * compareFits

flush(stderr()); flush(stdout())

### Name: compareFits
### Title: Compare Fitted Objects
### Aliases: compareFits print.compareFits
### Keywords: models

### ** Examples

fm1 <- lmList(Orthodont)
fm2 <- lme(fm1)
(cF12 <- compareFits(coef(fm1), coef(fm2)))



cleanEx()
nameEx("comparePred")
### * comparePred

flush(stderr()); flush(stdout())

### Name: comparePred
### Title: Compare Predictions
### Aliases: comparePred comparePred.gls comparePred.lme comparePred.lmList
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age * Sex, data = Orthodont, random = ~ age)
fm2 <- update(fm1, distance ~ age)
comparePred(fm1, fm2, length.out = 2)



cleanEx()
nameEx("corAR1")
### * corAR1

flush(stderr()); flush(stdout())

### Name: corAR1
### Title: AR(1) Correlation Structure
### Aliases: corAR1
### Keywords: models

### ** Examples

## covariate is observation order and grouping factor is Mare
cs1 <- corAR1(0.2, form = ~ 1 | Mare)

# Pinheiro and Bates, p. 236
cs1AR1 <- corAR1(0.8, form = ~ 1 | Subject)
cs1AR1. <- Initialize(cs1AR1, data = Orthodont)
corMatrix(cs1AR1.)

# Pinheiro and Bates, p. 240
fm1Ovar.lme <- lme(follicles ~ sin(2*pi*Time) + cos(2*pi*Time),
                   data = Ovary, random = pdDiag(~sin(2*pi*Time)))
fm2Ovar.lme <- update(fm1Ovar.lme, correlation = corAR1())

# Pinheiro and Bates, pp. 255-258:  use in gls
fm1Dial.gls <-
  gls(rate ~(pressure + I(pressure^2) + I(pressure^3) + I(pressure^4))*QB,
      Dialyzer)
fm2Dial.gls <- update(fm1Dial.gls,
                 weights = varPower(form = ~ pressure))
fm3Dial.gls <- update(fm2Dial.gls,
                    corr = corAR1(0.771, form = ~ 1 | Subject))

# Pinheiro and Bates use in nlme:  
# from p. 240 needed on p. 396
fm1Ovar.lme <- lme(follicles ~ sin(2*pi*Time) + cos(2*pi*Time),
                   data = Ovary, random = pdDiag(~sin(2*pi*Time)))
fm5Ovar.lme <- update(fm1Ovar.lme,
                corr = corARMA(p = 1, q = 1))
# p. 396
fm1Ovar.nlme <- nlme(follicles~
     A+B*sin(2*pi*w*Time)+C*cos(2*pi*w*Time),
   data=Ovary, fixed=A+B+C+w~1,
   random=pdDiag(A+B+w~1),
   start=c(fixef(fm5Ovar.lme), 1) )
# p. 397
fm2Ovar.nlme <- update(fm1Ovar.nlme,
         corr=corAR1(0.311) )



cleanEx()
nameEx("corARMA")
### * corARMA

flush(stderr()); flush(stdout())

### Name: corARMA
### Title: ARMA(p,q) Correlation Structure
### Aliases: corARMA coef.corARMA
### Keywords: models

### ** Examples

## ARMA(1,2) structure, with observation order as a covariate and
## Mare as grouping factor
cs1 <- corARMA(c(0.2, 0.3, -0.1), form = ~ 1 | Mare, p = 1, q = 2)

# Pinheiro and Bates, p. 237 
cs1ARMA <- corARMA(0.4, form = ~ 1 | Subject, q = 1)
cs1ARMA <- Initialize(cs1ARMA, data = Orthodont)
corMatrix(cs1ARMA)

cs2ARMA <- corARMA(c(0.8, 0.4), form = ~ 1 | Subject, p=1, q=1)
cs2ARMA <- Initialize(cs2ARMA, data = Orthodont)
corMatrix(cs2ARMA)

# Pinheiro and Bates use in nlme:  
# from p. 240 needed on p. 396
fm1Ovar.lme <- lme(follicles ~ sin(2*pi*Time) + cos(2*pi*Time),
                   data = Ovary, random = pdDiag(~sin(2*pi*Time)))
fm5Ovar.lme <- update(fm1Ovar.lme,
                corr = corARMA(p = 1, q = 1))
# p. 396
fm1Ovar.nlme <- nlme(follicles~
     A+B*sin(2*pi*w*Time)+C*cos(2*pi*w*Time),
   data=Ovary, fixed=A+B+C+w~1,
   random=pdDiag(A+B+w~1),
   start=c(fixef(fm5Ovar.lme), 1) )
# p. 397
fm3Ovar.nlme <- update(fm1Ovar.nlme,
         corr=corARMA(p=0, q=2) )



cleanEx()
nameEx("corCAR1")
### * corCAR1

flush(stderr()); flush(stdout())

### Name: corCAR1
### Title: Continuous AR(1) Correlation Structure
### Aliases: corCAR1
### Keywords: models

### ** Examples

## covariate is Time and grouping factor is Mare
cs1 <- corCAR1(0.2, form = ~ Time | Mare)

# Pinheiro and Bates, pp. 240, 243
fm1Ovar.lme <- lme(follicles ~
           sin(2*pi*Time) + cos(2*pi*Time),
   data = Ovary, random = pdDiag(~sin(2*pi*Time)))
fm4Ovar.lme <- update(fm1Ovar.lme,
          correlation = corCAR1(form = ~Time))




cleanEx()
nameEx("corCompSymm")
### * corCompSymm

flush(stderr()); flush(stdout())

### Name: corCompSymm
### Title: Compound Symmetry Correlation Structure
### Aliases: corCompSymm
### Keywords: models

### ** Examples

## covariate is observation order and grouping factor is Subject
cs1 <- corCompSymm(0.5, form = ~ 1 | Subject)

# Pinheiro and Bates, pp. 222-225 
fm1BW.lme <- lme(weight ~ Time * Diet, BodyWeight,
                   random = ~ Time)
# p. 223
fm2BW.lme <- update(fm1BW.lme, weights = varPower())
# p. 225
cs1CompSymm <- corCompSymm(value = 0.3, form = ~ 1 | Subject)
cs2CompSymm <- corCompSymm(value = 0.3, form = ~ age | Subject)
cs1CompSymm <- Initialize(cs1CompSymm, data = Orthodont)
corMatrix(cs1CompSymm)

## Print/Summary methods for the empty case:
(cCS <- corCompSymm()) # Uninitialized correlation struc..
summary(cCS)           #    (ditto)



cleanEx()
nameEx("corExp")
### * corExp

flush(stderr()); flush(stdout())

### Name: corExp
### Title: Exponential Correlation Structure
### Aliases: corExp
### Keywords: models

### ** Examples

sp1 <- corExp(form = ~ x + y + z)

# Pinheiro and Bates, p. 238
spatDat <- data.frame(x = (0:4)/4, y = (0:4)/4)

cs1Exp <- corExp(1, form = ~ x + y)
cs1Exp <- Initialize(cs1Exp, spatDat)
corMatrix(cs1Exp)

cs2Exp <- corExp(1, form = ~ x + y, metric = "man")
cs2Exp <- Initialize(cs2Exp, spatDat)
corMatrix(cs2Exp)

cs3Exp <- corExp(c(1, 0.2), form = ~ x + y,
                 nugget = TRUE)
cs3Exp <- Initialize(cs3Exp, spatDat)
corMatrix(cs3Exp)

# example lme(..., corExp ...)
# Pinheiro and Bates, pp. 222-247
# p. 222
options(contrasts = c("contr.treatment", "contr.poly"))
fm1BW.lme <- lme(weight ~ Time * Diet, BodyWeight,
                   random = ~ Time)
# p. 223
fm2BW.lme <- update(fm1BW.lme, weights = varPower())
# p. 246
fm3BW.lme <- update(fm2BW.lme,
           correlation = corExp(form = ~ Time))
# p. 247
fm4BW.lme <-
      update(fm3BW.lme, correlation = corExp(form =  ~ Time,
                        nugget = TRUE))
anova(fm3BW.lme, fm4BW.lme)




base::options(contrasts = c(unordered = "contr.treatment",ordered = "contr.poly"))
cleanEx()
nameEx("corFactor")
### * corFactor

flush(stderr()); flush(stdout())

### Name: corFactor
### Title: Factor of a Correlation Matrix
### Aliases: corFactor
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("corFactor.corStruct")
### * corFactor.corStruct

flush(stderr()); flush(stdout())

### Name: corFactor.corStruct
### Title: Factor of a corStruct Object Matrix
### Aliases: corFactor.corCompSymm corFactor.corAR1 corFactor.corARMA
###   corFactor.corCAR1 corFactor.corNatural corFactor.corSpatial
###   corFactor.corStruct corFactor.corSymm
### Keywords: models

### ** Examples

cs1 <- corAR1(form = ~1 | Subject)
cs1 <- Initialize(cs1, data = Orthodont)
corFactor(cs1)



cleanEx()
nameEx("corGaus")
### * corGaus

flush(stderr()); flush(stdout())

### Name: corGaus
### Title: Gaussian Correlation Structure
### Aliases: corGaus
### Keywords: models

### ** Examples

sp1 <- corGaus(form = ~ x + y + z)

# example lme(..., corGaus ...)
# Pinheiro and Bates, pp. 222-249
fm1BW.lme <- lme(weight ~ Time * Diet, BodyWeight,
                   random = ~ Time)
# p. 223
fm2BW.lme <- update(fm1BW.lme, weights = varPower())
# p 246 
fm3BW.lme <- update(fm2BW.lme,
           correlation = corExp(form = ~ Time))
# p. 249
fm8BW.lme <- update(fm3BW.lme, correlation = corGaus(form = ~ Time))




cleanEx()
nameEx("corLin")
### * corLin

flush(stderr()); flush(stdout())

### Name: corLin
### Title: Linear Correlation Structure
### Aliases: corLin
### Keywords: models

### ** Examples

sp1 <- corLin(form = ~ x + y)

# example lme(..., corLin ...)
# Pinheiro and Bates, pp. 222-249
fm1BW.lme <- lme(weight ~ Time * Diet, BodyWeight,
                   random = ~ Time)
# p. 223
fm2BW.lme <- update(fm1BW.lme, weights = varPower())
# p 246 
fm3BW.lme <- update(fm2BW.lme,
           correlation = corExp(form = ~ Time))
# p. 249
fm7BW.lme <- update(fm3BW.lme, correlation = corLin(form = ~ Time))




cleanEx()
nameEx("corMatrix")
### * corMatrix

flush(stderr()); flush(stdout())

### Name: corMatrix
### Title: Extract Correlation Matrix
### Aliases: corMatrix
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("corMatrix.corStruct")
### * corMatrix.corStruct

flush(stderr()); flush(stdout())

### Name: corMatrix.corStruct
### Title: Matrix of a corStruct Object
### Aliases: corMatrix.corStruct corMatrix.corCompSymm corMatrix.corAR1
###   corMatrix.corARMA corMatrix.corCAR1 corMatrix.corCompSymm
###   corMatrix.corNatural corMatrix.corSpatial corMatrix.corSymm
### Keywords: models

### ** Examples

cs1 <- corAR1(0.3)
corMatrix(cs1, covariate = 1:4)
corMatrix(cs1, covariate = 1:4, corr = FALSE)

# Pinheiro and Bates, p. 225
cs1CompSymm <- corCompSymm(value = 0.3, form = ~ 1 | Subject)
cs1CompSymm <- Initialize(cs1CompSymm, data = Orthodont)
corMatrix(cs1CompSymm)

# Pinheiro and Bates, p. 226
cs1Symm <- corSymm(value = c(0.2, 0.1, -0.1, 0, 0.2, 0),
                   form = ~ 1 | Subject)
cs1Symm <- Initialize(cs1Symm, data = Orthodont)
corMatrix(cs1Symm)

# Pinheiro and Bates, p. 236 
cs1AR1 <- corAR1(0.8, form = ~ 1 | Subject)
cs1AR1 <- Initialize(cs1AR1, data = Orthodont)
corMatrix(cs1AR1)

# Pinheiro and Bates, p. 237 
cs1ARMA <- corARMA(0.4, form = ~ 1 | Subject, q = 1)
cs1ARMA <- Initialize(cs1ARMA, data = Orthodont)
corMatrix(cs1ARMA)

# Pinheiro and Bates, p. 238 
spatDat <- data.frame(x = (0:4)/4, y = (0:4)/4)
cs1Exp <- corExp(1, form = ~ x + y)
cs1Exp <- Initialize(cs1Exp, spatDat)
corMatrix(cs1Exp)



cleanEx()
nameEx("corMatrix.pdMat")
### * corMatrix.pdMat

flush(stderr()); flush(stdout())

### Name: corMatrix.pdMat
### Title: Extract Correlation Matrix from a pdMat Object
### Aliases: corMatrix.pdBlocked corMatrix.pdCompSymm corMatrix.pdDiag
###   corMatrix.pdIdent corMatrix.pdMat corMatrix.pdSymm
### Keywords: models

### ** Examples

pd1 <- pdSymm(diag(1:4))
corMatrix(pd1)



cleanEx()
nameEx("corMatrix.reStruct")
### * corMatrix.reStruct

flush(stderr()); flush(stdout())

### Name: corMatrix.reStruct
### Title: Extract Correlation Matrix from Components of an reStruct Object
### Aliases: corMatrix.reStruct
### Keywords: models

### ** Examples

rs1 <- reStruct(pdSymm(diag(3), ~age+Sex, data = Orthodont))
corMatrix(rs1)



cleanEx()
nameEx("corNatural")
### * corNatural

flush(stderr()); flush(stdout())

### Name: corNatural
### Title: General correlation in natural parameterization
### Aliases: corNatural print.corNatural
### Keywords: models

### ** Examples

## covariate is observation order and grouping factor is Subject
cs1 <- corNatural(form = ~ 1 | Subject)



cleanEx()
nameEx("corRatio")
### * corRatio

flush(stderr()); flush(stdout())

### Name: corRatio
### Title: Rational Quadratic Correlation Structure
### Aliases: corRatio
### Keywords: models

### ** Examples

sp1 <- corRatio(form = ~ x + y + z)

# example lme(..., corRatio ...)
# Pinheiro and Bates, pp. 222-249
fm1BW.lme <- lme(weight ~ Time * Diet, BodyWeight,
                   random = ~ Time)
# p. 223
fm2BW.lme <- update(fm1BW.lme, weights = varPower())
# p 246 
fm3BW.lme <- update(fm2BW.lme,
           correlation = corExp(form = ~ Time))
# p. 249
fm5BW.lme <- update(fm3BW.lme, correlation =
                   corRatio(form = ~ Time))

# example gls(..., corRatio ...)
# Pinheiro and Bates, pp. 261, 263
fm1Wheat2 <- gls(yield ~ variety - 1, Wheat2)
# p. 263 
fm3Wheat2 <- update(fm1Wheat2, corr = 
    corRatio(c(12.5, 0.2),
       form = ~ latitude + longitude,
             nugget = TRUE))




cleanEx()
nameEx("corSpatial")
### * corSpatial

flush(stderr()); flush(stdout())

### Name: corSpatial
### Title: Spatial Correlation Structure
### Aliases: corSpatial
### Keywords: models

### ** Examples

sp1 <- corSpatial(form = ~ x + y + z, type = "g", metric = "man")



cleanEx()
nameEx("corSpher")
### * corSpher

flush(stderr()); flush(stdout())

### Name: corSpher
### Title: Spherical Correlation Structure
### Aliases: corSpher
### Keywords: models

### ** Examples

sp1 <- corSpher(form = ~ x + y)

# example lme(..., corSpher ...)
# Pinheiro and Bates, pp. 222-249
fm1BW.lme <- lme(weight ~ Time * Diet, BodyWeight,
                   random = ~ Time)
# p. 223
fm2BW.lme <- update(fm1BW.lme, weights = varPower())
# p 246 
fm3BW.lme <- update(fm2BW.lme,
           correlation = corExp(form = ~ Time))
# p. 249
fm6BW.lme <- update(fm3BW.lme,
          correlation = corSpher(form = ~ Time))

# example gls(..., corSpher ...)
# Pinheiro and Bates, pp. 261, 263
fm1Wheat2 <- gls(yield ~ variety - 1, Wheat2)
# p. 262 
fm2Wheat2 <- update(fm1Wheat2, corr =
   corSpher(c(28, 0.2),
     form = ~ latitude + longitude, nugget = TRUE))




cleanEx()
nameEx("corSymm")
### * corSymm

flush(stderr()); flush(stdout())

### Name: corSymm
### Title: General Correlation Structure
### Aliases: corSymm
### Keywords: models

### ** Examples

## covariate is observation order and grouping factor is Subject
cs1 <- corSymm(form = ~ 1 | Subject)

# Pinheiro and Bates, p. 225 
cs1CompSymm <- corCompSymm(value = 0.3, form = ~ 1 | Subject)
cs1CompSymm <- Initialize(cs1CompSymm, data = Orthodont)
corMatrix(cs1CompSymm)

# Pinheiro and Bates, p. 226
cs1Symm <- corSymm(value =
        c(0.2, 0.1, -0.1, 0, 0.2, 0),
                   form = ~ 1 | Subject)
cs1Symm <- Initialize(cs1Symm, data = Orthodont)
corMatrix(cs1Symm)

# example gls(..., corSpher ...)
# Pinheiro and Bates, pp. 261, 263
fm1Wheat2 <- gls(yield ~ variety - 1, Wheat2)
# p. 262 
fm2Wheat2 <- update(fm1Wheat2, corr =
   corSpher(c(28, 0.2),
     form = ~ latitude + longitude, nugget = TRUE))

# example gls(..., corSymm ... )
# Pinheiro and Bates, p. 251
fm1Orth.gls <- gls(distance ~ Sex * I(age - 11), Orthodont,
                   correlation = corSymm(form = ~ 1 | Subject),
                   weights = varIdent(form = ~ 1 | age))




cleanEx()
nameEx("ergoStool")
### * ergoStool

flush(stderr()); flush(stdout())

### Name: ergoStool
### Title: Ergometrics experiment with stool types
### Aliases: ergoStool
### Keywords: datasets

### ** Examples

fm1 <-
   lme(effort ~ Type, data = ergoStool, random = ~ 1 | Subject)
anova( fm1 )



cleanEx()
nameEx("fdHess")
### * fdHess

flush(stderr()); flush(stdout())

### Name: fdHess
### Title: Finite difference Hessian
### Aliases: fdHess
### Keywords: models

### ** Examples

(fdH <- fdHess(c(12.3, 2.34), function(x) x[1]*(1-exp(-0.4*x[2]))))
stopifnot(length(fdH$ mean) == 1,
          length(fdH$ gradient) == 2,
          identical(dim(fdH$ Hessian), c(2L, 2L)))



cleanEx()
nameEx("fitted.lmList")
### * fitted.lmList

flush(stderr()); flush(stdout())

### Name: fitted.lmList
### Title: Extract lmList Fitted Values
### Aliases: fitted.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
fitted(fm1)



cleanEx()
nameEx("fitted.lme")
### * fitted.lme

flush(stderr()); flush(stdout())

### Name: fitted.lme
### Title: Extract lme Fitted Values
### Aliases: fitted.lme
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age + Sex, data = Orthodont, random = ~ 1)
fitted(fm1, level = 0:1)



cleanEx()
nameEx("fixed.effects")
### * fixed.effects

flush(stderr()); flush(stdout())

### Name: fixed.effects
### Title: Extract Fixed Effects
### Aliases: fixed.effects fixef
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("fixef.lmList")
### * fixef.lmList

flush(stderr()); flush(stdout())

### Name: fixef.lmList
### Title: Extract lmList Fixed Effects
### Aliases: fixed.effects.lmList fixef.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
fixed.effects(fm1)



cleanEx()
nameEx("formula.pdBlocked")
### * formula.pdBlocked

flush(stderr()); flush(stdout())

### Name: formula.pdBlocked
### Title: Extract pdBlocked Formula
### Aliases: formula.pdBlocked
### Keywords: models

### ** Examples

pd1 <- pdBlocked(list(~ age, ~ Sex - 1))
formula(pd1)
formula(pd1, asList = TRUE)



cleanEx()
nameEx("formula.pdMat")
### * formula.pdMat

flush(stderr()); flush(stdout())

### Name: formula.pdMat
### Title: Extract pdMat Formula
### Aliases: formula.pdMat
### Keywords: models

### ** Examples

pd1 <- pdSymm(~Sex*age)
formula(pd1)



cleanEx()
nameEx("formula.reStruct")
### * formula.reStruct

flush(stderr()); flush(stdout())

### Name: formula.reStruct
### Title: Extract reStruct Object Formula
### Aliases: formula.reStruct
### Keywords: models

### ** Examples

rs1 <- reStruct(list(A = pdDiag(diag(2), ~age), B = ~1))
formula(rs1)



cleanEx()
nameEx("gapply")
### * gapply

flush(stderr()); flush(stdout())

### Name: gapply
### Title: Apply a Function by Groups
### Aliases: gapply
### Keywords: data

### ** Examples

## Find number of non-missing "conc" observations for each Subject
gapply( Phenobarb, FUN = function(x) sum(!is.na(x$conc)) )

# Pinheiro and Bates, p. 127 
table( gapply(Quinidine, "conc", function(x) sum(!is.na(x))) )
changeRecords <- gapply( Quinidine, FUN = function(frm)
    any(is.na(frm[["conc"]]) & is.na(frm[["dose"]])) )



cleanEx()
nameEx("getCovariate")
### * getCovariate

flush(stderr()); flush(stdout())

### Name: getCovariate
### Title: Extract Covariate from an Object
### Aliases: getCovariate
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("getCovariate.corStruct")
### * getCovariate.corStruct

flush(stderr()); flush(stdout())

### Name: getCovariate.corStruct
### Title: Extract corStruct Object Covariate
### Aliases: getCovariate.corStruct getCovariate.corSpatial
### Keywords: models

### ** Examples

cs1 <- corAR1(form = ~ 1 | Subject)
getCovariate(cs1, data = Orthodont)



cleanEx()
nameEx("getCovariate.data.frame")
### * getCovariate.data.frame

flush(stderr()); flush(stdout())

### Name: getCovariate.data.frame
### Title: Extract Data Frame Covariate
### Aliases: getCovariate.data.frame
### Keywords: models

### ** Examples

getCovariate(Orthodont)



cleanEx()
nameEx("getCovariate.varFunc")
### * getCovariate.varFunc

flush(stderr()); flush(stdout())

### Name: getCovariate.varFunc
### Title: Extract varFunc Covariate
### Aliases: getCovariate.varFunc
### Keywords: models

### ** Examples

vf1 <- varPower(1.1, form = ~age)
covariate(vf1) <- Orthodont[["age"]]
getCovariate(vf1)



cleanEx()
nameEx("getCovariateFormula")
### * getCovariateFormula

flush(stderr()); flush(stdout())

### Name: getCovariateFormula
### Title: Extract Covariates Formula
### Aliases: getCovariateFormula
### Keywords: models

### ** Examples

getCovariateFormula(y ~ x | g)
getCovariateFormula(y ~ x)



cleanEx()
nameEx("getData")
### * getData

flush(stderr()); flush(stdout())

### Name: getData
### Title: Extract Data from an Object
### Aliases: getData
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("getData.gls")
### * getData.gls

flush(stderr()); flush(stdout())

### Name: getData.gls
### Title: Extract gls Object Data
### Aliases: getData.gls getData.gnls
### Keywords: models

### ** Examples

fm1 <- gls(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), data = Ovary,
           correlation = corAR1(form = ~ 1 | Mare))
getData(fm1)



cleanEx()
nameEx("getData.lmList")
### * getData.lmList

flush(stderr()); flush(stdout())

### Name: getData.lmList
### Title: Extract lmList Object Data
### Aliases: getData.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
getData(fm1)



cleanEx()
nameEx("getData.lme")
### * getData.lme

flush(stderr()); flush(stdout())

### Name: getData.lme
### Title: Extract lme Object Data
### Aliases: getData.lme getData.nlme getData.nls
### Keywords: models

### ** Examples

fm1 <- lme(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), data = Ovary,
           random = ~ sin(2*pi*Time))
getData(fm1)



cleanEx()
nameEx("getGroups")
### * getGroups

flush(stderr()); flush(stdout())

### Name: getGroups
### Title: Extract Grouping Factors from an Object
### Aliases: getGroups
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("getGroups.corStruct")
### * getGroups.corStruct

flush(stderr()); flush(stdout())

### Name: getGroups.corStruct
### Title: Extract corStruct Groups
### Aliases: getGroups.corStruct
### Keywords: models

### ** Examples

cs1 <- corAR1(form = ~ 1 | Subject)
getGroups(cs1, data = Orthodont)



cleanEx()
nameEx("getGroups.data.frame")
### * getGroups.data.frame

flush(stderr()); flush(stdout())

### Name: getGroups.data.frame
### Title: Extract Groups from a Data Frame
### Aliases: getGroups.data.frame
### Keywords: models

### ** Examples

getGroups(Pixel)
getGroups(Pixel, level = 2)



cleanEx()
nameEx("getGroups.gls")
### * getGroups.gls

flush(stderr()); flush(stdout())

### Name: getGroups.gls
### Title: Extract gls Object Groups
### Aliases: getGroups.gls
### Keywords: models

### ** Examples

fm1 <- gls(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary,
           correlation = corAR1(form = ~ 1 | Mare))
getGroups(fm1)



cleanEx()
nameEx("getGroups.lmList")
### * getGroups.lmList

flush(stderr()); flush(stdout())

### Name: getGroups.lmList
### Title: Extract lmList Object Groups
### Aliases: getGroups.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
getGroups(fm1)



cleanEx()
nameEx("getGroups.lme")
### * getGroups.lme

flush(stderr()); flush(stdout())

### Name: getGroups.lme
### Title: Extract lme Object Groups
### Aliases: getGroups.lme
### Keywords: models

### ** Examples

fm1 <- lme(pixel ~ day + day^2, Pixel,
  random = list(Dog = ~day, Side = ~1))
getGroups(fm1, level = 1:2)



cleanEx()
nameEx("getGroups.varFunc")
### * getGroups.varFunc

flush(stderr()); flush(stdout())

### Name: getGroups.varFunc
### Title: Extract varFunc Groups
### Aliases: getGroups.varFunc
### Keywords: models

### ** Examples

vf1 <- varPower(form = ~ age | Sex)
vf1 <- Initialize(vf1, Orthodont)
getGroups(vf1)



cleanEx()
nameEx("getGroupsFormula")
### * getGroupsFormula

flush(stderr()); flush(stdout())

### Name: getGroupsFormula
### Title: Extract Grouping Formula
### Aliases: getGroupsFormula getGroupsFormula.default getGroupsFormula.gls
###   getGroupsFormula.lmList getGroupsFormula.lme
###   getGroupsFormula.reStruct
### Keywords: models

### ** Examples

getGroupsFormula(y ~ x | g1/g2)



cleanEx()
nameEx("getResponse")
### * getResponse

flush(stderr()); flush(stdout())

### Name: getResponse
### Title: Extract Response Variable from an Object
### Aliases: getResponse getResponse.data.frame
### Keywords: models

### ** Examples

getResponse(Orthodont)



cleanEx()
nameEx("getResponseFormula")
### * getResponseFormula

flush(stderr()); flush(stdout())

### Name: getResponseFormula
### Title: Extract Formula Specifying Response Variable
### Aliases: getResponseFormula
### Keywords: models

### ** Examples

getResponseFormula(y ~ x | g)



cleanEx()
nameEx("getVarCov")
### * getVarCov

flush(stderr()); flush(stdout())

### Name: getVarCov
### Title: Extract variance-covariance matrix
### Aliases: getVarCov getVarCov.lme getVarCov.gls print.VarCov
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age, data = Orthodont, subset = Sex == "Female")
getVarCov(fm1)
getVarCov(fm1, individual = "F01", type = "marginal")
getVarCov(fm1, type = "conditional")
fm2 <- gls(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary,
           correlation = corAR1(form = ~ 1 | Mare))
getVarCov(fm2)



cleanEx()
nameEx("gls")
### * gls

flush(stderr()); flush(stdout())

### Name: gls
### Title: Fit Linear Model Using Generalized Least Squares
### Aliases: gls update.gls
### Keywords: models

### ** Examples

# AR(1) errors within each Mare
fm1 <- gls(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary,
           correlation = corAR1(form = ~ 1 | Mare))
# variance increases as a power of the absolute fitted values
fm2 <- update(fm1, weights = varPower())



cleanEx()
nameEx("glsControl")
### * glsControl

flush(stderr()); flush(stdout())

### Name: glsControl
### Title: Control Values for gls Fit
### Aliases: glsControl
### Keywords: models

### ** Examples

# decrease the maximum number of iterations and request tracing
glsControl(msMaxIter = 20, msVerbose = TRUE)



cleanEx()
nameEx("glsStruct")
### * glsStruct

flush(stderr()); flush(stdout())

### Name: glsStruct
### Title: Generalized Least Squares Structure
### Aliases: glsStruct
### Keywords: models

### ** Examples

gls1 <- glsStruct(corAR1(), varPower())



cleanEx()
nameEx("gnls")
### * gnls

flush(stderr()); flush(stdout())

### Name: gnls
### Title: Fit Nonlinear Model Using Generalized Least Squares
### Aliases: gnls
### Keywords: models

### ** Examples

# variance increases with a power of the absolute fitted values
fm1 <- gnls(weight ~ SSlogis(Time, Asym, xmid, scal), Soybean,
            weights = varPower())
summary(fm1)



cleanEx()
nameEx("gnlsControl")
### * gnlsControl

flush(stderr()); flush(stdout())

### Name: gnlsControl
### Title: Control Values for gnls Fit
### Aliases: gnlsControl
### Keywords: models

### ** Examples

# decrease the maximum number of iterations and request tracing
gnlsControl(msMaxIter = 20, msVerbose = TRUE)



cleanEx()
nameEx("gnlsStruct")
### * gnlsStruct

flush(stderr()); flush(stdout())

### Name: gnlsStruct
### Title: Generalized Nonlinear Least Squares Structure
### Aliases: gnlsStruct Initialize.gnlsStruct
### Keywords: models

### ** Examples

gnls1 <- gnlsStruct(corAR1(), varPower())



cleanEx()
nameEx("groupedData")
### * groupedData

flush(stderr()); flush(stdout())

### Name: groupedData
### Title: Construct a groupedData Object
### Aliases: groupedData [.groupedData as.data.frame.groupedData
###   update.groupedData
### Keywords: manip attribute

### ** Examples


Orth.new <-  # create a new copy of the groupedData object
  groupedData( distance ~ age | Subject,
              data = as.data.frame( Orthodont ),
              FUN = mean,
              outer = ~ Sex,
              labels = list( x = "Age",
                y = "Distance from pituitary to pterygomaxillary fissure" ),
              units = list( x = "(yr)", y = "(mm)") )
plot( Orth.new )         # trellis plot by Subject
formula( Orth.new )      # extractor for the formula
gsummary( Orth.new )     # apply summary by Subject
fm1 <- lme( Orth.new )   # fixed and groups formulae extracted from object
Orthodont2 <- update(Orthodont, FUN = mean)



cleanEx()
nameEx("gsummary")
### * gsummary

flush(stderr()); flush(stdout())

### Name: gsummary
### Title: Summarize by Groups
### Aliases: gsummary
### Keywords: manip

### ** Examples

gsummary(Orthodont)  # default summary by Subject
## gsummary with invariantsOnly = TRUE and omitGroupingFactor = TRUE
## determines whether there are covariates like Sex that are invariant
## within the repeated observations on the same Subject.
gsummary(Orthodont, inv = TRUE, omit = TRUE)



cleanEx()
nameEx("intervals")
### * intervals

flush(stderr()); flush(stdout())

### Name: intervals
### Title: Confidence Intervals on Coefficients
### Aliases: intervals
### Keywords: models

### ** Examples

## see the method documentation



cleanEx()
nameEx("intervals.gls")
### * intervals.gls

flush(stderr()); flush(stdout())

### Name: intervals.gls
### Title: Confidence Intervals on gls Parameters
### Aliases: intervals.gls print.intervals.gls
### Keywords: models

### ** Examples

fm1 <- gls(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary,
           correlation = corAR1(form = ~ 1 | Mare))
intervals(fm1)



cleanEx()
nameEx("intervals.lmList")
### * intervals.lmList

flush(stderr()); flush(stdout())

### Name: intervals.lmList
### Title: Confidence Intervals on lmList Coefficients
### Aliases: intervals.lmList print.intervals.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
intervals(fm1)



cleanEx()
nameEx("intervals.lme")
### * intervals.lme

flush(stderr()); flush(stdout())

### Name: intervals.lme
### Title: Confidence Intervals on lme Parameters
### Aliases: intervals.lme print.intervals.lme
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age, Orthodont, random = ~ age | Subject)
intervals(fm1)



cleanEx()
nameEx("isBalanced")
### * isBalanced

flush(stderr()); flush(stdout())

### Name: isBalanced
### Title: Check a Design for Balance
### Aliases: isBalanced isBalanced.groupedData
### Keywords: data

### ** Examples

isBalanced(Orthodont)                    # should return TRUE
isBalanced(Orthodont, countOnly = TRUE)  # should return TRUE
isBalanced(Pixel)                        # should return FALSE
isBalanced(Pixel, level = 1)             # should return FALSE



cleanEx()
nameEx("isInitialized")
### * isInitialized

flush(stderr()); flush(stdout())

### Name: isInitialized
### Title: Check if Object is Initialized
### Aliases: isInitialized isInitialized.pdMat isInitialized.pdBlocked
### Keywords: models

### ** Examples

pd1 <- pdDiag(~age)
isInitialized(pd1)



cleanEx()
nameEx("lmList")
### * lmList

flush(stderr()); flush(stdout())

### Name: lmList
### Title: List of lm Objects with a Common Model
### Aliases: lmList lmList.formula print.lmList update.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
summary(fm1)



cleanEx()
nameEx("lmList.groupedData")
### * lmList.groupedData

flush(stderr()); flush(stdout())

### Name: lmList.groupedData
### Title: lmList Fit from a groupedData Object
### Aliases: lmList.groupedData
### Keywords: models

### ** Examples

fm1 <- lmList(Orthodont)
summary(fm1)



cleanEx()
nameEx("lme")
### * lme

flush(stderr()); flush(stdout())

### Name: lme
### Title: Linear Mixed-Effects Models
### Aliases: lme lme.formula update.lme
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age, data = Orthodont) # random is ~ age
fm2 <- lme(distance ~ age + Sex, data = Orthodont, random = ~ 1)
summary(fm1)
summary(fm2)



cleanEx()
nameEx("lme.groupedData")
### * lme.groupedData

flush(stderr()); flush(stdout())

### Name: lme.groupedData
### Title: LME fit from groupedData Object
### Aliases: lme.groupedData
### Keywords: models

### ** Examples

fm1 <- lme(Orthodont)
summary(fm1)



cleanEx()
nameEx("lme.lmList")
### * lme.lmList

flush(stderr()); flush(stdout())

### Name: lme.lmList
### Title: LME fit from lmList Object
### Aliases: lme.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(Orthodont)
fm2 <- lme(fm1)
summary(fm1)
summary(fm2)



cleanEx()
nameEx("lmeControl")
### * lmeControl

flush(stderr()); flush(stdout())

### Name: lmeControl
### Title: Specifying Control Values for lme Fit
### Aliases: lmeControl
### Keywords: models

### ** Examples

# decrease the maximum number iterations in the ms call and
# request that information on the evolution of the ms iterations be printed
str(lCtr <- lmeControl(msMaxIter = 20, msVerbose = TRUE))
## This should always work:
do.call(lmeControl, lCtr)



cleanEx()
nameEx("lmeStruct")
### * lmeStruct

flush(stderr()); flush(stdout())

### Name: lmeStruct
### Title: Linear Mixed-Effects Structure
### Aliases: lmeStruct
### Keywords: models

### ** Examples

lms1 <- lmeStruct(reStruct(~age), corAR1(), varPower())



cleanEx()
nameEx("logDet")
### * logDet

flush(stderr()); flush(stdout())

### Name: logDet
### Title: Extract the Logarithm of the Determinant
### Aliases: logDet
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("logDet.corStruct")
### * logDet.corStruct

flush(stderr()); flush(stdout())

### Name: logDet.corStruct
### Title: Extract corStruct Log-Determinant
### Aliases: logDet.corStruct
### Keywords: models

### ** Examples

cs1 <- corAR1(0.3)
logDet(cs1, covariate = 1:4)



cleanEx()
nameEx("logDet.pdMat")
### * logDet.pdMat

flush(stderr()); flush(stdout())

### Name: logDet.pdMat
### Title: Extract Log-Determinant from a pdMat Object
### Aliases: logDet.pdMat logDet.pdBlocked logDet.pdCompSymm logDet.pdDiag
###   logDet.pdIdent logDet.pdNatural logDet.pdSymm
### Keywords: models

### ** Examples

pd1 <- pdSymm(diag(1:3))
logDet(pd1)



cleanEx()
nameEx("logDet.reStruct")
### * logDet.reStruct

flush(stderr()); flush(stdout())

### Name: logDet.reStruct
### Title: Extract reStruct Log-Determinants
### Aliases: logDet.reStruct
### Keywords: models

### ** Examples

rs1 <- reStruct(list(A = pdSymm(diag(1:3), form = ~Score),
  B = pdDiag(2 * diag(4), form = ~Educ)))
logDet(rs1)



cleanEx()
nameEx("logLik.corStruct")
### * logLik.corStruct

flush(stderr()); flush(stdout())

### Name: logLik.corStruct
### Title: Extract corStruct Log-Likelihood
### Aliases: logLik.corStruct
### Keywords: models

### ** Examples

cs1 <- corAR1(0.2)
cs1 <- Initialize(cs1, data = Orthodont)
logLik(cs1)



cleanEx()
nameEx("logLik.gnls")
### * logLik.gnls

flush(stderr()); flush(stdout())

### Name: logLik.gnls
### Title: Log-Likelihood of a gnls Object
### Aliases: logLik.gnls
### Keywords: models

### ** Examples

fm1 <- gnls(weight ~ SSlogis(Time, Asym, xmid, scal), Soybean,
            weights = varPower())
logLik(fm1)



cleanEx()
nameEx("logLik.lmList")
### * logLik.lmList

flush(stderr()); flush(stdout())

### Name: logLik.lmList
### Title: Log-Likelihood of an lmList Object
### Aliases: logLik.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
logLik(fm1)   # returns NA when it should not



cleanEx()
nameEx("logLik.lme")
### * logLik.lme

flush(stderr()); flush(stdout())

### Name: logLik.lme
### Title: Log-Likelihood of an lme Object
### Aliases: logLik.lme logLik.gls
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ Sex * age, Orthodont, random = ~ age, method = "ML")
logLik(fm1)
logLik(fm1, REML = TRUE)



cleanEx()
nameEx("logLik.varFunc")
### * logLik.varFunc

flush(stderr()); flush(stdout())

### Name: logLik.varFunc
### Title: Extract varFunc logLik
### Aliases: logLik.varFunc logLik.varComb
### Keywords: models

### ** Examples

vf1 <- varPower(form = ~age)
vf1 <- Initialize(vf1, Orthodont)
coef(vf1) <- 0.1
logLik(vf1)



cleanEx()
nameEx("model.matrix.reStruct")
### * model.matrix.reStruct

flush(stderr()); flush(stdout())

### Name: model.matrix.reStruct
### Title: reStruct Model Matrix
### Aliases: model.matrix.reStruct
### Keywords: models

### ** Examples

rs1 <- reStruct(list(Dog = ~day, Side = ~1), data = Pixel)
model.matrix(rs1, Pixel)



cleanEx()
nameEx("needUpdate")
### * needUpdate

flush(stderr()); flush(stdout())

### Name: needUpdate
### Title: Check if Update is Needed
### Aliases: needUpdate needUpdate.default needUpdate.varComb
###   needUpdate.varIdent
### Keywords: models

### ** Examples

vf1 <- varExp()
vf1 <- Initialize(vf1, data = Orthodont)
needUpdate(vf1)



cleanEx()
nameEx("needUpdate.modelStruct")
### * needUpdate.modelStruct

flush(stderr()); flush(stdout())

### Name: needUpdate.modelStruct
### Title: Check if a modelStruct Object Needs Updating
### Aliases: needUpdate.modelStruct needUpdate.corStruct
###   needUpdate.reStruct
### Keywords: models

### ** Examples

lms1 <- lmeStruct(reStruct = reStruct(pdDiag(diag(2), ~age)),
   varStruct = varPower(form = ~age))
needUpdate(lms1)



cleanEx()
nameEx("nlme-deprecated")
### * nlme-deprecated

flush(stderr()); flush(stdout())

### Name: nlme-deprecated
### Title: Deprecated Functions in Package 'nlme'
### Aliases: nlme-deprecated nfGroupedData nmGroupedData corIdent
### Keywords: internal

### ** Examples

assertDeprecation <- function(expr)
  tools::assertCondition(expr, verbose = TRUE,
    if(getRversion() >= "3.6.0") "deprecatedWarning" else "warning")

assertDeprecation(
  nlme::nfGroupedData(height ~ age | Subject, as.data.frame(Oxboys))
)
assertDeprecation( csId <-  corIdent(~ 1 | Subject) )
assertDeprecation( csI. <- Initialize(csId, data = Orthodont) )
assertDeprecation( corMatrix(csI.) )  # actually errors



cleanEx()
nameEx("nlme")
### * nlme

flush(stderr()); flush(stdout())

### Name: nlme
### Title: Nonlinear Mixed-Effects Models
### Aliases: nlme nlme.formula
### Keywords: models

### ** Examples

fm1 <- nlme(height ~ SSasymp(age, Asym, R0, lrc),
            data = Loblolly,
            fixed = Asym + R0 + lrc ~ 1,
            random = Asym ~ 1,
            start = c(Asym = 103, R0 = -8.5, lrc = -3.3))
summary(fm1)
fm2 <- update(fm1, random = pdDiag(Asym + lrc ~ 1))
summary(fm2)



cleanEx()
nameEx("nlme.nlsList")
### * nlme.nlsList

flush(stderr()); flush(stdout())

### Name: nlme.nlsList
### Title: NLME fit from nlsList Object
### Aliases: nlme.nlsList
### Keywords: models

### ** Examples

fm1 <- nlsList(SSasymp, data = Loblolly)
fm2 <- nlme(fm1, random = Asym ~ 1)
summary(fm1)
summary(fm2)



cleanEx()
nameEx("nlmeControl")
### * nlmeControl

flush(stderr()); flush(stdout())

### Name: nlmeControl
### Title: Control Values for nlme Fit
### Aliases: nlmeControl
### Keywords: models

### ** Examples

# decrease the maximum number of iterations and request tracing
nlmeControl(msMaxIter = 20, msVerbose = TRUE)



cleanEx()
nameEx("nlmeStruct")
### * nlmeStruct

flush(stderr()); flush(stdout())

### Name: nlmeStruct
### Title: Nonlinear Mixed-Effects Structure
### Aliases: nlmeStruct
### Keywords: models

### ** Examples

nlms1 <- nlmeStruct(reStruct(~age), corAR1(), varPower())



cleanEx()
nameEx("nlsList")
### * nlsList

flush(stderr()); flush(stdout())

### Name: nlsList
### Title: List of nls Objects with a Common Model
### Aliases: nlsList nlsList.formula update.nlsList
### Keywords: models

### ** Examples

fm1 <- nlsList(uptake ~ SSasympOff(conc, Asym, lrc, c0),
   data = CO2, start = c(Asym = 30, lrc = -4.5, c0 = 52))
summary(fm1)
cfm1 <- confint(fm1) # via profiling each % FIXME: only *one* message instead of one *each*
mat.class <- class(matrix(1)) # ("matrix", "array") for R >= 4.0.0;  ("matrix" in older R)
i.ok <- which(vapply(cfm1,
                function(r) identical(class(r), mat.class), NA))
stopifnot(length(i.ok) > 0, !anyNA(match(c(2:4, 6:9, 12), i.ok)))
## where as (some of) the others gave errors during profile re-fitting :
str(cfm1[- i.ok])



cleanEx()
nameEx("nlsList.selfStart")
### * nlsList.selfStart

flush(stderr()); flush(stdout())

### Name: nlsList.selfStart
### Title: nlsList Fit from a selfStart Function
### Aliases: nlsList.selfStart
### Keywords: models

### ** Examples

fm1 <- nlsList(SSasympOff, CO2)
summary(fm1)



cleanEx()
nameEx("pairs.compareFits")
### * pairs.compareFits

flush(stderr()); flush(stdout())

### Name: pairs.compareFits
### Title: Pairs Plot of compareFits Object
### Aliases: pairs.compareFits
### Keywords: models

### ** Examples

example(compareFits) # cF12 <- compareFits(coef(lmList(Orthodont)), .. lme(*))
pairs(cF12)



cleanEx()
nameEx("pairs.lmList")
### * pairs.lmList

flush(stderr()); flush(stdout())

### Name: pairs.lmList
### Title: Pairs Plot of an lmList Object
### Aliases: pairs.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)

# scatter plot of coefficients by gender, identifying unusual subjects
pairs(fm1, ~coef(.) | Sex, id = 0.1, adj = -0.5)

# scatter plot of estimated random effects -- "bivariate Gaussian (?)"
pairs(fm1, ~ranef(.))



cleanEx()
nameEx("pairs.lme")
### * pairs.lme

flush(stderr()); flush(stdout())

### Name: pairs.lme
### Title: Pairs Plot of an lme Object
### Aliases: pairs.lme
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age, Orthodont, random = ~ age | Subject)

# scatter plot of coefficients by gender, identifying unusual subjects
pairs(fm1, ~coef(., augFrame = TRUE) | Sex, id = 0.1, adj = -0.5)    

# scatter plot of estimated random effects :
pairs(fm1, ~ranef(.))



cleanEx()
nameEx("pdBlocked")
### * pdBlocked

flush(stderr()); flush(stdout())

### Name: pdBlocked
### Title: Positive-Definite Block Diagonal Matrix
### Aliases: pdBlocked
### Keywords: models

### ** Examples

pd1 <- pdBlocked(list(diag(1:2), diag(c(0.1, 0.2, 0.3))),
                 nam = list(c("A","B"), c("a1", "a2", "a3")))
pd1



cleanEx()
nameEx("pdCompSymm")
### * pdCompSymm

flush(stderr()); flush(stdout())

### Name: pdCompSymm
### Title: Positive-Definite Matrix with Compound Symmetry Structure
### Aliases: pdCompSymm
### Keywords: models

### ** Examples

pd1 <- pdCompSymm(diag(3) + 1, nam = c("A","B","C"))
pd1



cleanEx()
nameEx("pdConstruct")
### * pdConstruct

flush(stderr()); flush(stdout())

### Name: pdConstruct
### Title: Construct pdMat Objects
### Aliases: pdConstruct pdConstruct.pdCompSymm pdConstruct.pdDiag
###   pdConstruct.pdIdent pdConstruct.pdMat pdConstruct.pdNatural
###   pdConstruct.pdSymm pdConstruct.pdLogChol
### Keywords: models

### ** Examples

pd1 <- pdSymm()
pdConstruct(pd1, diag(1:4))



cleanEx()
nameEx("pdConstruct.pdBlocked")
### * pdConstruct.pdBlocked

flush(stderr()); flush(stdout())

### Name: pdConstruct.pdBlocked
### Title: Construct pdBlocked Objects
### Aliases: pdConstruct.pdBlocked
### Keywords: models

### ** Examples

pd1 <- pdBlocked(list(c("A","B"), c("a1", "a2", "a3")))
pdConstruct(pd1, list(diag(1:2), diag(c(0.1, 0.2, 0.3))))



cleanEx()
nameEx("pdDiag")
### * pdDiag

flush(stderr()); flush(stdout())

### Name: pdDiag
### Title: Diagonal Positive-Definite Matrix
### Aliases: pdDiag
### Keywords: models

### ** Examples

pd1 <- pdDiag(diag(1:3), nam = c("A","B","C"))
pd1



cleanEx()
nameEx("pdFactor")
### * pdFactor

flush(stderr()); flush(stdout())

### Name: pdFactor
### Title: Square-Root Factor of a Positive-Definite Matrix
### Aliases: pdFactor pdFactor.pdBlocked pdFactor.pdCompSymm
###   pdFactor.pdDiag pdFactor.pdIdent pdFactor.pdMat pdFactor.pdNatural
###   pdFactor.pdSymm pdFactor.pdLogChol
### Keywords: models

### ** Examples

pd1 <- pdCompSymm(4 * diag(3) + 1)
pdFactor(pd1)



cleanEx()
nameEx("pdFactor.reStruct")
### * pdFactor.reStruct

flush(stderr()); flush(stdout())

### Name: pdFactor.reStruct
### Title: Extract Square-Root Factor from Components of an reStruct Object
### Aliases: pdFactor.reStruct
### Keywords: models

### ** Examples

rs1 <- reStruct(pdSymm(diag(3), ~age+Sex, data = Orthodont))
pdFactor(rs1)



cleanEx()
nameEx("pdIdent")
### * pdIdent

flush(stderr()); flush(stdout())

### Name: pdIdent
### Title: Multiple of the Identity Positive-Definite Matrix
### Aliases: pdIdent
### Keywords: models

### ** Examples

pd1 <- pdIdent(4 * diag(3), nam = c("A","B","C"))
pd1



cleanEx()
nameEx("pdLogChol")
### * pdLogChol

flush(stderr()); flush(stdout())

### Name: pdLogChol
### Title: General Positive-Definite Matrix
### Aliases: pdLogChol
### Keywords: models

### ** Examples

(pd1 <- pdLogChol(diag(1:3), nam = c("A","B","C")))

(pd4 <- pdLogChol(1:6))
(pd4c <- chol(pd4)) # -> upper-tri matrix with off-diagonals  4 5 6
pd4c[upper.tri(pd4c)]
log(diag(pd4c)) # 1 2 3



cleanEx()
nameEx("pdMat")
### * pdMat

flush(stderr()); flush(stdout())

### Name: pdMat
### Title: Positive-Definite Matrix
### Aliases: pdMat plot.pdMat
### Keywords: models

### ** Examples

pd1 <- pdMat(diag(1:4), pdClass = "pdDiag")
pd1
str(pd1)



cleanEx()
nameEx("pdMatrix")
### * pdMatrix

flush(stderr()); flush(stdout())

### Name: pdMatrix
### Title: Extract Matrix or Square-Root Factor from a pdMat Object
### Aliases: pdMatrix pdMatrix.pdBlocked pdMatrix.pdCompSymm
###   pdMatrix.pdDiag pdMatrix.pdIdent pdMatrix.pdMat pdMatrix.pdSymm
###   pdMatrix.pdNatural
### Keywords: models

### ** Examples

pd1 <- pdSymm(diag(1:4))
pdMatrix(pd1)



cleanEx()
nameEx("pdMatrix.reStruct")
### * pdMatrix.reStruct

flush(stderr()); flush(stdout())

### Name: pdMatrix.reStruct
### Title: Extract Matrix or Square-Root Factor from Components of an
###   reStruct Object
### Aliases: pdMatrix.reStruct
### Keywords: models

### ** Examples

rs1 <- reStruct(pdSymm(diag(3), ~age+Sex, data = Orthodont))
pdMatrix(rs1)



cleanEx()
nameEx("pdNatural")
### * pdNatural

flush(stderr()); flush(stdout())

### Name: pdNatural
### Title: General Positive-Definite Matrix in Natural Parametrization
### Aliases: pdNatural
### Keywords: models

### ** Examples

pdNatural(diag(1:3))



cleanEx()
nameEx("pdSymm")
### * pdSymm

flush(stderr()); flush(stdout())

### Name: pdSymm
### Title: General Positive-Definite Matrix
### Aliases: pdSymm
### Keywords: models

### ** Examples

pd1 <- pdSymm(diag(1:3), nam = c("A","B","C"))
pd1



cleanEx()
nameEx("plot.ACF")
### * plot.ACF

flush(stderr()); flush(stdout())

### Name: plot.ACF
### Title: Plot an ACF Object
### Aliases: plot.ACF
### Keywords: models

### ** Examples

fm1 <- lme(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary)
plot(ACF(fm1, maxLag = 10), alpha = 0.01)



cleanEx()
nameEx("plot.Variogram")
### * plot.Variogram

flush(stderr()); flush(stdout())

### Name: plot.Variogram
### Title: Plot a Variogram Object
### Aliases: plot.Variogram
### Keywords: models

### ** Examples

fm1 <- lme(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary)
plot(Variogram(fm1, form = ~ Time | Mare, maxDist = 0.7))



cleanEx()
nameEx("plot.augPred")
### * plot.augPred

flush(stderr()); flush(stdout())

### Name: plot.augPred
### Title: Plot an augPred Object
### Aliases: plot.augPred
### Keywords: models

### ** Examples

fm1 <- lme(Orthodont)
plot(augPred(fm1, level = 0:1, length.out = 2))



cleanEx()
nameEx("plot.compareFits")
### * plot.compareFits

flush(stderr()); flush(stdout())

### Name: plot.compareFits
### Title: Plot a compareFits Object
### Aliases: plot.compareFits
### Keywords: models

### ** Examples

example(compareFits) # cF12 <- compareFits(coef(lmList(Orthodont)), .. lme(*))
plot(cF12)



cleanEx()
nameEx("plot.gls")
### * plot.gls

flush(stderr()); flush(stdout())

### Name: plot.gls
### Title: Plot a gls Object
### Aliases: plot.gls
### Keywords: models

### ** Examples

fm1 <- gls(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary,
           correlation = corAR1(form = ~ 1 | Mare))
# standardized residuals versus fitted values by Mare
plot(fm1, resid(., type = "p") ~ fitted(.) | Mare, abline = 0)
# box-plots of residuals by Mare
plot(fm1, Mare ~ resid(.))
# observed versus fitted values by Mare
plot(fm1, follicles ~ fitted(.) | Mare, abline = c(0,1))



cleanEx()
nameEx("plot.intervals.lmList")
### * plot.intervals.lmList

flush(stderr()); flush(stdout())

### Name: plot.intervals.lmList
### Title: Plot lmList Confidence Intervals
### Aliases: plot.intervals.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
plot(intervals(fm1))



cleanEx()
nameEx("plot.lmList")
### * plot.lmList

flush(stderr()); flush(stdout())

### Name: plot.lmList
### Title: Plot an lmList Object
### Aliases: plot.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
# standardized residuals versus fitted values by gender
plot(fm1, resid(., type = "pool") ~ fitted(.) | Sex, abline = 0, id = 0.05)
# box-plots of residuals by Subject
plot(fm1, Subject ~ resid(.))
# observed versus fitted values by Subject
plot(fm1, distance ~ fitted(.) | Subject, abline = c(0,1))



cleanEx()
nameEx("plot.lme")
### * plot.lme

flush(stderr()); flush(stdout())

### Name: plot.lme
### Title: Plot an lme or nls object
### Aliases: plot.lme plot.nls
### Keywords: models hplot

### ** Examples

fm1 <- lme(distance ~ age, Orthodont, random = ~ age | Subject)
# standardized residuals versus fitted values by gender
plot(fm1, resid(., type = "p") ~ fitted(.) | Sex, abline = 0)
# box-plots of residuals by Subject
plot(fm1, Subject ~ resid(.))
# observed versus fitted values by Subject
plot(fm1, distance ~ fitted(.) | Subject, abline = c(0,1))



cleanEx()
nameEx("plot.nffGroupedData")
### * plot.nffGroupedData

flush(stderr()); flush(stdout())

### Name: plot.nffGroupedData
### Title: Plot an nffGroupedData Object
### Aliases: plot.nffGroupedData
### Keywords: models

### ** Examples

plot(Machines)
plot(Machines, inner = TRUE)



cleanEx()
nameEx("plot.nfnGroupedData")
### * plot.nfnGroupedData

flush(stderr()); flush(stdout())

### Name: plot.nfnGroupedData
### Title: Plot an nfnGroupedData Object
### Aliases: plot.nfnGroupedData
### Keywords: models

### ** Examples

# different panels per Subject
plot(Orthodont)
# different panels per gender
plot(Orthodont, outer = TRUE)



cleanEx()
nameEx("plot.nmGroupedData")
### * plot.nmGroupedData

flush(stderr()); flush(stdout())

### Name: plot.nmGroupedData
### Title: Plot an nmGroupedData Object
### Aliases: plot.nmGroupedData
### Keywords: models

### ** Examples

# no collapsing, panels by Dog
plot(Pixel, display = "Dog", inner = ~Side)
# collapsing by Dog, preserving day
plot(Pixel, collapse = "Dog", preserve = ~day)



cleanEx()
nameEx("plot.ranef.lmList")
### * plot.ranef.lmList

flush(stderr()); flush(stdout())

### Name: plot.ranef.lmList
### Title: Plot a ranef.lmList Object
### Aliases: plot.ranef.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
plot(ranef(fm1))
fm1RE <- ranef(fm1, aug = TRUE)
plot(fm1RE, form = ~ Sex)
plot(fm1RE, form = age ~ Sex)



cleanEx()
nameEx("plot.ranef.lme")
### * plot.ranef.lme

flush(stderr()); flush(stdout())

### Name: plot.ranef.lme
### Title: Plot a ranef.lme Object
### Aliases: plot.ranef.lme
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age, Orthodont, random = ~ age | Subject)
plot(ranef(fm1))
fm1RE <- ranef(fm1, aug = TRUE)
plot(fm1RE, form = ~ Sex)
plot(fm1RE, form = age ~ Sex) # "connected" boxplots



cleanEx()
nameEx("pooledSD")
### * pooledSD

flush(stderr()); flush(stdout())

### Name: pooledSD
### Title: Extract Pooled Standard Deviation
### Aliases: pooledSD
### Keywords: models

### ** Examples

fm1 <- lmList(Orthodont)
pooledSD(fm1)



cleanEx()
nameEx("predict.gls")
### * predict.gls

flush(stderr()); flush(stdout())

### Name: predict.gls
### Title: Predictions from a gls Object
### Aliases: predict.gls
### Keywords: models

### ** Examples

fm1 <- gls(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary,
           correlation = corAR1(form = ~ 1 | Mare))
newOvary <- data.frame(Time = c(-0.75, -0.5, 0, 0.5, 0.75))
predict(fm1, newOvary)



cleanEx()
nameEx("predict.gnls")
### * predict.gnls

flush(stderr()); flush(stdout())

### Name: predict.gnls
### Title: Predictions from a gnls Object
### Aliases: predict.gnls
### Keywords: models

### ** Examples

fm1 <- gnls(weight ~ SSlogis(Time, Asym, xmid, scal), Soybean,
            weights = varPower())
newSoybean <- data.frame(Time = c(10,30,50,80,100))
predict(fm1, newSoybean)



cleanEx()
nameEx("predict.lmList")
### * predict.lmList

flush(stderr()); flush(stdout())

### Name: predict.lmList
### Title: Predictions from an lmList Object
### Aliases: predict.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
predict(fm1, se.fit = TRUE)



cleanEx()
nameEx("predict.lme")
### * predict.lme

flush(stderr()); flush(stdout())

### Name: predict.lme
### Title: Predictions from an lme Object
### Aliases: predict.lme
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age, Orthodont, random = ~ age | Subject)
newOrth <- data.frame(Sex = c("Male","Male","Female","Female","Male","Male"),
                      age = c(15, 20, 10, 12, 2, 4),
                      Subject = c("M01","M01","F30","F30","M04","M04"))
## The 'Orthodont' data has *no* 'F30', so predict  NA  at level 1 :
predict(fm1, newOrth, level = 0:1)



cleanEx()
nameEx("predict.nlme")
### * predict.nlme

flush(stderr()); flush(stdout())

### Name: predict.nlme
### Title: Predictions from an nlme Object
### Aliases: predict.nlme
### Keywords: models

### ** Examples

head(Loblolly) # groupedData  w/  'Seed' is grouping variable :
## Grouped Data: height ~ age | Seed
##    height age Seed
## 1    4.51   3  301
## 15  10.89   5  301
## ..  .....   .  ...

fm1 <- nlme(height ~ SSasymp(age, Asym, R0, lrc),  data = Loblolly,
            fixed = Asym + R0 + lrc ~ 1,
            random = Asym ~ 1, ## <---grouping--->  Asym ~ 1 | Seed
            start = c(Asym = 103, R0 = -8.5, lrc = -3.3))
fm1

age. <- seq(from = 2, to = 30, by = 2)
newLL.301 <- data.frame(age = age., Seed = 301)
newLL.329 <- data.frame(age = age., Seed = 329)
(p301 <- predict(fm1, newLL.301, level = 0:1))
(p329 <- predict(fm1, newLL.329, level = 0:1))
## Prediction are the same at level 0 :
all.equal(p301[,"predict.fixed"],
          p329[,"predict.fixed"])
## and differ by the 'Seed' effect at level 1 :
p301[,"predict.Seed"] -
p329[,"predict.Seed"]



cleanEx()
nameEx("print.summary.pdMat")
### * print.summary.pdMat

flush(stderr()); flush(stdout())

### Name: print.summary.pdMat
### Title: Print a summary.pdMat Object
### Aliases: print.summary.pdMat
### Keywords: models

### ** Examples

pd1 <- pdCompSymm(3 * diag(2) + 1, form = ~age + age^2,
         data = Orthodont)
print(summary(pd1), sigma = 1.2, resid = TRUE)



cleanEx()
nameEx("print.varFunc")
### * print.varFunc

flush(stderr()); flush(stdout())

### Name: print.varFunc
### Title: Print a varFunc Object
### Aliases: print.varFunc print.varComb
### Keywords: models

### ** Examples

vf1 <- varPower(0.3, form = ~age)
vf1 <- Initialize(vf1, Orthodont)
print(vf1)



cleanEx()
nameEx("qqnorm.gls")
### * qqnorm.gls

flush(stderr()); flush(stdout())

### Name: qqnorm.gls
### Title: Normal Plot of Residuals from a gls Object
### Aliases: qqnorm.gls
### Keywords: models

### ** Examples

fm1 <- gls(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary,
           correlation = corAR1(form = ~ 1 | Mare))
qqnorm(fm1, abline = c(0,1))



cleanEx()
nameEx("qqnorm.lme")
### * qqnorm.lme

flush(stderr()); flush(stdout())

### Name: qqnorm.lme
### Title: Normal Plot of Residuals or Random Effects from an lme Object
### Aliases: qqnorm.lm qqnorm.lme qqnorm.lmList qqnorm.nls
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age, Orthodont, random = ~ age | Subject)
## normal plot of standardized residuals by gender
qqnorm(fm1, ~ resid(., type = "p") | Sex, abline = c(0, 1))
## normal plots of random effects
qqnorm(fm1, ~ranef(.))



cleanEx()
nameEx("random.effects")
### * random.effects

flush(stderr()); flush(stdout())

### Name: random.effects
### Title: Extract Random Effects
### Aliases: random.effects ranef print.ranef
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("ranef.lmList")
### * ranef.lmList

flush(stderr()); flush(stdout())

### Name: ranef.lmList
### Title: Extract lmList Random Effects
### Aliases: random.effects.lmList ranef.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
ranef(fm1)
random.effects(fm1)              # same as above



cleanEx()
nameEx("ranef.lme")
### * ranef.lme

flush(stderr()); flush(stdout())

### Name: ranef.lme
### Title: Extract lme Random Effects
### Aliases: ranef.lme random.effects.lme print.ranef.lme
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age, Orthodont, random = ~ age | Subject)
ranef(fm1)
random.effects(fm1)             # same as above
random.effects(fm1, augFrame = TRUE)



cleanEx()
nameEx("reStruct")
### * reStruct

flush(stderr()); flush(stdout())

### Name: reStruct
### Title: Random Effects Structure
### Aliases: reStruct [.reStruct print.reStruct
### Keywords: models

### ** Examples

rs1 <- reStruct(list(Dog = ~day, Side = ~1), data = Pixel)
rs1 # 2 entries "Uninitialized"
str(rs1) # a bit more



cleanEx()
nameEx("recalc")
### * recalc

flush(stderr()); flush(stdout())

### Name: recalc
### Title: Recalculate Condensed Linear Model Object
### Aliases: recalc
### Keywords: models

### ** Examples

## see the method function documentation



cleanEx()
nameEx("residuals.gls")
### * residuals.gls

flush(stderr()); flush(stdout())

### Name: residuals.gls
### Title: Extract gls Residuals
### Aliases: residuals.gls residuals.gnls
### Keywords: models

### ** Examples

fm1 <- gls(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary,
           correlation = corAR1(form = ~ 1 | Mare))
residuals(fm1)



cleanEx()
nameEx("residuals.lmList")
### * residuals.lmList

flush(stderr()); flush(stdout())

### Name: residuals.lmList
### Title: Extract lmList Residuals
### Aliases: residuals.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
residuals(fm1)



cleanEx()
nameEx("residuals.lme")
### * residuals.lme

flush(stderr()); flush(stdout())

### Name: residuals.lme
### Title: Extract lme Residuals
### Aliases: residuals.lme
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age + Sex, data = Orthodont, random = ~ 1)
head(residuals(fm1, level = 0:1))
summary(residuals(fm1) /
        residuals(fm1, type = "p")) # constant scaling factor 1.432



cleanEx()
nameEx("simulate.lme")
### * simulate.lme

flush(stderr()); flush(stdout())

### Name: simulate.lme
### Title: Simulate Results from 'lme' Models
### Aliases: simulate.lme plot.simulate.lme print.simulate.lme
### Keywords: models

### ** Examples




cleanEx()
nameEx("solve.pdMat")
### * solve.pdMat

flush(stderr()); flush(stdout())

### Name: solve.pdMat
### Title: Calculate Inverse of a Positive-Definite Matrix
### Aliases: solve.pdMat solve.pdBlocked solve.pdDiag solve.pdIdent
###   solve.pdLogChol solve.pdNatural solve.pdSymm
### Keywords: models

### ** Examples

pd1 <- pdCompSymm(3 * diag(3) + 1)
solve(pd1)



cleanEx()
nameEx("solve.reStruct")
### * solve.reStruct

flush(stderr()); flush(stdout())

### Name: solve.reStruct
### Title: Apply Solve to an reStruct Object
### Aliases: solve.reStruct
### Keywords: models

### ** Examples

rs1 <- reStruct(list(A = pdSymm(diag(1:3), form = ~Score),
  B = pdDiag(2 * diag(4), form = ~Educ)))
solve(rs1)



cleanEx()
nameEx("splitFormula")
### * splitFormula

flush(stderr()); flush(stdout())

### Name: splitFormula
### Title: Split a Formula
### Aliases: splitFormula
### Keywords: models

### ** Examples

splitFormula(~ g1/g2/g3)



cleanEx()
nameEx("summary.corStruct")
### * summary.corStruct

flush(stderr()); flush(stdout())

### Name: summary.corStruct
### Title: Summarize a corStruct Object
### Aliases: summary.corStruct summary.corAR1 summary.corARMA
###   summary.corCAR1 summary.corCompSymm summary.corExp summary.corGaus
###   summary.corLin summary.corNatural summary.corRatio summary.corSpher
###   summary.corSymm
### Keywords: models

### ** Examples

cs1 <- corAR1(0.2)
summary(cs1)



cleanEx()
nameEx("summary.gls")
### * summary.gls

flush(stderr()); flush(stdout())

### Name: summary.gls
### Title: Summarize a Generalized Least Squares 'gls' Object
### Aliases: summary.gls
### Keywords: models

### ** Examples

fm1 <- gls(follicles ~ sin(2*pi*Time) + cos(2*pi*Time), Ovary,
           correlation = corAR1(form = ~ 1 | Mare))
summary(fm1)
coef(summary(fm1)) # "the matrix"



cleanEx()
nameEx("summary.lmList")
### * summary.lmList

flush(stderr()); flush(stdout())

### Name: summary.lmList
### Title: Summarize an lmList Object
### Aliases: summary.lmList
### Keywords: models

### ** Examples

fm1 <- lmList(distance ~ age | Subject, Orthodont)
summary(fm1)



cleanEx()
nameEx("summary.lme")
### * summary.lme

flush(stderr()); flush(stdout())

### Name: summary.lme
### Title: Summarize an lme Object
### Aliases: summary.lme print.summary.lme
### Keywords: models

### ** Examples

fm1 <- lme(distance ~ age, Orthodont, random = ~ age | Subject)
(s1 <- summary(fm1))
## Don't show: 
stopifnot(is.matrix(coef(s1)))
## End(Don't show)



cleanEx()
nameEx("summary.modelStruct")
### * summary.modelStruct

flush(stderr()); flush(stdout())

### Name: summary.modelStruct
### Title: Summarize a modelStruct Object
### Aliases: summary.modelStruct summary.reStruct
### Keywords: models

### ** Examples

lms1 <- lmeStruct(reStruct = reStruct(pdDiag(diag(2), ~age)),
   corStruct = corAR1(0.3))
summary(lms1)



cleanEx()
nameEx("summary.nlsList")
### * summary.nlsList

flush(stderr()); flush(stdout())

### Name: summary.nlsList
### Title: Summarize an nlsList Object
### Aliases: summary.nlsList
### Keywords: models

### ** Examples

fm1 <- nlsList(SSasymp, Loblolly)
summary(fm1)



cleanEx()
nameEx("summary.pdMat")
### * summary.pdMat

flush(stderr()); flush(stdout())

### Name: summary.pdMat
### Title: Summarize a pdMat Object
### Aliases: summary.pdMat summary.pdBlocked summary.pdCompSymm
###   summary.pdDiag summary.pdIdent summary.pdNatural summary.pdSymm
###   summary.pdLogChol
### Keywords: models

### ** Examples

summary(pdSymm(diag(4)))



cleanEx()
nameEx("summary.varFunc")
### * summary.varFunc

flush(stderr()); flush(stdout())

### Name: summary.varFunc
### Title: Summarize "varFunc" Object
### Aliases: summary.varFunc summary.varComb summary.varConstPower
###   summary.varConstProp summary.varExp summary.varFixed summary.varIdent
###   summary.varPower
### Keywords: models

### ** Examples

vf1 <- varPower(0.3, form = ~age)
vf1 <- Initialize(vf1, Orthodont)
summary(vf1)



cleanEx()
nameEx("varComb")
### * varComb

flush(stderr()); flush(stdout())

### Name: varComb
### Title: Combination of Variance Functions
### Aliases: varComb
### Keywords: models

### ** Examples

vf1 <- varComb(varIdent(form = ~1|Sex), varPower())



cleanEx()
nameEx("varConstPower")
### * varConstPower

flush(stderr()); flush(stdout())

### Name: varConstPower
### Title: Constant Plus Power Variance Function
### Aliases: varConstPower
### Keywords: models

### ** Examples

vf1 <- varConstPower(1.2, 0.2, form = ~age|Sex)



cleanEx()
nameEx("varConstProp")
### * varConstProp

flush(stderr()); flush(stdout())

### Name: varConstProp
### Title: Constant Plus Proportion Variance Function
### Aliases: varConstProp
### Keywords: models

### ** Examples

# Generate some synthetic data using the two-component error model and use
# different variance functions, also with fixed sigma in order to avoid
# overparameterisation in the case of a constant term in the variance function
times <- c(0, 1, 3, 7, 14, 28, 56, 120)
pred <- 100 * exp(- 0.03 * times)
sd_pred <- sqrt(3^2 + 0.07^2 * pred^2)
n_replicates <- 2

set.seed(123456)
syn_data <- data.frame(
  time = rep(times, each = n_replicates),
  value = rnorm(length(times) * n_replicates,
    rep(pred, each = n_replicates),
    rep(sd_pred, each = n_replicates)))
syn_data$value <- ifelse(syn_data$value < 0, NA, syn_data$value)

f_const <- gnls(value ~ SSasymp(time, 0, parent_0, lrc),
  data = syn_data, na.action = na.omit,
  start = list(parent_0 = 100, lrc = -3))
f_varPower <- gnls(value ~ SSasymp(time, 0, parent_0, lrc),
  data = syn_data, na.action = na.omit,
  start = list(parent_0 = 100, lrc = -3),
  weights = varPower())
f_varConstPower <- gnls(value ~ SSasymp(time, 0, parent_0, lrc),
  data = syn_data, na.action = na.omit,
  start = list(parent_0 = 100, lrc = -3),
  weights = varConstPower())
f_varConstPower_sf <- gnls(value ~ SSasymp(time, 0, parent_0, lrc),
  data = syn_data, na.action = na.omit,
  control = list(sigma = 1),
  start = list(parent_0 = 100, lrc = -3),
  weights = varConstPower())
f_varConstProp <- gnls(value ~ SSasymp(time, 0, parent_0, lrc),
  data = syn_data, na.action = na.omit,
  start = list(parent_0 = 100, lrc = -3),
  weights = varConstProp())
f_varConstProp_sf <- gnls(value ~ SSasymp(time, 0, parent_0, lrc),
  data = syn_data, na.action = na.omit,
  start = list(parent_0 = 100, lrc = -3),
  control = list(sigma = 1),
  weights = varConstProp())

AIC(f_const, f_varPower, f_varConstPower, f_varConstPower_sf,
  f_varConstProp, f_varConstProp_sf)

# The error model parameters 3 and 0.07 are approximately recovered
intervals(f_varConstProp_sf)



cleanEx()
nameEx("varExp")
### * varExp

flush(stderr()); flush(stdout())

### Name: varExp
### Title: Exponential Variance Function
### Aliases: varExp
### Keywords: models

### ** Examples

vf1 <- varExp(0.2, form = ~age|Sex)



cleanEx()
nameEx("varFixed")
### * varFixed

flush(stderr()); flush(stdout())

### Name: varFixed
### Title: Fixed Variance Function
### Aliases: varFixed
### Keywords: models

### ** Examples

vf1 <- varFixed(~age)



cleanEx()
nameEx("varFunc")
### * varFunc

flush(stderr()); flush(stdout())

### Name: varFunc
### Title: Variance Function Structure
### Aliases: varFunc
### Keywords: models

### ** Examples

vf1 <- varFunc(~age)



cleanEx()
nameEx("varIdent")
### * varIdent

flush(stderr()); flush(stdout())

### Name: varIdent
### Title: Constant Variance Function
### Aliases: varIdent
### Keywords: models

### ** Examples

vf1 <- varIdent(c(Female = 0.5), form = ~ 1 | Sex)



cleanEx()
nameEx("varPower")
### * varPower

flush(stderr()); flush(stdout())

### Name: varPower
### Title: Power Variance Function
### Aliases: varPower
### Keywords: models

### ** Examples

vf1 <- varPower(0.2, form = ~age|Sex)



cleanEx()
nameEx("varWeights")
### * varWeights

flush(stderr()); flush(stdout())

### Name: varWeights
### Title: Extract Variance Function Weights
### Aliases: varWeights varWeights.varComb varWeights.varFunc
### Keywords: models

### ** Examples

vf1 <- varPower(form=~age)
vf1 <- Initialize(vf1, Orthodont)
coef(vf1) <- 0.3
varWeights(vf1)[1:10]



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
