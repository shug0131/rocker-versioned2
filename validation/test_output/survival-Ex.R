pkgname <- "survival"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('survival')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("Surv")
### * Surv

flush(stderr()); flush(stdout())

### Name: Surv
### Title: Create a Survival Object
### Aliases: Surv is.Surv [.Surv
### Keywords: survival

### ** Examples

with(aml, Surv(time, status))
survfit(Surv(time, status) ~ ph.ecog, data=lung)
Surv(heart$start, heart$stop, heart$event) 



cleanEx()
nameEx("aareg")
### * aareg

flush(stderr()); flush(stdout())

### Name: aareg
### Title: Aalen's additive regression model for censored data
### Aliases: aareg
### Keywords: survival

### ** Examples

# Fit a model to the lung cancer data set
lfit <- aareg(Surv(time, status) ~ age + sex + ph.ecog, data=lung,
                     nmin=1)
## Not run: 
##D lfit
##D Call:
##D aareg(formula = Surv(time, status) ~ age + sex + ph.ecog, data = lung, nmin = 1
##D         )
##D 
##D   n=227 (1 observations deleted due to missing values)
##D     138 out of 138 unique event times used
##D 
##D               slope      coef se(coef)     z        p 
##D Intercept  5.26e-03  5.99e-03 4.74e-03  1.26 0.207000
##D       age  4.26e-05  7.02e-05 7.23e-05  0.97 0.332000
##D       sex -3.29e-03 -4.02e-03 1.22e-03 -3.30 0.000976
##D   ph.ecog  3.14e-03  3.80e-03 1.03e-03  3.70 0.000214
##D 
##D Chisq=26.73 on 3 df, p=6.7e-06; test weights=aalen
##D 
##D plot(lfit[4], ylim=c(-4,4))  # Draw a plot of the function for ph.ecog
## End(Not run)
lfit2 <- aareg(Surv(time, status) ~ age + sex + ph.ecog, data=lung,
                  nmin=1, taper=1:10)
## Not run: lines(lfit2[4], col=2)  # Nearly the same, until the last point

# A fit to the mulitple-infection data set of children with
# Chronic Granuomatous Disease.  See section 8.5 of Therneau and Grambsch.
fita2 <- aareg(Surv(tstart, tstop, status) ~ treat + age + inherit +
                         steroids + cluster(id), data=cgd)
## Not run: 
##D   n= 203 
##D     69 out of 70 unique event times used
##D 
##D                      slope      coef se(coef) robust se     z        p
##D Intercept         0.004670  0.017800 0.002780  0.003910  4.55 5.30e-06
##D treatrIFN-g      -0.002520 -0.010100 0.002290  0.003020 -3.36 7.87e-04
##D age              -0.000101 -0.000317 0.000115  0.000117 -2.70 6.84e-03
##D inheritautosomal  0.001330  0.003830 0.002800  0.002420  1.58 1.14e-01
##D steroids          0.004620  0.013200 0.010600  0.009700  1.36 1.73e-01
##D 
##D Chisq=16.74 on 4 df, p=0.0022; test weights=aalen
## End(Not run)



cleanEx()
nameEx("aggregate.survfit")
### * aggregate.survfit

flush(stderr()); flush(stdout())

### Name: aggregate.survfit
### Title: Average survival curves
### Aliases: aggregate.survfit
### Keywords: survival

### ** Examples

cfit <- coxph(Surv(futime, death) ~ sex + age*hgb, data=mgus2)
# marginal effect of sex, after adjusting for the others
dummy <- rbind(mgus2, mgus2)
dummy$sex <- rep(c("F", "M"), each=nrow(mgus2)) # population data set
dummy <- na.omit(dummy)   # don't count missing hgb in our "population
csurv <- survfit(cfit, newdata=dummy)
dim(csurv)  # 2 * 1384 survival curves
csurv2 <- aggregate(csurv, dummy$sex)



cleanEx()
nameEx("anova.coxph")
### * anova.coxph

flush(stderr()); flush(stdout())

### Name: anova.coxph
### Title: Analysis of Deviance for a Cox model.
### Aliases: anova.coxph anova.coxphlist
### Keywords: models regression survival

### ** Examples

fit <- coxph(Surv(futime, fustat) ~ resid.ds *rx + ecog.ps, data = ovarian) 
anova(fit)
fit2 <- coxph(Surv(futime, fustat) ~ resid.ds +rx + ecog.ps, data=ovarian)
anova(fit2,fit)
 


cleanEx()
nameEx("attrassign")
### * attrassign

flush(stderr()); flush(stdout())

### Name: attrassign
### Title: Create new-style "assign" attribute
### Aliases: attrassign.default attrassign attrassign.lm
### Keywords: models

### ** Examples

formula <- Surv(time,status)~factor(ph.ecog)
tt <- terms(formula)
mf <- model.frame(tt,data=lung)
mm <- model.matrix(tt,mf)
## a few rows of data
mm[1:3,]
## old-style assign attribute
attr(mm,"assign")
## alternate style assign attribute
attrassign(mm,tt)



cleanEx()
nameEx("blogit")
### * blogit

flush(stderr()); flush(stdout())

### Name: blogit
### Title: Bounded link functions
### Aliases: blogit bcloglog bprobit blog
### Keywords: survival

### ** Examples

py <- pseudo(survfit(Surv(time, status) ~1, lung), time=730) #2 year survival
range(py)
pfit <- glm(py ~ ph.ecog, data=lung, family=gaussian(link=blogit()))
# For each +1 change in performance score, the odds of 2 year survival
#  are multiplied by 1/2  = exp of the coefficient.



cleanEx()
nameEx("brier")
### * brier

flush(stderr()); flush(stdout())

### Name: brier
### Title: Compute the Brier score for a Cox model
### Aliases: brier
### Keywords: survival

### ** Examples

cfit <- coxph(Surv(rtime, recur) ~ age + meno + size + pmin(nodes,11), 
              data= rotterdam)
round(cfit$concordance["concordance"], 3)  # some predictive power
brier(cfit, times=c(4,6)*365.25)   # values at 4 and 6 years



cleanEx()
nameEx("cch")
### * cch

flush(stderr()); flush(stdout())

### Name: cch
### Title: Fits proportional hazards regression model to case-cohort data
### Aliases: cch
### Keywords: survival

### ** Examples

## The complete Wilms Tumor Data 
## (Breslow and Chatterjee, Applied Statistics, 1999)
## subcohort selected by simple random sampling.
##

subcoh <- nwtco$in.subcohort
selccoh <- with(nwtco, rel==1|subcoh==1)
ccoh.data <- nwtco[selccoh,]
ccoh.data$subcohort <- subcoh[selccoh]
## central-lab histology 
ccoh.data$histol <- factor(ccoh.data$histol,labels=c("FH","UH"))
## tumour stage
ccoh.data$stage <- factor(ccoh.data$stage,labels=c("I","II","III","IV"))
ccoh.data$age <- ccoh.data$age/12 # Age in years

##
## Standard case-cohort analysis: simple random subcohort 
##

fit.ccP <- cch(Surv(edrel, rel) ~ stage + histol + age, data =ccoh.data,
   subcoh = ~subcohort, id=~seqno, cohort.size=4028)


fit.ccP

fit.ccSP <- cch(Surv(edrel, rel) ~ stage + histol + age, data =ccoh.data,
   subcoh = ~subcohort, id=~seqno, cohort.size=4028, method="SelfPren")

summary(fit.ccSP)

##
## (post-)stratified on instit
##
stratsizes<-table(nwtco$instit)
fit.BI<- cch(Surv(edrel, rel) ~ stage + histol + age, data =ccoh.data,
   subcoh = ~subcohort, id=~seqno, stratum=~instit, cohort.size=stratsizes,
   method="I.Borgan")

summary(fit.BI)



cleanEx()
nameEx("cipoisson")
### * cipoisson

flush(stderr()); flush(stdout())

### Name: cipoisson
### Title: Confidence limits for the Poisson
### Aliases: cipoisson

### ** Examples

cipoisson(4) # 95% confidence limit 
# lower    upper  
# 1.089865 10.24153 
ppois(4, 10.24153)     #chance of seeing 4 or fewer events with large rate  
# [1] 0.02500096 
1-ppois(3, 1.08986)    #chance of seeing 4 or more, with a small rate 
# [1] 0.02499961




cleanEx()
nameEx("clogit")
### * clogit

flush(stderr()); flush(stdout())

### Name: clogit
### Title: Conditional logistic regression
### Aliases: clogit
### Keywords: survival models

### ** Examples

## Not run: clogit(case ~ spontaneous + induced + strata(stratum), data=infert)

# A multinomial response recoded to use clogit
#  The revised data set has one copy per possible outcome level, with new
#  variable tocc = target occupation for this copy, and case = whether
#  that is the actual outcome for each subject.
# See the reference below for the data.
resp <- levels(logan$occupation)
n <- nrow(logan)
indx <- rep(1:n, length(resp))
logan2 <- data.frame(logan[indx,],
                     id = indx,
                     tocc = factor(rep(resp, each=n)))
logan2$case <- (logan2$occupation == logan2$tocc)
clogit(case ~ tocc + tocc:education + strata(id), logan2)



cleanEx()
nameEx("cluster")
### * cluster

flush(stderr()); flush(stdout())

### Name: cluster
### Title: Identify clusters.
### Aliases: cluster
### Keywords: survival

### ** Examples

marginal.model <- coxph(Surv(time, status) ~ rx, data= rats, cluster=litter,
                         subset=(sex=='f'))
frailty.model  <- coxph(Surv(time, status) ~ rx + frailty(litter), rats,
                         subset=(sex=='f'))



cleanEx()
nameEx("concordance")
### * concordance

flush(stderr()); flush(stdout())

### Name: concordance
### Title: Compute the concordance statistic for data or a model
### Aliases: concordance concordance.coxph concordance.formula
###   concordance.lm concordance.survreg
### Keywords: survival

### ** Examples

fit1 <- coxph(Surv(ptime, pstat) ~ age + sex + mspike, mgus2)
concordance(fit1, timewt="n/G2")  # Uno's weighting

# logistic regression 
fit2 <- glm(I(sex=='M') ~ age + log(creatinine), binomial, data= flchain)
concordance(fit2)  # equal to the AUC

# compare multiple models 
options(na.action = na.exclude)   # predict all 1384 obs, including missing
fit3 <- glm(pstat ~ age + sex + mspike + offset(log(ptime)), 
            poisson, data= mgus2)
fit4 <- coxph(Surv(ptime, pstat) ~ age + sex + mspike, mgus2)
fit5 <- coxph(Surv(ptime, pstat) ~ age + sex + hgb + creat, mgus2)

tdata <- mgus2; tdata$ptime <- 60   # prediction at 60 months
p3 <- -predict(fit3, newdata=tdata) 
p4 <- -predict(fit4) # high risk scores predict shorter survival
p5 <- -predict(fit5)
options(na.action = na.omit)      # return to the R default

cfit <- concordance(Surv(ptime, pstat) ~p3 +  p4 + p5, mgus2)
cfit
round(coef(cfit), 3)
round(cov2cor(vcov(cfit)), 3)  # high correlation

test <- c(1, -1, 0)  # contrast vector for model 1 - model 2 
round(c(difference = test %*% coef(cfit),
        sd= sqrt(test %*% vcov(cfit) %*% test)), 3)



cleanEx()
nameEx("cox.zph")
### * cox.zph

flush(stderr()); flush(stdout())

### Name: cox.zph
### Title: Test the Proportional Hazards Assumption of a Cox Regression
### Aliases: cox.zph [.cox.zph print.cox.zph
### Keywords: survival

### ** Examples

fit <- coxph(Surv(futime, fustat) ~ age + ecog.ps,  
             data=ovarian) 
temp <- cox.zph(fit) 
print(temp)                  # display the results 
plot(temp)                   # plot curves 



cleanEx()
nameEx("coxph")
### * coxph

flush(stderr()); flush(stdout())

### Name: coxph
### Title: Fit Proportional Hazards Regression Model
### Aliases: coxph print.coxph.null print.coxph.penal coxph.penalty
###   coxph.getdata summary.coxph.penal
### Keywords: survival

### ** Examples

# Create the simplest test data set 
test1 <- list(time=c(4,3,1,1,2,2,3), 
              status=c(1,1,1,0,1,1,0), 
              x=c(0,2,1,1,1,0,0), 
              sex=c(0,0,0,0,1,1,1)) 
# Fit a stratified model 
coxph(Surv(time, status) ~ x + strata(sex), test1) 
# Create a simple data set for a time-dependent model 
test2 <- list(start=c(1,2,5,2,1,7,3,4,8,8), 
              stop=c(2,3,6,7,8,9,9,9,14,17), 
              event=c(1,1,1,1,1,1,1,0,0,0), 
              x=c(1,0,0,1,0,1,1,1,0,0)) 
summary(coxph(Surv(start, stop, event) ~ x, test2)) 

#
# Create a simple data set for a time-dependent model
#
test2 <- list(start=c(1, 2, 5, 2, 1, 7, 3, 4, 8, 8),
                stop =c(2, 3, 6, 7, 8, 9, 9, 9,14,17),
                event=c(1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
                x    =c(1, 0, 0, 1, 0, 1, 1, 1, 0, 0) )


summary( coxph( Surv(start, stop, event) ~ x, test2))

# Fit a stratified model, clustered on patients 

bladder1 <- bladder[bladder$enum < 5, ] 
coxph(Surv(stop, event) ~ (rx + size + number) * strata(enum),
      cluster = id, bladder1)

# Fit a time transform model using current age
coxph(Surv(time, status) ~ ph.ecog + tt(age), data=lung,
     tt=function(x,t,...) pspline(x + t/365.25))



cleanEx()
nameEx("coxph.detail")
### * coxph.detail

flush(stderr()); flush(stdout())

### Name: coxph.detail
### Title: Details of a Cox Model Fit
### Aliases: coxph.detail
### Keywords: survival

### ** Examples

fit   <- coxph(Surv(futime,fustat) ~ age + rx + ecog.ps, ovarian, x=TRUE)
fitd  <- coxph.detail(fit)
#  There is one Schoenfeld residual for each unique death.  It is a
# vector (covariates for the subject who died) - (weighted mean covariate
# vector at that time).  The weighted mean is defined over the subjects
# still at risk, with exp(X beta) as the weight.

events <- fit$y[,2]==1
etime  <- fit$y[events,1]   #the event times --- may have duplicates
indx   <- match(etime, fitd$time)
schoen <- fit$x[events,] - fitd$means[indx,]



cleanEx()
nameEx("diabetic")
### * diabetic

flush(stderr()); flush(stdout())

### Name: diabetic
### Title: Ddiabetic retinopathy
### Aliases: diabetic
### Keywords: datasets survival

### ** Examples

# juvenile diabetes is defined as and age less than 20
juvenile <- 1*(diabetic$age < 20)
coxph(Surv(time, status) ~ trt + juvenile, cluster= id,
            data= diabetic)



cleanEx()
nameEx("dsurvreg")
### * dsurvreg

flush(stderr()); flush(stdout())

### Name: dsurvreg
### Title: Distributions available in survreg.
### Aliases: dsurvreg psurvreg qsurvreg rsurvreg
### Keywords: distribution

### ** Examples

# List of distributions available
names(survreg.distributions)
## Not run: 
##D  [1] "extreme"     "logistic"    "gaussian"    "weibull"     "exponential"
##D  [6] "rayleigh"    "loggaussian" "lognormal"   "loglogistic" "t"          
## End(Not run)
# Compare results
all.equal(dsurvreg(1:10, 2, 5, dist='lognormal'), dlnorm(1:10, 2, 5))

# Hazard function for a Weibull distribution
x   <- seq(.1, 3, length=30)
haz <- dsurvreg(x, 2, 3)/ (1-psurvreg(x, 2, 3))
## Not run: 
##D plot(x, haz, log='xy', ylab="Hazard") #line with slope (1/scale -1)
## End(Not run)

# Estimated CDF of a simple Weibull
fit <- survreg(Surv(time, status) ~ 1, data=lung)
pp <- 1:99/100  
q1 <- qsurvreg(pp, coef(fit), fit$scale)
q2 <- qweibull(pp, shape= 1/fit$scale, scale= exp(coef(fit)))
all.equal(q1, q2)
## Not run: 
##D plot(q1, pp, type='l', xlab="Months", ylab="CDF")
## End(Not run)
# per the help page for dweibull, the mean is scale * gamma(1 + 1/shape)
c(mean = exp(coef(fit))* gamma(1 + fit$scale))




cleanEx()
nameEx("finegray")
### * finegray

flush(stderr()); flush(stdout())

### Name: finegray
### Title: Create data for a Fine-Gray model
### Aliases: finegray
### Keywords: survival

### ** Examples

# Treat time to death and plasma cell malignancy as competing risks
etime <- with(mgus2, ifelse(pstat==0, futime, ptime))
event <- with(mgus2, ifelse(pstat==0, 2*death, 1))
event <- factor(event, 0:2, labels=c("censor", "pcm", "death"))

# FG model for PCM
pdata <- finegray(Surv(etime, event) ~ ., data=mgus2)
fgfit <- coxph(Surv(fgstart, fgstop, fgstatus) ~ age + sex,
                     weight=fgwt, data=pdata)

# Compute the weights separately by sex
adata <- finegray(Surv(etime, event) ~ . + strata(sex),
             data=mgus2, na.action=na.pass)



cleanEx()
nameEx("flchain")
### * flchain

flush(stderr()); flush(stdout())

### Name: flchain
### Title: Assay of serum free light chain for 7874 subjects.
### Aliases: flchain
### Keywords: datasets

### ** Examples

data(flchain)
age.grp <-  cut(flchain$age, c(49,54, 59,64, 69,74,79, 89, 110),
               labels= paste(c(50,55,60,65,70,75,80,90),
                             c(54,59,64,69,74,79,89,109), sep='-'))
table(flchain$sex, age.grp)



cleanEx()
nameEx("frailty")
### * frailty

flush(stderr()); flush(stdout())

### Name: frailty
### Title: Random effects terms
### Aliases: frailty frailty.gamma frailty.gaussian frailty.t
### Keywords: survival

### ** Examples

# Random institutional effect
coxph(Surv(time, status) ~ age + frailty(inst, df=4), lung)

# Litter effects for the rats data
rfit2a <- coxph(Surv(time, status) ~ rx +
                  frailty.gaussian(litter, df=13, sparse=FALSE), rats,
                  subset= (sex=='f'))
rfit2b <- coxph(Surv(time, status) ~ rx +
                  frailty.gaussian(litter, df=13, sparse=TRUE), rats,
                  subset= (sex=='f'))



cleanEx()
nameEx("hoel")
### * hoel

flush(stderr()); flush(stdout())

### Name: hoel
### Title: Mouse cancer data
### Aliases: hoel
### Keywords: datasets

### ** Examples

hsurv <- survfit(Surv(days, outcome) ~ trt, data = hoel, id= id)
plot(hsurv, lty=1:2, col=rep(1:3, each=2), lwd=2, xscale=30.5,
      xlab="Months", ylab= "Death")
legend("topleft", c("Lymphoma control", "Lymphoma germ free",
                    "Sarcoma control", "Sarcoma germ free",
                    "Other control", "Other germ free"),
       col=rep(1:3, each=2), lty=1:2, lwd=2, bty='n')
hfit <- coxph(Surv(days, outcome) ~ trt, data= hoel, id = id)



cleanEx()
nameEx("is.ratetable")
### * is.ratetable

flush(stderr()); flush(stdout())

### Name: is.ratetable
### Title: Verify that an object is of class ratetable.
### Aliases: is.ratetable Math.ratetable Ops.ratetable
### Keywords: survival

### ** Examples

is.ratetable(survexp.us)  # True
is.ratetable(lung)        # False



cleanEx()
nameEx("kidney")
### * kidney

flush(stderr()); flush(stdout())

### Name: kidney
### Title: Kidney catheter data
### Aliases: kidney
### Keywords: survival

### ** Examples

kfit <- coxph(Surv(time, status)~ age + sex + disease + frailty(id), kidney)
kfit0 <- coxph(Surv(time, status)~ age + sex + disease, kidney)
kfitm1 <- coxph(Surv(time,status) ~ age + sex + disease + 
		frailty(id, dist='gauss'), kidney)



cleanEx()
nameEx("levels.Surv")
### * levels.Surv

flush(stderr()); flush(stdout())

### Name: levels.Surv
### Title: Return the states of a multi-state Surv object
### Aliases: levels.Surv
### Keywords: survival

### ** Examples

y1 <- Surv(c(1,5, 9, 17,21, 30),
           factor(c(0, 1, 2,1,0,2), 0:2, c("censored", "progression", "death")))
levels(y1)

y2 <- Surv(1:6, rep(0:1, 3))
y2
levels(y2)



cleanEx()
nameEx("lines.survfit")
### * lines.survfit

flush(stderr()); flush(stdout())

### Name: lines.survfit
### Title: Add Lines or Points to a Survival Plot
### Aliases: lines.survfit points.survfit lines.survexp
### Keywords: survival

### ** Examples

fit <- survfit(Surv(time, status==2) ~ sex, pbc,subset=1:312)
plot(fit, mark.time=FALSE, xscale=365.25,
        xlab='Years', ylab='Survival')
lines(fit[1], lwd=2)    #darken the first curve and add marks


# Add expected survival curves for the two groups,
#   based on the US census data
# The data set does not have entry date, use the midpoint of the study
efit <- survexp(~sex, data=pbc, times= (0:24)*182, ratetable=survexp.us, 
                 rmap=list(sex=sex, age=age*365.35, year=as.Date('1979/01/01')))
temp <- lines(efit, lty=2, lwd=2:1)
text(temp, c("Male", "Female"), adj= -.1) #labels just past the ends
title(main="Primary Biliary Cirrhosis, Observed and Expected")




cleanEx()
nameEx("mgus")
### * mgus

flush(stderr()); flush(stdout())

### Name: mgus
### Title: Monoclonal gammopathy data
### Aliases: mgus mgus1
### Keywords: datasets survival

### ** Examples

# Create the competing risk curves for time to first of death or PCM
sfit <- survfit(Surv(start, stop, event) ~ sex, mgus1, id=id,
                subset=(enum==1))
print(sfit)  # the order of printout is the order in which they plot

plot(sfit, xscale=365.25, lty=c(2,2,1,1), col=c(1,2,1,2),
     xlab="Years after MGUS detection", ylab="Proportion")
legend(0, .8, c("Death/male", "Death/female", "PCM/male", "PCM/female"),
       lty=c(1,1,2,2), col=c(2,1,2,1), bty='n')

title("Curves for the first of plasma cell malignancy or death")
# The plot shows that males have a higher death rate than females (no
# surprise) but their rates of conversion to PCM are essentially the same.



cleanEx()
nameEx("model.matrix.coxph")
### * model.matrix.coxph

flush(stderr()); flush(stdout())

### Name: model.matrix.coxph
### Title: Model.matrix method for coxph models
### Aliases: model.matrix.coxph
### Keywords: survival

### ** Examples

fit1 <- coxph(Surv(time, status) ~ age + factor(ph.ecog), data=lung)
xfit <- model.matrix(fit1)

fit2 <- coxph(Surv(time, status) ~ age + factor(ph.ecog), data=lung,
                                 x=TRUE)
all.equal(model.matrix(fit1), fit2$x)



cleanEx()
nameEx("myeloid")
### * myeloid

flush(stderr()); flush(stdout())

### Name: myeloid
### Title: Acute myeloid leukemia
### Aliases: myeloid
### Keywords: datasets

### ** Examples

coxph(Surv(futime, death) ~ trt + flt3, data=myeloid)
# See the mstate vignette for a more complete analysis



cleanEx()
nameEx("myeloma")
### * myeloma

flush(stderr()); flush(stdout())

### Name: myeloma
### Title: Survival times of patients with multiple myeloma
### Aliases: myeloma
### Keywords: datasets

### ** Examples

# Incorrect survival curve, which ignores left truncation
fit1 <- survfit(Surv(futime, death) ~ 1, myeloma)
# Correct curve
fit2 <- survfit(Surv(entry, futime, death) ~1, myeloma)



cleanEx()
nameEx("neardate")
### * neardate

flush(stderr()); flush(stdout())

### Name: neardate
### Title: Find the index of the closest value in data set 2, for each
###   entry in data set one.
### Aliases: neardate
### Keywords: manip utilities

### ** Examples

data1 <- data.frame(id = 1:10,
                    entry.dt = as.Date(paste("2011", 1:10, "5", sep='-')))
temp1 <- c(1,4,5,1,3,6,9, 2,7,8,12,4,6,7,10,12,3)
data2 <- data.frame(id = c(1,1,1,2,2,4,4,5,5,5,6,8,8,9,10,10,12),
                    lab.dt = as.Date(paste("2011", temp1, "1", sep='-')),
                    chol = round(runif(17, 130, 280)))

#first cholesterol on or after enrollment
indx1 <- neardate(data1$id, data2$id, data1$entry.dt, data2$lab.dt)
data2[indx1, "chol"]

# Closest one, either before or after. 
# 
indx2 <- neardate(data1$id, data2$id, data1$entry.dt, data2$lab.dt, 
                   best="prior")
ifelse(is.na(indx1), indx2, # none after, take before
       ifelse(is.na(indx2), indx1, #none before
       ifelse(abs(data2$lab.dt[indx2]- data1$entry.dt) <
              abs(data2$lab.dt[indx1]- data1$entry.dt), indx2, indx1)))

# closest date before or after, but no more than 21 days prior to index
indx2 <- ifelse((data1$entry.dt - data2$lab.dt[indx2]) >21, NA, indx2)
ifelse(is.na(indx1), indx2, # none after, take before
       ifelse(is.na(indx2), indx1, #none before
       ifelse(abs(data2$lab.dt[indx2]- data1$entry.dt) <
              abs(data2$lab.dt[indx1]- data1$entry.dt), indx2, indx1)))



cleanEx()
nameEx("nsk")
### * nsk

flush(stderr()); flush(stdout())

### Name: nsk
### Title: Natural splines with knot heights as the basis.
### Aliases: nsk
### Keywords: smooth

### ** Examples

# make some dummy data
tdata <- data.frame(x= lung$age, y = 10*log(lung$age-35) + rnorm(228, 0, 2))
fit1 <- lm(y ~ -1 + nsk(x, df=4, intercept=TRUE) , data=tdata)
fit2 <- lm(y ~ nsk(x, df=3), data=tdata)

# the knots (same for both fits)
knots <- unlist(attributes(fit1$model[[2]])[c('Boundary.knots', 'knots')])
sort(unname(knots))
unname(coef(fit1))  # predictions at the knot points

unname(coef(fit1)[-1] - coef(fit1)[1])  # differences: yhat[2:4] - yhat[1]
unname(coef(fit2))[-1]                  # ditto

## Not run: 
##D plot(y ~ x, data=tdata)
##D points(sort(knots), coef(fit1), col=2, pch=19)
##D coef(fit)[1] + c(0, coef(fit)[-1])
## End(Not run)



cleanEx()
nameEx("nwtco")
### * nwtco

flush(stderr()); flush(stdout())

### Name: nwtco
### Title: Data from the National Wilm's Tumor Study
### Aliases: nwtco
### Keywords: datasets

### ** Examples

with(nwtco, table(instit,histol))
anova(coxph(Surv(edrel,rel)~histol+instit,data=nwtco))
anova(coxph(Surv(edrel,rel)~instit+histol,data=nwtco))



cleanEx()
nameEx("pbcseq")
### * pbcseq

flush(stderr()); flush(stdout())

### Name: pbcseq
### Title: Mayo Clinic Primary Biliary Cirrhosis, sequential data
### Aliases: pbcseq
### Keywords: datasets

### ** Examples

# Create the start-stop-event triplet needed for coxph
first <- with(pbcseq, c(TRUE, diff(id) !=0)) #first id for each subject
last  <- c(first[-1], TRUE)  #last id

time1 <- with(pbcseq, ifelse(first, 0, day))
time2 <- with(pbcseq, ifelse(last,  futime, c(day[-1], 0)))
event <- with(pbcseq, ifelse(last,  status, 0))

fit1 <- coxph(Surv(time1, time2, event) ~ age + sex + log(bili), pbcseq)



cleanEx()
nameEx("plot.cox.zph")
### * plot.cox.zph

flush(stderr()); flush(stdout())

### Name: plot.cox.zph
### Title: Graphical Test of Proportional Hazards
### Aliases: plot.cox.zph
### Keywords: survival

### ** Examples

vfit <- coxph(Surv(time,status) ~ trt + factor(celltype) + 
              karno + age, data=veteran, x=TRUE) 
temp <- cox.zph(vfit) 
plot(temp, var=3)      # Look at Karnofsy score, old way of doing plot 
plot(temp[3])     # New way with subscripting 
abline(0, 0, lty=3) 
# Add the linear fit as well  
abline(lm(temp$y[,3] ~ temp$x)$coefficients, lty=4, col=3)  
title(main="VA Lung Study") 



cleanEx()
nameEx("plot.survfit")
### * plot.survfit

flush(stderr()); flush(stdout())

### Name: plot.survfit
### Title: Plot method for 'survfit' objects
### Aliases: plot.survfit
### Keywords: survival hplot

### ** Examples

leukemia.surv <- survfit(Surv(time, status) ~ x, data = aml) 
plot(leukemia.surv, lty = 2:3) 
legend(100, .9, c("Maintenance", "No Maintenance"), lty = 2:3) 
title("Kaplan-Meier Curves\nfor AML Maintenance Study") 
lsurv2 <- survfit(Surv(time, status) ~ x, aml, type='fleming') 
plot(lsurv2, lty=2:3, fun="cumhaz", 
	xlab="Months", ylab="Cumulative Hazard") 



cleanEx()
nameEx("predict.coxph")
### * predict.coxph

flush(stderr()); flush(stdout())

### Name: predict.coxph
### Title: Predictions for a Cox model
### Aliases: predict.coxph predict.coxph.penal
### Keywords: survival

### ** Examples

options(na.action=na.exclude) # retain NA in predictions
fit <- coxph(Surv(time, status) ~ age + ph.ecog + strata(inst), lung)
#lung data set has status coded as 1/2
mresid <- (lung$status-1) - predict(fit, type='expected') #Martingale resid 
predict(fit,type="lp")
predict(fit,type="expected")
predict(fit,type="risk",se.fit=TRUE)
predict(fit,type="terms",se.fit=TRUE)

# For someone who demands reference='zero'
pzero <- function(fit)
  predict(fit, reference="sample") + sum(coef(fit) * fit$means, na.rm=TRUE)



cleanEx()
nameEx("predict.survreg")
### * predict.survreg

flush(stderr()); flush(stdout())

### Name: predict.survreg
### Title: Predicted Values for a 'survreg' Object
### Aliases: predict.survreg predict.survreg.penal
### Keywords: survival

### ** Examples

# Draw figure 1 from Escobar and Meeker, 1992.
fit <- survreg(Surv(time,status) ~ age + I(age^2), data=stanford2, 
	dist='lognormal') 
with(stanford2, plot(age, time, xlab='Age', ylab='Days', 
	xlim=c(0,65), ylim=c(.1, 10^5), log='y', type='n'))
with(stanford2, points(age, time, pch=c(2,4)[status+1], cex=.7))
pred <- predict(fit, newdata=list(age=1:65), type='quantile', 
	         p=c(.1, .5, .9)) 
matlines(1:65, pred, lty=c(2,1,2), col=1) 

# Predicted Weibull survival curve for a lung cancer subject with
#  ECOG score of 2
lfit <- survreg(Surv(time, status) ~ ph.ecog, data=lung)
pct <- 1:98/100   # The 100th percentile of predicted survival is at +infinity
ptime <- predict(lfit, newdata=data.frame(ph.ecog=2), type='quantile',
                 p=pct, se=TRUE)
matplot(cbind(ptime$fit, ptime$fit + 2*ptime$se.fit,
                         ptime$fit - 2*ptime$se.fit)/30.5, 1-pct,
        xlab="Months", ylab="Survival", type='l', lty=c(1,2,2), col=1)



cleanEx()
nameEx("pseudo")
### * pseudo

flush(stderr()); flush(stdout())

### Name: pseudo
### Title: Pseudo values for survival.
### Aliases: pseudo
### Keywords: survival

### ** Examples

fit1 <- survfit(Surv(time, status) ~ 1, data=lung)
yhat <- pseudo(fit1, times=c(365, 730))
dim(yhat)
lfit <- lm(yhat[,1] ~ ph.ecog + age + sex, data=lung)

# Restricted Mean Time in State (RMST) 
rms <- pseudo(fit1, times= 730, type='RMST') # 2 years
rfit <- lm(rms ~ ph.ecog + sex, data=lung)
rhat <- predict(rfit, newdata=expand.grid(ph.ecog=0:3, sex=1:2), se.fit=TRUE)
# print it out nicely
temp1 <- cbind(matrix(rhat$fit, 4,2))
temp2 <- cbind(matrix(rhat$se.fit, 4, 2))
temp3 <- cbind(temp1[,1], temp2[,1], temp1[,2], temp2[,2])
dimnames(temp3) <- list(paste("ph.ecog", 0:3), 
                        c("Male RMST", "(se)", "Female RMST", "(se)"))

round(temp3, 1)
# compare this to the fully non-parametric estimate
fit2 <- survfit(Surv(time, status) ~ ph.ecog, data=lung)
print(fit2, rmean=730)
# the estimate for ph.ecog=3 is very unstable (n=1), pseudovalues smooth it.
#
# In all the above we should be using the robust variance, e.g., svyglm, but
#  a recommended package can't depend on external libraries.
# See the vignette for a more complete exposition.



cleanEx()
nameEx("pspline")
### * pspline

flush(stderr()); flush(stdout())

### Name: pspline
### Title: Smoothing splines using a pspline basis
### Aliases: pspline psplineinverse
### Keywords: survival

### ** Examples

lfit6 <- survreg(Surv(time, status)~pspline(age, df=2), lung)
plot(lung$age, predict(lfit6), xlab='Age', ylab="Spline prediction")
title("Cancer Data")
fit0 <- coxph(Surv(time, status) ~ ph.ecog + age, lung)
fit1 <- coxph(Surv(time, status) ~ ph.ecog + pspline(age,3), lung)
fit3 <- coxph(Surv(time, status) ~ ph.ecog + pspline(age,8), lung)
fit0
fit1
fit3



cleanEx()
nameEx("pyears")
### * pyears

flush(stderr()); flush(stdout())

### Name: pyears
### Title: Person Years
### Aliases: pyears
### Keywords: survival

### ** Examples

# Look at progression rates jointly by calendar date and age
# 
temp.yr  <- tcut(mgus$dxyr, 55:92, labels=as.character(55:91)) 
temp.age <- tcut(mgus$age, 34:101, labels=as.character(34:100))
ptime <- ifelse(is.na(mgus$pctime), mgus$futime, mgus$pctime)
pstat <- ifelse(is.na(mgus$pctime), 0, 1)
pfit <- pyears(Surv(ptime/365.25, pstat) ~ temp.yr + temp.age + sex,  mgus,
     data.frame=TRUE) 
# Turn the factor back into numerics for regression
tdata <- pfit$data
tdata$age <- as.numeric(as.character(tdata$temp.age))
tdata$year<- as.numeric(as.character(tdata$temp.yr))
fit1 <- glm(event ~ year + age+ sex +offset(log(pyears)),
             data=tdata, family=poisson)
## Not run: 
##D # fit a gam model 
##D gfit.m <- gam(y ~ s(age) + s(year) + offset(log(time)),  
##D                         family = poisson, data = tdata) 
## End(Not run)

# Example #2  Create the hearta data frame: 
hearta <- by(heart, heart$id,  
             function(x)x[x$stop == max(x$stop),]) 
hearta <- do.call("rbind", hearta) 
# Produce pyears table of death rates on the surgical arm
#  The first is by age at randomization, the second by current age
fit1 <- pyears(Surv(stop/365.25, event) ~ cut(age + 48, c(0,50,60,70,100)) + 
       surgery, data = hearta, scale = 1)
fit2 <- pyears(Surv(stop/365.25, event) ~ tcut(age + 48, c(0,50,60,70,100)) + 
       surgery, data = hearta, scale = 1)
fit1$event/fit1$pyears  #death rates on the surgery and non-surg arm

fit2$event/fit2$pyears  #death rates on the surgery and non-surg arm



cleanEx()
nameEx("quantile.survfit")
### * quantile.survfit

flush(stderr()); flush(stdout())

### Name: quantile.survfit
### Title: Quantiles from a survfit object
### Aliases: quantile.survfit quantile.survfitms median.survfit
### Keywords: survival

### ** Examples

fit <- survfit(Surv(time, status) ~ ph.ecog, data=lung)
quantile(fit)

cfit <- coxph(Surv(time, status) ~ age + strata(ph.ecog), data=lung)
csurv<- survfit(cfit, newdata=data.frame(age=c(40, 60, 80)),
                  conf.type ="none")
temp <- quantile(csurv, 1:5/10)
temp[2,3,]  # quantiles for second level of ph.ecog, age=80
quantile(csurv[2,3], 1:5/10)  # quantiles of a single curve, same result



cleanEx()
nameEx("reliability")
### * reliability

flush(stderr()); flush(stdout())

### Name: reliability
### Title: Reliability data sets
### Aliases: reliability capacitor cracks genfan ifluid imotor turbine
###   valveSeat
### Keywords: datasets

### ** Examples

survreg(Surv(time, status) ~ temperature + voltage, capacitor)



cleanEx()
nameEx("residuals.coxph")
### * residuals.coxph

flush(stderr()); flush(stdout())

### Name: residuals.coxph
### Title: Calculate Residuals for a 'coxph' Fit
### Aliases: residuals.coxph.penal residuals.coxph.null residuals.coxph
###   residuals.coxphms
### Keywords: survival

### ** Examples


 fit <- coxph(Surv(start, stop, event) ~ (age + surgery)* transplant,
               data=heart)
 mresid <- resid(fit, collapse=heart$id)



cleanEx()
nameEx("residuals.survfit")
### * residuals.survfit

flush(stderr()); flush(stdout())

### Name: residuals.survfit
### Title: IJ residuals from a survfit object.
### Aliases: residuals.survfit

### ** Examples

fit <- survfit(Surv(time, status) ~ x, aml)
resid(fit, times=c(24, 48), type="RMTS")



cleanEx()
nameEx("residuals.survreg")
### * residuals.survreg

flush(stderr()); flush(stdout())

### Name: residuals.survreg
### Title: Compute Residuals for 'survreg' Objects
### Aliases: residuals.survreg residuals.survreg.penal
### Keywords: survival

### ** Examples

fit <- survreg(Surv(futime, death) ~ age + sex, mgus2)
summary(fit)   # age and sex are both important

rr  <- residuals(fit, type='matrix')
sum(rr[,1]) - with(mgus2, sum(log(futime[death==1]))) # loglik

plot(mgus2$age, rr[,2], col= (1+mgus2$death)) # ldresp



cleanEx()
nameEx("retinopathy")
### * retinopathy

flush(stderr()); flush(stdout())

### Name: retinopathy
### Title: Diabetic Retinopathy
### Aliases: retinopathy
### Keywords: datasets

### ** Examples

coxph(Surv(futime, status) ~ type + trt, cluster= id, retinopathy)



cleanEx()
nameEx("rhDNase")
### * rhDNase

flush(stderr()); flush(stdout())

### Name: rhDNase
### Title: rhDNASE data set
### Aliases: rhDNase
### Keywords: datasets

### ** Examples

# Build the start-stop data set for analysis, and
#  replicate line 2 of table 8.13 in the book
first <- subset(rhDNase, !duplicated(id)) #first row for each subject
dnase <- tmerge(first, first, id=id, tstop=as.numeric(end.dt -entry.dt))

# Subjects whose fu ended during the 6 day window are the reason for
#  this next line
temp.end <- with(rhDNase, pmin(ivstop+6, end.dt-entry.dt))
dnase <- tmerge(dnase, rhDNase, id=id,
                       infect=event(ivstart),
                       end=  event(temp.end))
# toss out the non-at-risk intervals, and extra variables
#  3 subjects had an event on their last day of fu, infect=1 and end=1
dnase <- subset(dnase, (infect==1 | end==0), c(id:trt, fev:infect))
agfit <- coxph(Surv(tstart, tstop, infect) ~ trt + fev, cluster=id,
                 data=dnase)



cleanEx()
nameEx("ridge")
### * ridge

flush(stderr()); flush(stdout())

### Name: ridge
### Title: Ridge regression
### Aliases: ridge
### Keywords: survival

### ** Examples


coxph(Surv(futime, fustat) ~ rx + ridge(age, ecog.ps, theta=1),
	      ovarian)

lfit0 <- survreg(Surv(time, status) ~1, lung)
lfit1 <- survreg(Surv(time, status) ~ age + ridge(ph.ecog, theta=5), lung)
lfit2 <- survreg(Surv(time, status) ~ sex + ridge(age, ph.ecog, theta=1), lung)
lfit3 <- survreg(Surv(time, status) ~ sex + age + ph.ecog, lung)




cleanEx()
nameEx("rotterdam")
### * rotterdam

flush(stderr()); flush(stdout())

### Name: rotterdam
### Title: Breast cancer data set used in Royston and Altman (2013)
### Aliases: rotterdam
### Keywords: datasets survival

### ** Examples

# liberal definition of rfs (count later deaths)
rfs  <- pmax(rotterdam$recur, rotterdam$death)
rfstime <- with(rotterdam, ifelse(recur==1, rtime, dtime))
fit1 <- coxph(Surv(rfstime, rfs) ~ pspline(age) + meno + size + 
        pspline(nodes) + er,  data = rotterdam)

# conservative (no deaths after last fu for recurrence)
ignore <- with(rotterdam, recur ==0 & death==1 & rtime < dtime)
table(ignore)
rfs2 <- with(rotterdam, ifelse(recur==1 | ignore, recur, death))
rfstime2 <- with(rotterdam, ifelse(recur==1 | ignore, rtime, dtime))
fit2 <- coxph(Surv(rfstime2, rfs2) ~ pspline(age) + meno + size + 
        pspline(nodes) + er,  data = rotterdam)

# Note: Both age and nodes show non-linear effects.
# Royston and Altman used fractional polynomials for the nonlinear terms



cleanEx()
nameEx("royston")
### * royston

flush(stderr()); flush(stdout())

### Name: royston
### Title: Compute Royston's D for a Cox model
### Aliases: royston
### Keywords: survival

### ** Examples

# An example used in Royston and Sauerbrei
pbc2 <- na.omit(pbc)  # no missing values
cfit <- coxph(Surv(time, status==2) ~ age + log(bili) + edema + albumin +
                   stage + copper, data=pbc2, ties="breslow")
royston(cfit)



cleanEx()
nameEx("rttright")
### * rttright

flush(stderr()); flush(stdout())

### Name: rttright
### Title: Compute redistribute-to-the-right weights
### Aliases: rttright
### Keywords: survival

### ** Examples

afit <- survfit(Surv(time, status) ~1, data=aml)
rwt  <- rttright(Surv(time, status) ~1, data=aml)

# Reproduce a Kaplan-Meier
index <- order(aml$time)
cdf <- cumsum(rwt[index])  # weighted CDF
cdf <- cdf[!duplicated(aml$time[index], fromLast=TRUE)]  # remove duplicate times
cbind(time=afit$time, KM= afit$surv, RTTR= 1-cdf)

# Hormonal patients have a diffent censoring pattern
wt2 <- rttright(Surv(dtime, death) ~ hormon, rotterdam, times= 365*c(3, 5))
dim(wt2)



cleanEx()
nameEx("solder")
### * solder

flush(stderr()); flush(stdout())

### Name: solder
### Title: Data from a soldering experiment
### Aliases: solder
### Keywords: datasets

### ** Examples

# The balanced subset used by Chambers and Hastie
#   contains the first 180 of each mask and deletes mask A6. 
index <- 1 + (1:nrow(solder)) - match(solder$Mask, solder$Mask)
solder.balance <- droplevels(subset(solder, Mask != "A6" & index <= 180))



cleanEx()
nameEx("statefig")
### * statefig

flush(stderr()); flush(stdout())

### Name: statefig
### Title: Draw a state space figure.
### Aliases: statefig
### Keywords: survival hplot

### ** Examples

# Draw a simple competing risks figure
states <- c("Entry", "Complete response", "Relapse", "Death")
connect <- matrix(0, 4, 4, dimnames=list(states, states))
connect[1, -1] <- c(1.1, 1, 0.9)
statefig(c(1, 3), connect)



cleanEx()
nameEx("strata")
### * strata

flush(stderr()); flush(stdout())

### Name: strata
### Title: Identify Stratification Variables
### Aliases: strata
### Keywords: survival

### ** Examples

a <- factor(rep(1:3,4), labels=c("low", "medium", "high"))
b <- factor(rep(1:4,3))
levels(strata(b))
levels(strata(a,b,shortlabel=TRUE))

coxph(Surv(futime, fustat) ~ age + strata(rx), data=ovarian) 



cleanEx()
nameEx("summary.aareg")
### * summary.aareg

flush(stderr()); flush(stdout())

### Name: summary.aareg
### Title: Summarize an aareg fit
### Aliases: summary.aareg
### Keywords: survival

### ** Examples

afit <- aareg(Surv(time, status) ~ age + sex + ph.ecog, data=lung,
     dfbeta=TRUE)
summary(afit)
## Not run: 
##D               slope   test se(test) robust se     z        p 
##D Intercept  5.05e-03    1.9     1.54      1.55  1.23 0.219000
##D       age  4.01e-05  108.0   109.00    106.00  1.02 0.307000
##D       sex -3.16e-03  -19.5     5.90      5.95 -3.28 0.001030
##D   ph.ecog  3.01e-03   33.2     9.18      9.17  3.62 0.000299
##D 
##D Chisq=22.84 on 3 df, p=4.4e-05; test weights=aalen
## End(Not run)

summary(afit, maxtime=600)
## Not run: 
##D               slope   test se(test) robust se      z        p 
##D Intercept  4.16e-03   2.13     1.48      1.47  1.450 0.146000
##D       age  2.82e-05  85.80   106.00    100.00  0.857 0.392000
##D       sex -2.54e-03 -20.60     5.61      5.63 -3.660 0.000256
##D   ph.ecog  2.47e-03  31.60     8.91      8.67  3.640 0.000271
##D 
##D Chisq=27.08 on 3 df, p=5.7e-06; test weights=aalen
## End(Not run)


cleanEx()
nameEx("summary.coxph")
### * summary.coxph

flush(stderr()); flush(stdout())

### Name: summary.coxph
### Title: Summary method for Cox models
### Aliases: summary.coxph
### Keywords: survival

### ** Examples

fit <- coxph(Surv(time, status) ~ age + sex, lung) 
summary(fit)



cleanEx()
nameEx("summary.survfit")
### * summary.survfit

flush(stderr()); flush(stdout())

### Name: summary.survfit
### Title: Summary of a Survival Curve
### Aliases: summary.survfit
### Keywords: survival

### ** Examples

summary( survfit( Surv(futime, fustat)~1, data=ovarian))
summary( survfit( Surv(futime, fustat)~rx, data=ovarian))



cleanEx()
nameEx("survSplit")
### * survSplit

flush(stderr()); flush(stdout())

### Name: survSplit
### Title: Split a survival data set at specified times
### Aliases: survSplit
### Keywords: survival utilities

### ** Examples

fit1 <- coxph(Surv(time, status) ~ karno + age + trt, veteran)
plot(cox.zph(fit1)[1])
# a cox.zph plot of the data suggests that the effect of Karnofsky score
#  begins to diminish by 60 days and has faded away by 120 days.
# Fit a model with separate coefficients for the three intervals.
#
vet2 <- survSplit(Surv(time, status) ~., veteran,
                   cut=c(60, 120), episode ="timegroup")
fit2 <- coxph(Surv(tstart, time, status) ~ karno* strata(timegroup) +
                age + trt, data= vet2)
c(overall= coef(fit1)[1],
  t0_60  = coef(fit2)[1],
  t60_120= sum(coef(fit2)[c(1,4)]),
  t120   = sum(coef(fit2)[c(1,5)]))

# Sometimes we want to split on one scale and analyse on another
#  Add a "current age" variable to the mgus2 data set.
temp1 <- mgus2
temp1$endage <- mgus2$age + mgus2$futime/12    # futime is in months
temp1$startage <- temp1$age
temp2 <- survSplit(Surv(age, endage, death) ~ ., temp1, cut=25:100,
                   start= "age1", end= "age2")

# restore the time since enrollment scale
temp2$time1 <- (temp2$age1 - temp2$startage)*12
temp2$time2 <- (temp2$age2 - temp2$startage)*12

# In this data set, initial age and current age have similar utility
mfit1 <- coxph(Surv(futime, death) ~ age + sex, data=mgus2)
mfit2 <- coxph(Surv(time1, time2, death) ~ age1 + sex, data=temp2)



cleanEx()
nameEx("survcondense")
### * survcondense

flush(stderr()); flush(stdout())

### Name: survcondense
### Title: Shorten a (time1, time2) survival dataset
### Aliases: survcondense
### Keywords: survival

### ** Examples

dim(aml)
test1 <- survSplit(Surv(time, status) ~ ., data=aml, 
                   cut=c(10, 20, 30), id="newid")
dim(test1)

# remove the added rows
test2 <- survcondense(Surv(tstart, time, status) ~ x, test1, id=newid)
dim(test2)



cleanEx()
nameEx("survdiff")
### * survdiff

flush(stderr()); flush(stdout())

### Name: survdiff
### Title: Test Survival Curve Differences
### Aliases: survdiff print.survdiff
### Keywords: survival

### ** Examples

## Two-sample test
survdiff(Surv(futime, fustat) ~ rx,data=ovarian)

## Stratified 7-sample test

survdiff(Surv(time, status) ~ pat.karno + strata(inst), data=lung)

## Expected survival for heart transplant patients based on
## US mortality tables
expect <- survexp(futime ~ 1, data=jasa, cohort=FALSE,
                  rmap= list(age=(accept.dt - birth.dt), sex=1, year=accept.dt),
                  ratetable=survexp.us)
## actual survival is much worse (no surprise)
survdiff(Surv(jasa$futime, jasa$fustat) ~ offset(expect))

# The free light chain data set is close to the population.
e2 <- survexp(futime ~ 1, data=flchain, cohort=FALSE,
              rmap= list(age= age*365.25, sex=sex, 
                         year=as.Date(paste0(sample.yr, "-07-01"))),
              ratetable= survexp.mn)
survdiff(Surv(futime, death) ~ offset(e2), flchain)



cleanEx()
nameEx("survexp")
### * survexp

flush(stderr()); flush(stdout())

### Name: survexp
### Title: Compute Expected Survival
### Aliases: survexp print.survexp
### Keywords: survival

### ** Examples

# 
# Stanford heart transplant data
#  We don't have sex in the data set, but know it to be nearly all males.
# Estimate of conditional survival  
fit1 <- survexp(futime ~ 1, rmap=list(sex="male", year=accept.dt,   
          age=(accept.dt-birth.dt)), method='conditional', data=jasa)
summary(fit1, times=1:10*182.5, scale=365) #expected survival by 1/2 years

# Estimate of expected  survival stratified by prior surgery 
survexp(~ surgery, rmap= list(sex="male", year=accept.dt,  
	age=(accept.dt-birth.dt)), method='ederer', data=jasa,
        times=1:10 * 182.5) 

## Compare the survival curves for the Mayo PBC data to Cox model fit
## 
pfit <-coxph(Surv(time,status>0) ~ trt + log(bili) + log(protime) + age +
                platelet, data=pbc)
plot(survfit(Surv(time, status>0) ~ trt, data=pbc), mark.time=FALSE)
lines(survexp( ~ trt, ratetable=pfit, data=pbc), col='purple')



cleanEx()
nameEx("survexp.us")
### * survexp.us

flush(stderr()); flush(stdout())

### Name: ratetables
### Title: Census Data Sets for the Expected Survival and Person Years
###   Functions
### Aliases: survexp.us survexp.usr survexp.mn
### Keywords: survival datasets

### ** Examples

survexp.uswhite <- survexp.usr[,,"white",]



cleanEx()
nameEx("survfit.formula")
### * survfit.formula

flush(stderr()); flush(stdout())

### Name: survfit.formula
### Title: Compute a Survival Curve for Censored Data
### Aliases: survfit.formula [.survfit
### Keywords: survival

### ** Examples

#fit a Kaplan-Meier and plot it 
fit <- survfit(Surv(time, status) ~ x, data = aml) 
plot(fit, lty = 2:3) 
legend(100, .8, c("Maintained", "Nonmaintained"), lty = 2:3) 

#fit a Cox proportional hazards model and plot the  
#predicted survival for a 60 year old 
fit <- coxph(Surv(futime, fustat) ~ age, data = ovarian) 
plot(survfit(fit, newdata=data.frame(age=60)),
     xscale=365.25, xlab = "Years", ylab="Survival") 

# Here is the data set from Turnbull
#  There are no interval censored subjects, only left-censored (status=3),
#  right-censored (status 0) and observed events (status 1)
#
#                             Time
#                         1    2   3   4
# Type of observation
#           death        12    6   2   3
#          losses         3    2   0   3
#      late entry         2    4   2   5
#
tdata <- data.frame(time  =c(1,1,1,2,2,2,3,3,3,4,4,4),
                    status=rep(c(1,0,2),4),
                    n     =c(12,3,2,6,2,4,2,0,2,3,3,5))
fit  <- survfit(Surv(time, time, status, type='interval') ~1, 
              data=tdata, weight=n)

#
# Three curves for patients with monoclonal gammopathy.
#  1. KM of time to PCM, ignoring death (statistically incorrect)
#  2. Competing risk curves (also known as "cumulative incidence")
#  3. Multi-state, showing Pr(in each state, at time t)
#
fitKM <- survfit(Surv(stop, event=='pcm') ~1, data=mgus1,
                    subset=(start==0))
fitCR <- survfit(Surv(stop, event) ~1,
                    data=mgus1, subset=(start==0))
fitMS <- survfit(Surv(start, stop, event) ~ 1, id=id, data=mgus1)
## Not run: 
##D # CR curves show the competing risks
##D plot(fitCR, xscale=365.25, xmax=7300, mark.time=FALSE,
##D             col=2:3, xlab="Years post diagnosis of MGUS",
##D             ylab="P(state)")
##D lines(fitKM, fun='event', xmax=7300, mark.time=FALSE,
##D             conf.int=FALSE)
##D text(3652, .4, "Competing risk: death", col=3)
##D text(5840, .15,"Competing risk: progression", col=2)
##D text(5480, .30,"KM:prog")
## End(Not run)



cleanEx()
nameEx("survfit.matrix")
### * survfit.matrix

flush(stderr()); flush(stdout())

### Name: survfit.matrix
### Title: Create Aalen-Johansen estimates of multi-state survival from a
###   matrix of hazards.
### Aliases: survfit.matrix
### Keywords: survival

### ** Examples

etime <- with(mgus2, ifelse(pstat==0, futime, ptime))
event <- with(mgus2, ifelse(pstat==0, 2*death, 1))
event <- factor(event, 0:2, labels=c("censor", "pcm", "death"))

cfit1 <- coxph(Surv(etime, event=="pcm") ~ age + sex, mgus2)
cfit2 <- coxph(Surv(etime, event=="death") ~ age + sex, mgus2)

# predicted competing risk curves for a 72 year old with mspike of 1.2
# (median values), male and female.
# The survfit call is a bit faster without standard errors.
newdata <- expand.grid(sex=c("F", "M"), age=72, mspike=1.2)

AJmat <- matrix(list(), 3,3)
AJmat[1,2] <- list(survfit(cfit1, newdata, std.err=FALSE))
AJmat[1,3] <- list(survfit(cfit2, newdata, std.err=FALSE))
csurv  <- survfit(AJmat, p0 =c(entry=1, PCM=0, death=0))



cleanEx()
nameEx("survobrien")
### * survobrien

flush(stderr()); flush(stdout())

### Name: survobrien
### Title: O'Brien's Test for Association of a Single Variable with
###   Survival
### Aliases: survobrien
### Keywords: survival

### ** Examples

xx <- survobrien(Surv(futime, fustat) ~ age + factor(rx) + I(ecog.ps), 
			       data=ovarian) 
coxph(Surv(time, status) ~ age + strata(.strata.), data=xx) 



cleanEx()
nameEx("survreg")
### * survreg

flush(stderr()); flush(stdout())

### Name: survreg
### Title: Regression for a Parametric Survival Model
### Aliases: survreg model.frame.survreg labels.survreg print.survreg.penal
###   print.summary.survreg survReg anova.survreg anova.survreglist
### Keywords: survival

### ** Examples

# Fit an exponential model: the two fits are the same
survreg(Surv(futime, fustat) ~ ecog.ps + rx, ovarian, dist='weibull',
                                    scale=1)
survreg(Surv(futime, fustat) ~ ecog.ps + rx, ovarian,
        dist="exponential")

#
# A model with different baseline survival shapes for two groups, i.e.,
#   two different scale parameters
survreg(Surv(time, status) ~ ph.ecog + age + strata(sex), lung)

# There are multiple ways to parameterize a Weibull distribution. The survreg 
# function embeds it in a general location-scale family, which is a 
# different parameterization than the rweibull function, and often leads
# to confusion.
#   survreg's scale  =    1/(rweibull shape)
#   survreg's intercept = log(rweibull scale)
#   For the log-likelihood all parameterizations lead to the same value.
y <- rweibull(1000, shape=2, scale=5)
survreg(Surv(y)~1, dist="weibull")

# Economists fit a model called `tobit regression', which is a standard
# linear regression with Gaussian errors, and left censored data.
tobinfit <- survreg(Surv(durable, durable>0, type='left') ~ age + quant,
	            data=tobin, dist='gaussian')



cleanEx()
nameEx("survreg.distributions")
### * survreg.distributions

flush(stderr()); flush(stdout())

### Name: survreg.distributions
### Title: Parametric Survival Distributions
### Aliases: survreg.distributions
### Keywords: survival

### ** Examples

# time transformation
survreg(Surv(time, status) ~ ph.ecog + sex, dist='weibull', data=lung)
# change the transformation to work in years
# intercept changes by log(365), everything else stays the same
my.weibull <- survreg.distributions$weibull
my.weibull$trans <- function(y) log(y/365)
my.weibull$itrans <- function(y) 365*exp(y)
survreg(Surv(time, status) ~ ph.ecog + sex, lung, dist=my.weibull)

# Weibull parametrisation
y<-rweibull(1000, shape=2, scale=5)
survreg(Surv(y)~1, dist="weibull")
# survreg scale parameter maps to 1/shape, linear predictor to log(scale)

# Cauchy fit
mycauchy <- list(name='Cauchy',
                 init= function(x, weights, ...) 
                      c(median(x), mad(x)),
                 density= function(x, parms) {
                      temp <- 1/(1 + x^2)
                      cbind(.5 + atan(x)/pi, .5+ atan(-x)/pi,
                            temp/pi, -2 *x*temp, 2*temp*(4*x^2*temp -1))
                      },
                 quantile= function(p, parms) tan((p-.5)*pi),
                 deviance= function(...) stop('deviance residuals not defined')
                 )
survreg(Surv(log(time), status) ~ ph.ecog + sex, lung, dist=mycauchy)



cleanEx()
nameEx("survregDtest")
### * survregDtest

flush(stderr()); flush(stdout())

### Name: survregDtest
### Title: Verify a survreg distribution
### Aliases: survregDtest
### Keywords: survival

### ** Examples

# An invalid distribution (it should have "init =" on line 2)
#  surveg would give an error message
mycauchy <- list(name='Cauchy',
                 init<- function(x, weights, ...) 
                      c(median(x), mad(x)),
                 density= function(x, parms) {
                      temp <- 1/(1 + x^2)
                      cbind(.5 + atan(temp)/pi, .5+ atan(-temp)/pi,
                            temp/pi, -2 *x*temp, 2*temp^2*(4*x^2*temp -1))
                      },
                 quantile= function(p, parms) tan((p-.5)*pi),
                 deviance= function(...) stop('deviance residuals not defined')
                 )

survregDtest(mycauchy, TRUE)



cleanEx()
nameEx("tcut")
### * tcut

flush(stderr()); flush(stdout())

### Name: tcut
### Title: Factors for person-year calculations
### Aliases: tcut [.tcut levels.tcut
### Keywords: survival

### ** Examples

# For pyears, all time variable need to be on the same scale; but
# futime is in months and age is in years
test <- mgus2
test$years <- test$futime/30.5   # follow-up in years

# first grouping based on years from starting age (= current age)
# second based on years since enrollment (all start at 0)
test$agegrp <- tcut(test$age, c(0,60, 70, 80, 100), 
                     c("<=60", "60-70", "70-80", ">80"))
test$fgrp  <- tcut(rep(0, nrow(test)), c(0, 1, 5, 10, 100),
                   c("0-1yr", "1-5yr", "5-10yr", ">10yr"))

# death rates per 1000, by age group
pfit1 <- pyears(Surv(years, death) ~ agegrp, scale =1000, data=test)
round(pfit1$event/ pfit1$pyears) 

#death rates per 100, by follow-up year and age
# there are excess deaths in the first year, within each age stratum
pfit2 <- pyears(Surv(years, death) ~ fgrp + agegrp, scale =1000, data=test)
round(pfit2$event/ pfit2$pyears)  



cleanEx()
nameEx("tmerge")
### * tmerge

flush(stderr()); flush(stdout())

### Name: tmerge
### Title: Time based merge for survival data
### Aliases: tmerge
### Keywords: survival

### ** Examples

# The pbc data set contains baseline data and follow-up status
# for a set of subjects with primary biliary cirrhosis, while the
# pbcseq data set contains repeated laboratory values for those
# subjects.  
# The first data set contains data on 312 subjects in a clinical trial plus
# 106 that agreed to be followed off protocol, the second data set has data
# only on the trial subjects.
temp <- subset(pbc, id <= 312, select=c(id:sex, stage)) # baseline data
pbc2 <- tmerge(temp, temp, id=id, endpt = event(time, status))
pbc2 <- tmerge(pbc2, pbcseq, id=id, ascites = tdc(day, ascites),
               bili = tdc(day, bili), albumin = tdc(day, albumin),
               protime = tdc(day, protime), alk.phos = tdc(day, alk.phos))

fit <- coxph(Surv(tstart, tstop, endpt==2) ~ protime + log(bili), data=pbc2)



cleanEx()
nameEx("tobin")
### * tobin

flush(stderr()); flush(stdout())

### Name: tobin
### Title: Tobin's Tobit data
### Aliases: tobin
### Keywords: datasets

### ** Examples

tfit <- survreg(Surv(durable, durable>0, type='left') ~age + quant,
                data=tobin, dist='gaussian')

predict(tfit,type="response")




cleanEx()
nameEx("transplant")
### * transplant

flush(stderr()); flush(stdout())

### Name: transplant
### Title: Liver transplant waiting list
### Aliases: transplant
### Keywords: datasets

### ** Examples

#since event is a factor, survfit creates competing risk curves
pfit <- survfit(Surv(futime, event) ~ abo, transplant)
pfit[,2]  #time to liver transplant, by blood type
plot(pfit[,2], mark.time=FALSE, col=1:4, lwd=2, xmax=735,
       xscale=30.5, xlab="Months", ylab="Fraction transplanted",
       xaxt = 'n')
temp <- c(0, 6, 12, 18, 24)
axis(1, temp*30.5, temp)
legend(450, .35, levels(transplant$abo), lty=1, col=1:4, lwd=2)

# competing risks for type O
plot(pfit[4,], xscale=30.5, xmax=735, col=1:3, lwd=2)
legend(450, .4, c("Death", "Transpant", "Withdrawal"), col=1:3, lwd=2)



cleanEx()
nameEx("udca")
### * udca

flush(stderr()); flush(stdout())

### Name: udca
### Title: Data from a trial of usrodeoxycholic acid
### Aliases: udca udca1 udca2
### Keywords: datasets

### ** Examples

# values found in table 8.3 of the book
fit1 <- coxph(Surv(futime, status) ~ trt + log(bili) + stage,
          cluster =id , data=udca1)
fit2 <- coxph(Surv(futime, status) ~ trt + log(bili) + stage +
          strata(endpoint), cluster=id,  data=udca2)




cleanEx()
nameEx("untangle.specials")
### * untangle.specials

flush(stderr()); flush(stdout())

### Name: untangle.specials
### Title: Help Process the 'specials' Argument of the 'terms' Function.
### Aliases: untangle.specials
### Keywords: survival

### ** Examples

formula <- Surv(tt,ss) ~ x + z*strata(id)
tms <- terms(formula, specials="strata")
## the specials attribute
attr(tms, "specials")
## main effects 
untangle.specials(tms, "strata")
## and interactions
untangle.specials(tms, "strata", order=1:2)



cleanEx()
nameEx("uspop2")
### * uspop2

flush(stderr()); flush(stdout())

### Name: uspop2
### Title: Projected US Population
### Aliases: uspop2
### Keywords: datasets

### ** Examples

us50 <- uspop2[51:101,, "2000"]  #US 2000 population, 50 and over
age <- as.integer(dimnames(us50)[[1]])
smat <- model.matrix( ~ factor(floor(age/5)) -1)
ustot <- t(smat) %*% us50  #totals by 5 year age groups
temp <- c(50,55, 60, 65, 70, 75, 80, 85, 90, 95)
dimnames(ustot) <- list(c(paste(temp, temp+4, sep="-"), "100+"),
                         c("male", "female"))



cleanEx()
nameEx("xtfrm.Surv")
### * xtfrm.Surv

flush(stderr()); flush(stdout())

### Name: xtfrm.Surv
### Title: Sorting order for Surv objects
### Aliases: xtfrm.Surv sort.Surv order.Surv
### Keywords: survival

### ** Examples

test <- c(Surv(c(10, 9,9, 8,8,8,7,5,5,4), rep(1:0, 5)), Surv(6.2, NA))
test
sort(test)



cleanEx()
nameEx("yates")
### * yates

flush(stderr()); flush(stdout())

### Name: yates
### Title: Population prediction
### Aliases: yates
### Keywords: models survival

### ** Examples

fit1 <- lm(skips ~ Solder*Opening + Mask, data = solder)
yates(fit1, ~Opening, population = "factorial")

fit2 <- coxph(Surv(time, status) ~ factor(ph.ecog)*sex + age, lung)
yates(fit2, ~ ph.ecog, predict="risk")  # hazard ratio



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
