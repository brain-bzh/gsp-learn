library("neuRosim")
# Valeurs constantes
dim <- c(53,63,46)
nscan <- 421
TR <- 2
total.time <- nscan * TR

onsets.F <- c((10:30),(100:120),(160:180),(190:210),(250:270),(280:300),(400:420))*TR #
onsets.H <- c((40:60),(70:90),(130:150),(220:240),(310:330),(340:360),(370:390))*TR #
lis<-(1:nscan)
event<-lis
event[lis]<-'Rest'
event[(onsets.F/TR)+1]<-'Face'
event[(onsets.H/TR)+1]<-'House'
write(event,'D:/sim/event_sim.csv')
region.1a.radius <- 3
region.1b.radius <- 3
region.4a.radius <- 3
region.2a.radius <-5
region.2b.radius <- 5
region.3a.radius <- 4
region.3b.radius <- 4
region.4b.radius <- 3

onsets <- list(onsets.H, onsets.F)
onsets.regions <- list(onsets, onsets, onsets, onsets, onsets,onsets, onsets, onsets)
dur <- list(0, 0)
dur.regions <- list(dur, dur, dur, dur, dur, dur, dur, dur)

# Valeurs qui varient selon sujet
library("oro.nifti")
mask <- readNIfTI("D:/mask.nii.gz")
Haxby <- readNIfTI("D:/restbaseline.nii.gz")
baseline <- apply(Haxby@.Data, 1:3, mean)

r1a.center<- c(12,21,9)#110
r1a.d<- c(239, 258)
r1b.center<- c(39,21,9)#697
r1b.d<- c(229,247)
r2a.center<- c(23,7,16)#93
r2a.d<- c(225,225)
r2b.center<- c(30,7,16)#46
r2b.d<- c(252,252)
r3a.center<- c(31,12,13)#200
r3a.d<- c(309,291)
r3b.center<- c(21,13,13)#716
r3b.d<- c(333,323)
r4a.center<- c(35,21,13)#860
r4a.d<- c(335,314)
r4b.center<- c(17,22,12)#680
r4b.d<- c(308,288)
snr.or <- 4.3

for(s in 2:5) {
# Assign value to each subject
region.1a.center <- round(r1a.center*runif(3,min=0.9, max=1.1),0)
region.1b.center <- round(r1b.center*runif(3,min=0.9, max=1.1),0)
region.2a.center <- round(r2a.center*runif(3,min=0.99, max=1.01),0)
region.2b.center <- round(r2b.center*runif(3,min=0.99, max=1.01),0)
region.3a.center <- round(r3a.center*runif(3,min=0.9, max=1.1),0)
region.3b.center <- round(r3b.center*runif(3,min=0.9, max=1.1),0)
region.4a.center <- round(r4a.center*runif(3,min=0.9, max=1.1),0)
region.4b.center <- round(r4b.center*runif(3,min=0.9, max=1.1),0)

region.1a.d <- list(r1a.d[1]*runif(1,min=0.95, max=1.05),r1a.d[2]*runif(1,min=0.95, max=1.05))
region.1b.d <- list(r1b.d[1]*runif(1,min=0.95, max=1.05),r1b.d[2]*runif(1,min=0.95, max=1.05))
region.2a.d <- list(r2a.d[1]*runif(1,min=0.95, max=1.05),r2a.d[2]*runif(1,min=0.95, max=1.05))
region.2b.d <- list(r2b.d[1]*runif(1,min=0.95, max=1.05),r2b.d[2]*runif(1,min=0.95, max=1.05))
region.3a.d <- list(r3a.d[1]*runif(1,min=0.95, max=1.05),r3a.d[2]*runif(1,min=0.95, max=1.05))
region.3b.d <- list(r3b.d[1]*runif(1,min=0.95, max=1.05),r3b.d[2]*runif(1,min=0.95, max=1.05))
region.4a.d<- list(r4a.d[1]*runif(1,min=0.95, max=1.05),r4a.d[2]*runif(1,min=0.95, max=1.05))
region.4b.d<- list(r4b.d[1]*runif(1,min=0.95, max=1.05),r4b.d[2]*runif(1,min=0.95, max=1.05))
effect <- list(region.1a.d, region.1b.d, region.2a.d, region.2b.d,region.3a.d,region.3b.d,region.4a.d,region.4b.d)
snr <- snr.or*runif(1,min=0.9, max=1.1)


design <- simprepTemporal(regions = 8, onsets = onsets.regions,
durations = dur.regions, hrf = "Balloon", TR = TR,
totaltime = total.time, effectsize = effect)

spatial <- simprepSpatial(regions = 8, coord = list(region.1a.center,
region.1b.center, region.2a.center, region.2b.center, region.3a.center, region.3b.center,region.4a.center, region.4b.center),
radius = c(region.1a.radius, region.1b.radius, region.2a.radius,
region.2b.radius, region.3a.radius, region.3b.radius, region.4a.radius, region.4b.radius), form = "sphere", fading = 0.01)

name_val = paste('D:/sim/sim',s,'val.txt', sep='_')
write.table(rbind(region.1a.center,region.1b.center,region.2a.center,
                  region.2b.center,region.3a.center,region.3b.center,region.4a.center,
                  region.4b.center,region.1a.d,region.1b.d,region.2a.d ,
                  region.2b.d,region.3a.d,region.3b.d,region.4a.d,region.4b.d,snr),file=name_val)


for(n in 1:50) {
sim.data <- simVOLfmri(design = design, image = spatial,
SNR = snr, noise = "mixture"
, type = "rician", rho.temp = c(0.142,
0.108, 0.084), base= baseline, rho.spat = 0.4, w = c(0.05, 0.1, 0.01,
0.09, 0.05, 0.7), dim = dim, nscan = nscan, vee = 0,template=mask,
spat = "gaussRF")

sim.nifti <- nifti(img = sim.data, dim=c(53,63,46,nscan),pixdim = c(-1,3,3,3,2,0,0,0),
xyzt_units=10)
name = paste('D:/sim/sim',s,n, sep='_')

writeNIfTI(sim.nifti,name)
}
}