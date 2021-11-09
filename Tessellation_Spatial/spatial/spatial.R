# Generate coordinates for spatially distributed bugs used by NUFEB software
# Joe Weaver jeweave4@ncsu.edu
#
# This is currently intended as a one-shot for generating a small CSV file
# with a bunch of realizations.  Parameters are edited by hand in code.
#
# If we keep using it, we will want to discuss wrapping the code up in functions
# so we can generate individual point patterns. The utility of this script could
# be reproduced by calling the functions.
#
# That would be useful, for example, if we end up with a workflow that dynamically
# selects parameter values based on previous results.

library(spatstat)
library(here)
# Apart from the csv generation, there's a bunch of base-R in here and that's OK
library(dplyr)
library(tidyr)
library(readr)

set.seed(1701)

#rescale a point pattern distribution
scale_pp <- function(pp, xscale, yscale){
  return(cbind(pp$x*xscale,pp$y*yscale))
}


# removes bugs which are less than min_dist apart
# Very inefficient, but straightforward 
# rdist can give the distance matrix quickly, but reassociating that
# with a specific point to remove is somewhat tricky
remove_collisions <- function(pp,min_dist){
  filtered_ptsx <- vector()
  filtered_ptsy <- vector()
  # return an empty list if there's 0 points to filter
  # such patterns can occur when the spatstats generators are called
  # with low intensities
  if(nrow(pp)<1){
    filtered_points = cbind(filtered_ptsx,filtered_ptsy)
    return(filtered_points)
  }
  # in the case of just one point, no filtering necessary
  if(nrow(pp)==1){
    filtered_ptsx = append(filtered_ptsx,pp[1,1])
    filtered_ptsy = append(filtered_ptsy,pp[1,2])
    filtered_points = cbind(filtered_ptsx,filtered_ptsy)
    return(filtered_points)
  }
  # compute all pairwise distances
  # for each run through outer loop, if any pair is too close, do not record point a
  for (pointa in 1:(nrow(pp) - 1)) {
    pointb_start <- pointa + 1
    tooclose = FALSE
    #TODO can remove fewer points by only checking against point B not already
    # removed
    for (pointb in pointb_start:nrow(pp)) {
      x1 = pp[pointa,1]
      x2 = pp[pointb,1]
      y1 = pp[pointa,2]
      y2 = pp[pointb,2]
      dist <- sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
      if (dist < min_dist) {
        tooclose = TRUE
        }
    }
    if (!tooclose) {
  
      filtered_ptsx = append(filtered_ptsx,pp[pointa,1])
      filtered_ptsy = append(filtered_ptsy,pp[pointa,2])
    }
  }
  # need to add final point in list
  filtered_ptsx = append(filtered_ptsx,pp[nrow(pp),1])
  filtered_ptsy = append(filtered_ptsy,pp[nrow(pp),2])
  filtered_points = cbind(filtered_ptsx,filtered_ptsy)
  return(filtered_points)
}

# default scales and minimum distance
xscale <- 1e-4
yscale <- 1e-4
mdist <- 2*1.94e-6 # twice the diameter of the bugs
# number of realizations per unique pattern process+parameter set
# eg, if we want a poisson process with intensity 2 & nsim 3 we will end up
# generating 3 point sets
nsim <- 3
genplots = TRUE # generate plots?
writeplots = FALSE # save plots to disk, meaningless if we don't generate

# The following code generates a bunch of realizations for different
# point pattern processes and parameters.  It has rampant code duplication
# TODO pay off that technical debt ASAP if we decide to use this more

# very hacky id number generation, iterated with each realization
id <- 0

# intialize the dataframe which will be written out as a CSV
point_patterns <- data.frame(x=double(),
                              y=double(),
                              pattern=character(),
                              target_intensity=double(),
                              realized_intensity=double(),
                              radius=double(),
                              pts_per_cluster=integer(),
                              id=integer(),
                              stringsAsFactors=FALSE)

# generate random (poisson) point patterns with target intensities
# of 2, 20, 40, and 80 and do so nsim times. Plots may or may not be generated
lambdas <- c(rep(2,nsim),rep(20,nsim),rep(40,nsim),rep(80,nsim))
for (lambda in lambdas) {
  rpoints <- rpoispp(lambda)
  filtered <- remove_collisions(scale_pp(rpoints,1e-4,1e-4),mdist)
  #only produce output if we have points after filtering
  #TODO wrap rpoints and generation in a loop which doesn't end until 
  #a valid point is generated.  Could also enhance so that it runs until a particular
  #intensity is realized.  Probably want a max_iteration watchdog
  if(nrow(filtered)>0 ){
    id <- id+1
    #TODO DRY using npoints
    npoints <-length(filtered[,1])
    #create a dataframe capturing all the relevant data
    #note the dataframe is in TIDY form rather than two tables in normal form
    realization <-  data.frame(x=filtered[,1],
                              y=filtered[,2],
                              pattern="poisson",
                              target_intensity=rep(lambda,npoints),
                              realized_intensity=rep(npoints,npoints), # i know
                              radius=rep(0,npoints), # not applicable
                              pts_per_cluster=rep(0,npoints), # not applicable
                              id=rep(id,npoints),
                              stringsAsFactors=FALSE)

    #fold realization into db of all realizations
    point_patterns <- rbind(point_patterns,realization)
    # Create a simple diagnostic plot of the points
    if(genplots){
      # TI is the target intensity, RI is realized intensity (pre filtering),
      # FI is the post-filtering intensity
      title=paste("POISSON TI: ",lambda,
                  ". RI: ",intensity(rpoints), ". FI: ",length(filtered[,1]))
      if(writeplots){
        filename <- paste0("random_lambda-",lambda,
                         "_id-",id,"_npts-",length(filtered[,1]),".png")
        #TODO ensure noclobber
        png(here::here("output",filename),width=800,height=800,units="px")
      }
      # TODO DRY on scaling RPOINTS for plot
      plot(rpoints$x*1e-4,rpoints$y*1e-4,col="black",main=title,
           xlim=c(0,xscale),ylim=c(0,yscale))
      points(filtered,col="blue",pch=20)
      if(writeplots){
        dev.off()
      }
    }
  }
}



# generate regular point patterns with target intensities
# and repulsion (radius) levels
# and do so nsim times. Plots may or may not be generated
lambdas <- c(rep(2,nsim),rep(20,nsim),rep(40,nsim),rep(80,nsim))
for (lambda in lambdas) {
  for(repel in c(rep(0.1,nsim),rep(0.05,nsim),rep(0.01,nsim))){
    #FIXME 0.1 should be repel
    #n.b. for the paper, we only look at bugs with a repel of 0.1
    rpoints <-   rpoints <- rMaternII(lambda,0.1)#,win=sim_bounds)
    filtered <- remove_collisions(scale_pp(rpoints,1e-4,1e-4),mdist)
    #only produce output if we have points after filtering
    #TODO wrap rpoints and generation in a loop which doesn't end until 
    #a valid point is generated.  Could also enhance so that it runs until a particular
    #intensity is realized.  Probably want a max_iteration watchdog
    if(nrow(filtered)>0 ){
      id <- id+1
      #TODO DRY using npoints
      npoints <-length(filtered[,1])
      #create a dataframe capturing all the relevant data
      #note the dataframe is in TIDY form rather than two tables in normal form
      realization <-  data.frame(x=filtered[,1],
                                 y=filtered[,2],
                                 pattern="regular",
                                 target_intensity=rep(lambda,npoints),
                                 realized_intensity=rep(npoints,npoints), # i know
                                 radius=rep(repel,npoints), 
                                 pts_per_cluster=rep(0,npoints), # not applicable
                                 id=rep(id,npoints),
                                 stringsAsFactors=FALSE)
      
      #fold realization into db of all realizations
      point_patterns <- rbind(point_patterns,realization)
      # Create a simple diagnostic plot of the points
      if(genplots){
        # TI is the target intensity, RI is realized intensity (pre filtering),
        # FI is the post-filtering intensity
        title=paste("REGULAR TI: ",lambda,
                    ". RI: ",intensity(rpoints), 
                    ". FI: ",length(filtered[,1]),
                     "Rep: ",repel)
        if(writeplots){
          filename <- paste0("repel_lambda-",lambda,
                             "repel-",repel,
                             "_id-",id,"_npts-",length(filtered[,1]),".png")
          #TODO ensure noclobber
          png(here::here("output",filename),width=800,height=800,units="px")
        }
        # TODO DRY on scaling RPOINTS for plot
        plot(rpoints$x*1e-4,rpoints$y*1e-4,col="black",main=title,
             xlim=c(0,xscale),ylim=c(0,yscale))
        points(filtered,col="blue",pch=20)
        if(writeplots){
          dev.off()
        }
      }
    }
  }
}

# generate regular point patterns with target intensities
# and repulsion (radius) levels
# and do so nsim times. Plots may or may not be generated
lambdas <- c(rep(2,nsim),rep(20,nsim),rep(40,nsim),rep(80,nsim))
for (lambda in lambdas) {
  for(radius in c(rep(0.1,nsim),rep(0.05,nsim),rep(0.01,nsim))){
    for(children in c(rep(3,nsim),rep(5,nsim),rep(10,nsim))){
      #rMatClust uses lambda to see parents, so rescale
      #by the number of children and make sure you've got at least
      #one point
      rescale_lambda <- as.integer(max(lambda/children,1))
      rpoints <-   rpoints <- rMatClust(lambda/children,radius,children)
      filtered <- remove_collisions(scale_pp(rpoints,1e-4,1e-4),mdist)
      #only produce output if we have points after filtering
      #TODO wrap rpoints and generation in a loop which doesn't end until 
      #a valid point is generated.  Could also enhance so that it runs until a particular
      #intensity is realized.  Probably want a max_iteration watchdog
      if(nrow(filtered)>0 ){
        id <- id+1
        #TODO DRY using npoints
        npoints <-length(filtered[,1])
        #create a dataframe capturing all the relevant data
        #note the dataframe is in TIDY form rather than two tables in normal form
        realization <-  data.frame(x=filtered[,1],
                                   y=filtered[,2],
                                   pattern="aggregate",
                                   target_intensity=rep(lambda,npoints),
                                   realized_intensity=rep(npoints,npoints), # i know
                                   radius=rep(radius,npoints), 
                                   pts_per_cluster=rep(children,npoints), # not applicable
                                   id=rep(id,npoints),
                                   stringsAsFactors=FALSE)
        
        #fold realization into db of all realizations
        point_patterns <- rbind(point_patterns,realization)
        # Create a simple diagnostic plot of the points
        if(genplots){
          # TI is the target intensity, RI is realized intensity (pre filtering),
          # FI is the post-filtering intensity
          title=paste("Aggregate TI: ",lambda,
                      ". RI: ",intensity(rpoints), 
                      ". FI: ",length(filtered[,1]),
                      "Rad: ",radius,
                      "Kids: ",children)
          if(writeplots){
            filename <- paste0("aggregate_lambda-",lambda,
                               "radius-",radius,
                               "children-",children,
                               "_id-",id,"_npts-",length(filtered[,1]),".png")
            #TODO ensure noclobber
            png(here::here("output",filename),width=800,height=800,units="px")
          }
          # TODO DRY on scaling RPOINTS for plot
          plot(rpoints$x*1e-4,rpoints$y*1e-4,col="black",main=title,
               xlim=c(0,xscale),ylim=c(0,yscale))
          points(filtered,col="blue",pch=20)
          if(writeplots){
            dev.off()
          }
        }
      }
    }
  }
}
# Write the generated patterns as a csv file for use in generating NUFEB runs
gentime <- format(Sys.time(), "%Y-%m-%d-%H-%M-%s")
write_csv(point_patterns,here::here("output",paste0(gentime,"_point_patterns.csv")))

#TODO write out library versions

# parameters
# random intensity (lambda)
# clustered parent_intensity(kappa1), radius of cluse (scale), mean pts per cluster (mu)
# regular proposal_intensity(kappa)2, inhib_dist(r)
