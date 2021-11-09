# Plots and statistical analysis of spatial patterns for paper

library(here)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(RColorBrewer)
library(cowplot)
library(broom)

final_timestep <- read_csv(
  here::here("data","final_timestep_combined_gathered_results_metadata.csv")) %>%
  filter(!is.na(`Is Edge Facet`))

centroid_info <- read_csv(here::here("data","centroid_dist.csv"))
final_timestep<-final_timestep %>% left_join(y=centroid_info,by=c("RunID","seed"))


dist_metrics <- read_csv(here::here("data","seed_dist_metrics.csv"))
final_timestep<-final_timestep %>% left_join(y=dist_metrics,by=c("RunID","seed"))
# n.b. there will be some with -Inf log inv sq dist, but that's only when there's
# one seed location and one only.

filtered <- final_timestep %>%
  filter(Radius %in% c(0.1,0)) %>%
  filter(between(Intensity,20,45)) %>%
  filter(`Points Per Cluster` %in% c(0,3,5)) %>%
  filter(!is.na(`Is Edge Facet`))

max_perims <- filtered %>% group_by(RunID) %>% summarise(max_perim = max(`Facet Perimeter (pixels)`),
                                                         max_area = max(`Facet Area (pixels)`))
filtered<-filtered %>% left_join(y=max_perims) %>% mutate(normalized_perim = `Facet Perimeter (pixels)`/max_perim,
                                                          normalized_area = `Facet Area (pixels)`/max_area)
filtered <- filtered %>% mutate(dist_bin = cut(`dist_m`,breaks=10))
filtered <- filtered %>% mutate(intensity_bin = cut(`Intensity`,breaks=5))
poisson <- filtered %>% filter(Pattern %in% c("poisson"))
aggregate <- filtered %>% filter(Pattern %in% c("aggregate"))
regular <- filtered %>% filter(Pattern %in% c("regular"))



theme_update(axis.title = element_text(size=14),
             axis.text  = element_text(size=14),
             strip.text = element_text(size=13),
             axis.text.x = element_text(angle=0),
             title = element_text(size=18,hjust=0.5))



filtered$Pattern = factor(filtered$Pattern, levels=c("aggregate","regular","poisson"))
pattern.labs <- c("Aggregate", "Random", "Regular")
names(pattern.labs) <- c("aggregate", "poisson", "regular")

edges.labs <- c("Non-Edge Colony", "Edge Colony")
names(edges.labs) <- c(FALSE, TRUE)

p <- ggplot(filtered,aes(x=log(`Facet Area (pixels)`),
                         y=`Median-Normed Colony Area`,
                         color=factor(`intensity_bin`))) +
  geom_abline(slope=0,intercept=0,linetype="dashed",colour="grey")+
  geom_point(size=2,alpha=0.8)+
  xlab("lognorm of Facet Area") +
  facet_grid(cols=vars(`Is Edge Facet`), rows=vars(Pattern), scale="free_x",
             labeller = labeller(Pattern = pattern.labs, `Is Edge Facet` = edges.labs)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(size=.2,colour="black"),
        legend.position="top",
        legend.text = element_text(size=8),
        legend.title= element_text(size=12))+
  scale_color_brewer(palette = "Set1")+
  guides(colour=guide_legend(title="Intensity Range"))
p

ggsave(here::here("output","lognorm_facet_area.png"),
                  width=6,height=6,units="in",dpi=300)
nbins=50
nbinwidth=1.0/nbins
pA <- ggplot(filtered ,aes(x=(normalized_area),
                         color=factor(`Pattern`),
                         fill=factor(`Pattern`))) +
  geom_histogram(aes(y=stat(density)*nbinwidth ),alpha=.4,binwidth=nbinwidth) +
  scale_color_brewer(palette = "Set1")+
  scale_fill_brewer(palette = "Set1")+
  facet_grid(rows=vars(Pattern),scale="free_x",
             labeller = labeller(Pattern = pattern.labs, `Is Edge Facet` = edges.labs))+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(size=.2,colour="black"),
        legend.position="none",
        legend.text = element_text(size=8),
        legend.title= element_text(size=12),
        strip.background = element_blank(),
        strip.text = element_blank(),
        axis.text.x=element_text(size=10),
        axis.text.y=element_text(size=8))+
  scale_color_brewer(palette = "Set1")+
  scale_y_continuous(labels=scales::percent_format(accuracy = 5L)) +
  guides(colour=guide_legend(title="Spatial Pattern"),
         fill = guide_legend("Spatial Pattern"))+
  xlab("Normalized Facet Area")+
  ylab("Frequency")
pA

pB <- ggplot(filtered,aes(x=(`Median-Normed Colony Area`),
                         color=factor(`Pattern`),
                         fill=factor(`Pattern`))) +
  geom_histogram(aes(y=stat(density)*nbinwidth ),alpha=.4,binwidth=nbinwidth) +
  scale_color_brewer(palette = "Set1")+
  scale_fill_brewer(palette = "Set1")+
  facet_grid(rows=vars(Pattern),scale="free_x",
    labeller = labeller(Pattern = pattern.labs, `Is Edge Facet` = edges.labs))+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(size=.2,colour="black"),
        legend.position="none",
        legend.text = element_text(size=8),
        legend.title= element_text(size=12),
        strip.text.y=element_text(size=10),
        axis.text.x=element_text(size=10),
        axis.text.y=element_text(size=8))+
  scale_color_brewer(palette = "Set1")+
  scale_y_continuous(labels=scales::percent_format(accuracy = 5L)) +
  guides(colour=guide_legend(title="Spatial Pattern"),
         fill = guide_legend("Spatial Pattern"))+
  xlab("Median-Normed Colony Area")+
  ylab("")
pB

plot_grid(pA,pB,labels=c('A','B'))

ggsave(here::here("output","normed_area_distributions.png"),
       width=8,height=4,units="in",dpi=300)

nbins=50
nbinwidth=1.0/nbins
pA <- ggplot(filtered %>% filter(!`Is Edge Facet`) ,aes(x=(normalized_area),
                           color=factor(`Pattern`),
                           fill=factor(`Pattern`))) +
  geom_histogram(aes(y=stat(density)*nbinwidth ),alpha=.4,binwidth=nbinwidth) +
  scale_color_brewer(palette = "Set1")+
  scale_fill_brewer(palette = "Set1")+
  facet_grid(rows=vars(Pattern),scale="free_x",
             labeller = labeller(Pattern = pattern.labs, `Is Edge Facet` = edges.labs))+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(size=.2,colour="black"),
        legend.position="none",
        legend.text = element_text(size=8),
        legend.title= element_text(size=12),
        strip.background = element_blank(),
        strip.text = element_blank(),
        axis.text.x=element_text(size=10),
        axis.text.y=element_text(size=8))+
  scale_color_brewer(palette = "Set1")+
  scale_y_continuous(labels=scales::percent_format(accuracy = 5L)) +
  guides(colour=guide_legend(title="Spatial Pattern"),
         fill = guide_legend("Spatial Pattern"))+
  xlab("Normalized Facet Area")+
  ylab("Frequency")
pA

pB <- ggplot(filtered %>% filter(!`Is Edge Facet`) ,aes(x=(`Median-Normed Colony Area`),
                          color=factor(`Pattern`),
                          fill=factor(`Pattern`))) +
  geom_histogram(aes(y=stat(density)*nbinwidth ),alpha=.4,binwidth=nbinwidth) +
  scale_color_brewer(palette = "Set1")+
  scale_fill_brewer(palette = "Set1")+
  facet_grid(rows=vars(Pattern),scale="free_x",
             labeller = labeller(Pattern = pattern.labs, `Is Edge Facet` = edges.labs))+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(size=.2,colour="black"),
        legend.position="none",
        legend.text = element_text(size=8),
        legend.title= element_text(size=12),
        strip.text.y=element_text(size=10),
        axis.text.x=element_text(size=10),
        axis.text.y=element_text(size=8))+
  scale_color_brewer(palette = "Set1")+
  scale_y_continuous(labels=scales::percent_format(accuracy = 5L)) +
  guides(colour=guide_legend(title="Spatial Pattern"),
         fill = guide_legend("Spatial Pattern"))+
  xlab("Median-Normed Colony Area")+
  ylab("")
pB

plot_grid(pA,pB,labels=c('A','B'))

ggsave(here::here("output","normed_area_distributions_no_edge.png"),
       width=8,height=4,units="in",dpi=300)


##TODO we could probably work through summarizing all this into a nested
##dataframe and do some fancy tricks with kableExtra to automatically write
##the table, but getting it all working is taking far too long for a good ROI
##right now
## Briefly, for nested columns, smartly use pivot_longer and pivot_wider
## to get easily parsed column names, e.g. Edge-linear-r2, then use
## kableExtra stuff to rename. For nested rows, use the group_rows stuff

##Note, using ensym rather than the more usual enquo since we're
##using the column name as part of formula
glanceFit <- function(theFactor, factorName, fitName){
  theFactor = ensym(theFactor)
  facetAreaGlance <- filtered %>% group_by(Pattern,`Is Edge Facet`) %>%
    do(the_fit = glance(lm(`Median-Normed Colony Area` ~ !!(theFactor),
                           data = .))) %>% unnest(the_fit) %>%
    mutate(Factor = factorName) %>%
    mutate(Fit = fitName)%>% select(Pattern, `Is Edge Facet`, Factor,
                                    Fit, r.squared, p.value)
  return(facetAreaGlance)
}


##TODO DRY this up with glanceFit
glanceFitLog <- function(theFactor, factorName, fitName){
  theFactor = ensym(theFactor)
  facetAreaGlance <- filtered %>% group_by(Pattern,`Is Edge Facet`) %>%
    do(the_fit = glance(lm(`Median-Normed Colony Area` ~ log(!!(theFactor)),
                           data = .))) %>% unnest(the_fit) %>%
    mutate(Factor = factorName) %>%
    mutate(Fit = fitName)%>% select(Pattern, `Is Edge Facet`, Factor,
                                    Fit, r.squared, p.value)
  return(facetAreaGlance)
}

# Linear single variable regressions
fitSummary <- glanceFit(`Facet Area (pixels)`, "Area", "Linear")
fitSummary <- rbind(fitSummary,
                glanceFit(`Facet Perimeter (pixels)`, "Perimeter", "Linear"))
fitSummary <- rbind(fitSummary,
                    glanceFit(`Facet Sides`, "Sides", "Linear"))
fitSummary <- rbind(fitSummary,
                    glanceFit(`dist_m`, "Centroid Separation", "Linear"))
fitSummary <- rbind(fitSummary,
                    glanceFit( `Facet Aspect Ratio`, "Aspect Ratio", "Linear"))

#Ln(x) single variable regressions
fitSummary <- rbind(fitSummary,
                    glanceFitLog(`Facet Area (pixels)`, "Area", "Log"))
fitSummary <- rbind(fitSummary,
                    glanceFitLog(`Facet Perimeter (pixels)`, "Perimeter", "Log"))
fitSummary <- rbind(fitSummary,
                    glanceFitLog(`Facet Sides`, "Sides", "Log"))
fitSummary <- rbind(fitSummary,
                    glanceFitLog(`dist_m`, "Centroid Separation", "Log"))
fitSummary <- rbind(fitSummary,
                    glanceFitLog( `Facet Aspect Ratio`, "Aspect Ratio", "Log"))


## Multiple linear regressions

### No transform, no edge
lmfit<-(lm(`Median-Normed Colony Area` ~
             `Facet Area (pixels)` +
             `Facet Perimeter (pixels)` +
             `Facet Sides` +
             `Facet Aspect Ratio` +
             `dist_m` +
             `log_inv_sq_dist_m`,
           data = aggregate %>% filter(!`Is Edge Facet`)))
sm<-step(lmfit)
tidy(sm)
glance(sm)

# Aspect ratio was not significant, so remove
lmfit<-(lm(`Median-Normed Colony Area` ~
             `Facet Perimeter (pixels)` +
                          `dist_m`,
           data = aggregate %>% filter(!`Is Edge Facet`)))
tidy(lmfit)
glance(lmfit)
car::vif(lmfit)

### No transform, include edges
lmfit<-(lm(`Median-Normed Colony Area` ~
             `Facet Area (pixels)` +
             `Facet Perimeter (pixels)` +
             `Facet Sides` +
             `Facet Aspect Ratio` +
             `dist_m` +
             `log_inv_sq_dist_m`,
           data = aggregate ))
sm<-step(lmfit)
tidy(sm)
glance(sm)
# All are signficant, let's check VIF and remove top
car::vif(sm)

lmfit<-(lm(`Median-Normed Colony Area` ~
             `Facet Area (pixels)` +
             `Facet Sides` +
             `Facet Aspect Ratio` +
             `dist_m` +
             `log_inv_sq_dist_m`,
           data = aggregate ))
tidy(lmfit)
glance(lmfit)
car::vif(lmfit)


### Log transform, no edge
## not incorporating inv sq distance due to log(0)
lmfit<-(lm(`Median-Normed Colony Area` ~
             log(`Facet Area (pixels)`) +
             log(`Facet Perimeter (pixels)`) +
             log(`Facet Sides`) +
             log(`Facet Aspect Ratio`) +
             log(`dist_m`),
           data = aggregate %>% filter(!`Is Edge Facet`)))
sm<-step(lmfit)
tidy(sm)
glance(sm)
car::vif(sm)

# technically can get the same info from glance and tidy on sm above
lmfit<-(lm(`Median-Normed Colony Area` ~
             log(`Facet Area (pixels)`) +
             log(`dist_m`),
           data = aggregate %>% filter(!`Is Edge Facet`)))

tidy(lmfit)
glance(lmfit)
car::vif(sm)

### Log transform, with edges
## not incorporating inv sq distance due to log(0)
lmfit<-(lm(`Median-Normed Colony Area` ~
             log(`Facet Area (pixels)`) +
             log(`Facet Perimeter (pixels)`) +
             log(`Facet Sides`) +
             log(`Facet Aspect Ratio`) +
             log(`dist_m`),
           data = aggregate ))
sm<-step(lmfit)
tidy(sm)
glance(sm)
car::vif(sm)

# remove highest VIF
lmfit<-(lm(`Median-Normed Colony Area` ~
             log(`Facet Perimeter (pixels)`) +
             log(`Facet Sides`) +
             log(`Facet Aspect Ratio`) +
             log(`dist_m`),
           data = aggregate ))
sm<-step(lmfit)
tidy(lmfit)
glance(lmfit)
car::vif(lmfit)

# still vif > 2.75, remove next highest
lmfit<-(lm(`Median-Normed Colony Area` ~
             log(`Facet Sides`) +
             log(`Facet Aspect Ratio`) +
             log(`dist_m`),
           data = aggregate ))
sm<-step(lmfit)
tidy(lmfit)
glance(lmfit)
car::vif(lmfit)
