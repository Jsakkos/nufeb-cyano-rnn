# Plots and statistical analysis of spatial patterns for paper

library(here)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(RColorBrewer)
library(cowplot)

final_timestep <- read_csv(
  here::here("data","debug_final_timestep_combined_gathered_results_metadata.csv")) %>%
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

plot_factor_histogram_panel <- function(df,factor_name,factor_string,lhs=TRUE,
                                        pct_scale=5L,ymax=-1,nbins=50){
  nbinwidth=1.0/nbins
  if(!lhs){
    df <- df %>% filter(!`Is Edge Facet`)
  }
  ystring = "Frequency"
  if(!lhs){
    ystring=""
  }

  pB <- ggplot(df ,aes(x=!! enquo(factor_name),
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
    scale_y_continuous(labels=scales::percent_format(accuracy = pct_scale)) +
    guides(colour=guide_legend(title="Spatial Pattern"),
           fill = guide_legend("Spatial Pattern"))+
    xlab(factor_string)+
    ylab(ystring)
    if(ymax>0){
      pB <- pB+coord_cartesian(ylim=c(0,ymax))
    }
  if(lhs){
    pB<-pB+theme(strip.text.y = element_blank(),
          strip.background = element_blank())
  }
  return(pB)
}

# Could also probably dry this up, but getting tired of tideval quo unquo stuff
hist_area <- plot_grid(plot_factor_histogram_panel(filtered,
                                      normalized_area,
                                      "Normalized Area",
                                      lhs=TRUE,
                                      pct_scale=2L,
                                      ymax=0.1,nbins=50),
          plot_factor_histogram_panel(filtered,
                                      normalized_area,
                                      "Normalized Area",
                                      lhs=FALSE,
                                      pct_scale=2L,
                                      ymax=0.1,nbins=50),
          labels=c('A - No Edge Facets','B - Edge Facets Included'))
hist_area
ggsave(here::here("output","SInormed_area_distributions.png"),
       width=8,height=4,units="in",dpi=300)

hist_perim <- plot_grid(plot_factor_histogram_panel(filtered,
                                                   normalized_perim,
                                                   "Normalized Perimiter",
                                                   lhs=TRUE,
                                                   pct_scale=2L,
                                                   ymax=0.1,nbins=50),
                       plot_factor_histogram_panel(filtered,
                                                   normalized_perim,
                                                   "Normalized Perimiter",
                                                   lhs=FALSE,
                                                   pct_scale=2L,
                                                   ymax=0.1,nbins=50),
                       labels=c('A - No Edge Facets','B - Edge Facets Included'))
hist_perim
ggsave(here::here("output","normed_perim_distributions.png"),
       width=8,height=4,units="in",dpi=300)

hist_sides <- plot_grid(plot_factor_histogram_panel(filtered,
                                                    `Facet Sides`,
                                                    "Facet Sides",
                                                    lhs=TRUE,
                                                    pct_scale=2L,
                                                    ymax=0.5,nbins=50),
                        plot_factor_histogram_panel(filtered,
                                                    `Facet Sides`,
                                                    "Facet Sides",
                                                    lhs=FALSE,
                                                    pct_scale=2L,
                                                    ymax=0.5,nbins=50),
                        labels=c('A - No Edge Facets','B - Edge Facets Included'))
hist_sides
ggsave(here::here("output","facet_sides_distributions.png"),
       width=8,height=4,units="in",dpi=300)

hist_dist <- plot_grid(plot_factor_histogram_panel(filtered %>% mutate(logdist = log(dist_m)),
                                                    logdist,
                                                    "ln(Centroid-Seed Distance)",
                                                    lhs=TRUE,
                                                    pct_scale=1L,
                                                    ymax=0.04,nbins=30),
                        plot_factor_histogram_panel(filtered %>% mutate(logdist = log(dist_m)),
                                                    logdist,
                                                    "ln(Centroid-Seed Distance)",
                                                    lhs=FALSE,
                                                    pct_scale=1L,
                                                    ymax=0.04,nbins=30),
                        labels=c('A - No Edge Facets','B - Edge Facets Included'))
hist_dist
ggsave(here::here("output","centroid_seed_distance_distributions.png"),
       width=8,height=4,units="in",dpi=300)

hist_AR <- plot_grid(plot_factor_histogram_panel(filtered %>% mutate(inv_ar = 1/(`Facet Aspect Ratio`)),
                                                    inv_ar,
                                                    "1/Facet Aspect Ratio",
                                                    lhs=TRUE,
                                                    pct_scale=2L,
                                                    ymax=0.1,nbins=50),
                        plot_factor_histogram_panel(filtered %>% mutate(inv_ar = 1/(`Facet Aspect Ratio`)),
                                                    inv_ar,
                                                    "1/Facet Aspect Ratio",
                                                    lhs=FALSE,
                                                    pct_scale=2L,
                                                    ymax=0.1,nbins=50),
                        labels=c('A - No Edge Facets','B - Edge Facets Included'))
hist_AR
ggsave(here::here("output","aspect_ratio_distributions.png"),
       width=8,height=4,units="in",dpi=300)

hist_lisd <- plot_grid(plot_factor_histogram_panel(filtered,
                                                   log_inv_sq_dist_m,
                                                   "Log Inverse Square Distance",
                                                 lhs=TRUE,
                                                 pct_scale=2L,
                                                 ymax=0.08,nbins=50),
                     plot_factor_histogram_panel(filtered,
                                                 log_inv_sq_dist_m,
                                                 "Log Inverse Square Distance",
                                                 lhs=FALSE,
                                                 pct_scale=2L,
                                                 ymax=0.08,nbins=50),
                     labels=c('A - No Edge Facets','B - Edge Facets Included'))
hist_lisd
ggsave(here::here("output","log_inverse_square_distance_distributions.png"),
       width=8,height=4,units="in",dpi=300)
