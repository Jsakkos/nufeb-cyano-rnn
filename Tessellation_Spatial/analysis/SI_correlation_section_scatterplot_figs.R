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
poisson <- filtered %>% filter(Pattern %in% c("poisson"))
aggregate <- filtered %>% filter(Pattern %in% c("aggregate"))
regular <- filtered %>% filter(Pattern %in% c("regular"))



theme_update(axis.title = element_text(size=14),
             axis.text  = element_text(size=14),
             strip.text = element_text(size=11),
             axis.text.x = element_text(angle=0,size=12),
             axis.text.y = element_text(size=9),
             title = element_text(size=18,hjust=0.5))



filtered$Pattern = factor(filtered$Pattern, levels=c("aggregate","regular","poisson"))
pattern.labs <- c("Aggregate", "Random", "Regular")
names(pattern.labs) <- c("aggregate", "poisson", "regular")

edges.labs <- c("Non-Edge Colony", "Edge Colony")
names(edges.labs) <- c(FALSE, TRUE)

plot_correlation_scatter <- function(df,x_name,x_str){
  p <- ggplot(df,aes(x=!! enquo(x_name),
                           y=`Median-Normed Colony Area`,
                           color=factor(`intensity_bin`))) +
    geom_abline(slope=0,intercept=0,linetype="dashed",colour="grey")+
    geom_point(size=2,alpha=0.8)+
    xlab(x_str) +
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
  return(p)
}



plot_correlation_scatter(filtered %>% mutate(logperim = log(`Facet Perimeter (pixels)`)),
                         logperim, "lognorm of Facet Perimeter")
ggsave(here::here("output","SI_scatter_lognorm_facet_perim.png"),
        width=8,height=4,units="in",dpi=300)

plot_correlation_scatter(filtered,`Facet Sides`, "Number of Facet Sides")
ggsave(here::here("output","SI_scatter_facet_sides.png"),
       width=8,height=4,units="in",dpi=300)

plot_correlation_scatter(filtered %>% mutate(dist_m_um = `dist_m`*1e6),
                         `dist_m_um`, "Centroid-Seed Distance (microns)")
ggsave(here::here("output","SI_scatter_centroid_dist.png"),
       width=8,height=4,units="in",dpi=300)

plot_correlation_scatter(filtered %>% mutate(inv_AR = 1/`Facet Aspect Ratio`),
                         `inv_AR`, "1/Aspect Ratio")
ggsave(here::here("output","SI_scatter_AR.png"),
       width=8,height=4,units="in",dpi=300)

plot_correlation_scatter(filtered,`log_inv_sq_dist_m`, "Log Inverse Square Distance")
ggsave(here::here("output","SI_scatter_log_inverse_squar_distance.png"),
       width=8,height=4,units="in",dpi=300)

# ggsave(here::here("output","lognorm_facet_area.png"),
#        width=6,height=6,units="in",dpi=300)