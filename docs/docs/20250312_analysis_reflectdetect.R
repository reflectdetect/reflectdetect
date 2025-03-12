# =====================================================================
# Spectral Data Analysis in R: Comparing Spectral Shapes and Differences
# 
# This R script analyzes spectral data to assess differences in both:
# 1. Absolute band-wise reflectance values across three data sources:
#    apriltag, geolocation, and sELM calibration.
# 2. Spectral shapes across the data sources using cosine distance.
# 
# The input data consists of spectral measurements extracted from three 
# distinct, overlapping orthophotos. These orthophotos underwent separate 
# calibration procedures and were resampled to the same ground sampling 
# distance (GSD) for consistent comparison. The analysis aims to evaluate 
# both the absolute reflectance values and the similarity of spectral shapes 
# between the different data sources.
# 
# Methods used:
# 1. **Cosine Distance**: Focuses on comparing the angular difference 
#    between spectra, useful for assessing the similarity in spectral 
#    shapes regardless of absolute values.
# 
# Key analyses:
# - Perform ANOVA and Tukey tests to assess significant differences in 
#   spectral values across different data sources.
# - Calculate pairwise cosine distances between spectral 
#   shapes to evaluate how similar or different the spectra are.
# 
# Data source (calibration types) names:
# - AprilTag
# - Geolocation
# - sELM
# 
# This script is based on the paper:
# 
# Francis, L., Geissler, L., Okole, N., Stachniss, C., Gipp, B., Heim, R. HJ. (2025). ReflectDetect: A software tool for AprilTag-Guided In-Flight Radiometric Calibration for UAV Optical Data. (in review) SoftwareX.
# 
# To cite this paper, please use the following reference:
# 
# @article{Francis2025,
#   author = {Francis, L. and Geissler, L. and Okole, N. and Stachniss, C. and Gipp, B. and Heim, R. HJ.},
#   title = {ReflectDetect: A software tool for AprilTag-Guided In-Flight Radiometric Calibration for UAV Optical Data},
#   journal = {SoftwareX},
#   year = {2025},
#   volume = {},
#   number = {},
#   pages = {},
#   doi = {}
# }
# 
# =====================================================================

# Packages ----

library(sf)
library(ggplot2)
library(tidyr)
library(dplyr)
library(knitr)
library(tidyverse)
library(emmeans)
library(patchwork)

# Functions ----

# Function to compute angular distance between two vectors
angular_distance <- function(A, B) {
  # Compute cosine similarity
  cosine_similarity <- sum(A * B) / (sqrt(sum(A^2)) * sqrt(sum(B^2)))
  
  # Convert cosine similarity to angular distance (in radians)
  angular_distance <- acos(cosine_similarity)
  
  return(angular_distance)
}

# Function to compute pairwise angular distances
compute_angular_distances <- function(data) {
  num_rows <- nrow(data)
  angular_distances <- matrix(0, nrow = num_rows, ncol = num_rows)
  
  # Loop over each pair of rows (spectra)
  for (i in 1:num_rows) {
    for (j in i:num_rows) {
      if (i != j) {
        angular_distances[i, j] <- angular_distance(data[i, ], data[j, ])
        angular_distances[j, i] <- angular_distances[i, j]  # Since the distance is symmetric
      }
    }
  }
  
  return(angular_distances)
}

# Data ----

# Load data
sampling_points <- "add path here/20250310_sample_data.gpkg"
output <-  "add path here"

# Preprocess data
gdf <- st_read(sampling_points)
gdf <- gdf %>% select(-id)
colnames(gdf) <- c("apriltag_b", "apriltag_g", "apriltag_r", "apriltag_re", "apriltag_nir",
                   "geolocation_b", "geolocation_g", "geolocation_r", 
                   "geolocation_re", "geolocation_nir",
                   "sELM_b", "sELM_g", "sELM_r", "sELM_re", "sELM_nir",
                   "geom")

# Reshape data to long format
gdf_long <- gdf %>%
  pivot_longer(cols = -geom, names_to = c("data_source", "band"), names_sep = "_") %>%
  rename(spectral_value = value)

# Aggregate data by calculating mean and standard deviation
gdf_agg <- gdf_long %>%
  group_by(data_source, band) %>%
  summarise(
    mean_spectral_value = mean(spectral_value, na.rm = TRUE),
    sd_spectral_value = sd(spectral_value, na.rm = TRUE),
    se_spectral_value = sd_spectral_value / sqrt(n()),
    .groups = "drop"
  )

# Aggregate wide for spectral distance metrics

gdf_wide <- gdf_long %>%
  pivot_wider(names_from = band, values_from = spectral_value)

gdf_wide$geom <- NULL  # Remove a single column


# Summarize the data by calculating the mean spectral value for each band and group
gdf_mean_spectra <- gdf_wide %>%
  group_by(data_source) %>%
  summarise(
    across(c(b, g, r, re, nir), ~ mean(., na.rm = TRUE))
  )

# Visualizations ----

# Spectral reflectance signatures 

p <- ggplot(gdf_agg, aes(x = band, y = mean_spectral_value, color = data_source, group = data_source)) +
  geom_line(size = 1) +  # Increase line thickness for visibility
  geom_ribbon(
    aes(ymin = mean_spectral_value - se_spectral_value,
        ymax = mean_spectral_value + se_spectral_value),
    alpha = 0.2  # Set the transparency of the ribbon
  ) +
  theme_minimal(base_size = 16) +  # Increase base font size
  scale_color_brewer(palette = "Pastel1", labels = c("AprilTag", "Geolocation", "sELM")) +  # Use a pastel color palette and change the labels
  labs(
    title = "",
    x = "Spectral Band",
    y = "Reflectance",
    color = "Calibration Type"  # Change the legend title
  ) +
  scale_x_discrete(limits = c("b", "g", "r", "re", "nir"), labels = c("B", "G", "R", "RE", "NIR")) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels
    axis.title = element_text(size = 16),  # Adjust axis title size
    axis.text = element_text(size = 16),  # Adjust axis text size
    legend.title = element_text(size = 16),  # Adjust legend title size
    legend.text = element_text(size = 16),  # Adjust legend text size
    plot.title = element_text(hjust = 0.5, size = 20)  # Center and adjust plot title size
  )
p

# Save the figure at 600 dpi
#ggsave(file.path(output, "20250312_spectral_signatures.png"), plot = p, 
#       units = "cm", width = 30, height = 15, dpi = 600)


# Band-wise box plots

p2 <- ggplot(gdf_long, aes(x = factor(band, levels = c("b", "g", "r", "re", "nir")), y = spectral_value, fill = data_source)) +
  geom_boxplot(key_glyph = "rect") +
  labs(
    title = "",
    x = "Spectral Band",
    y = "Reflectance"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels
    axis.title = element_text(size = 16),  # Adjust axis title size
    axis.text = element_text(size = 16),  # Adjust axis text size
    legend.title = element_text(size = 14),  # Adjust legend title size
    legend.text = element_text(size = 16),  # Adjust legend text size
    plot.title = element_text(hjust = 0.5, size = 20)  # Center and adjust plot title size
  )+
  scale_fill_brewer(palette = "Pastel1")+
  guides(fill=guide_legend(title="Calibration Type"))

p2

# Save the figure at 600 dpi
#ggsave(file.path(output, "20250306_combine_spectra_box.png"), plot= plot,
#       units = "cm", width = 30, height = 15, dpi = 600)


# Statistical Modeling ----

if ("geom" %in% names(gdf)) {
  gdf$geom <- NULL
}


# Reshape data to long format
data_long <- gdf %>%
  pivot_longer(cols = everything(),
               names_to = c("source", "band"),
               names_sep = "_",
               values_to = "value")


# Convert to factors
data_long$source <- as.factor(data_long$source)
data_long$band <- as.factor(data_long$band)


# Fit anova
anova_model <- aov(value ~ source*band, data = data_long)
summary(anova_model)

# Sliced ANOVA to get accurate results
sliced_anova <- joint_tests(anova_model, by = "band")
summary(sliced_anova)

posthoc_results <- pairs(emmeans(anova_model, ~ source | band), adjust = "Bonferroni")
posthoc_results

plot(posthoc_results, comparison = TRUE) +theme_bw()

# Spectral Distances ----

# Compute angular distances
spectra_matrix <- gdf_mean_spectra %>% select(b, g, r, re, nir) %>% as.matrix()
angular_distances <- compute_angular_distances(spectra_matrix)

# Data sources (replace these with the actual sources in your dataset)
data_sources <- c("apriltag", "geolocation", "sELM")

# Assign the data sources as row and column names
rownames(angular_distances) <- data_sources
colnames(angular_distances) <- data_sources

# Print the matrix with labels
kable(angular_distances)


