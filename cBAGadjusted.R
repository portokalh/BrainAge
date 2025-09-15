# ===== lmer: cBAG_AB ~ HBA1c * Age * APOE4_dosage * Sex  =====================
suppressPackageStartupMessages({
  library(tidyverse)
  library(lmerTest)      # lmer + Type III (Satterthwaite / KR)
  library(emmeans)
  library(broom.mixed)
  library(stringr)
})

# --- Paths ---
data_path <- "/Users/alex/AlexBadea_MyGrants/BAG_R01_100325/code/data/HABS_metadata_AB_enriched_v2.csv"
out_dir   <- file.path(dirname(data_path), "lmer_results_hba1c_age_dosage")
dir.create(out_dir, showWarnings = FALSE)

# --- Load & tidy ---
dat <- read.csv(data_path, check.names = FALSE)
names(dat) <- trimws(names(dat))
stopifnot(all(c("SubjectID","cBAG_AB","Sex") %in% names(dat)),
          "BW_HBA1c" %in% names(dat))

# Pick per-visit Age column
age_candidates <- intersect(c("Age_at_scan","Age","AgeYears","Age_at_visit","AGE"), names(dat))
if (!length(age_candidates)) stop("No per-visit Age column found.")
age_var <- age_candidates[1]

# --- APOE genotype -> dosage (0/1/2 copies of e4) ---
parse_apoe <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  s <- tolower(x); s <- gsub("ε","e",s); s <- gsub("[^234e/]","",s)
  digs <- str_extract_all(s, "[234]")
  vapply(digs, function(v){
    if (length(v) < 2) return(NA_character_)
    v <- sort(v[1:2]); paste0("e", v[1], "/e", v[2])
  }, character(1))
}
cand <- intersect(c("APOE4_Genotype","APOE_Genotype"), names(dat))
if (!length(cand)) stop("Need 'APOE4_Genotype' or 'APOE_Genotype'.")

geno_std <- factor(parse_apoe(dat[[cand[1]]]),
                   levels = c("e2/e2","e2/e3","e2/e4","e3/e3","e3/e4","e4/e4"))
dosage <- ifelse(is.na(geno_std), NA_integer_,
                 stringr::str_count(as.character(geno_std), "e4"))  # 0,1,2

# --- Build model frame & center continuous predictors (AFTER NA filtering) ---
datm <- tibble(
  SubjectID      = factor(dat$SubjectID),
  cBAG_AB        = dat$cBAG_AB,
  BW_HBA1c       = as.numeric(dat$BW_HBA1c),
  Age            = as.numeric(dat[[age_var]]),
  APOE4_dosage   = as.integer(dosage),
  Sex            = factor(dat$Sex)
) %>% na.omit()

datm <- datm %>%
  mutate(
    BW_HBA1c_c       = BW_HBA1c - mean(BW_HBA1c, na.rm = TRUE),
    Age_c            = Age      - mean(Age,      na.rm = TRUE),
    APOE4_dosage_c   = APOE4_dosage - mean(APOE4_dosage, na.rm = TRUE)
  )

# Save subject counts by dosage × sex (unique subjects)
subj_counts <- datm %>%
  distinct(SubjectID, APOE4_dosage, Sex) %>%
  count(APOE4_dosage, Sex, name="n_subjects") %>%
  arrange(APOE4_dosage, Sex)
write.csv(subj_counts, file.path(out_dir, "subject_counts_dosage_by_sex.csv"), row.names = FALSE)

# --- Type III setup ---
options(contrasts = c("contr.sum","contr.poly"))
ddf_method <- if (requireNamespace("pbkrtest", quietly = TRUE)) "Kenward-Roger" else "Satterthwaite"

# --- Fit model (random slope for Age; simplify to (1|SubjectID) if singular) ---
fit <- lmer(
  cBAG_AB ~ BW_HBA1c_c * Age_c * APOE4_dosage_c + Sex +
    (1 | SubjectID),
  data = datm, REML = TRUE
)

# --- Type III ANOVA & fixed effects ---
a3 <- anova(fit, type = 3, ddf = ddf_method) %>% as.data.frame() %>% tibble::rownames_to_column("Effect")
write.csv(a3, file.path(out_dir, "anova_typeIII_4way_DOSAGE.csv"), row.names = FALSE)

fx <- broom.mixed::tidy(fit, effects="fixed", conf.int=TRUE)
write.csv(fx, file.path(out_dir, "fixed_effects_4way_DOSAGE.csv"), row.names = FALSE)

# --- Post-hoc summaries that are easy to read --------------------------------
# Evaluate at representative dosage levels 0/1/2 (converted to centered scale)
dbar <- mean(datm$APOE4_dosage, na.rm = TRUE)
dos_c_levels <- c(0,1,2) - dbar
age_sd <- sd(datm$Age_c, na.rm = TRUE)

# 1) HbA1c slope at mean Age (Age_c=0) within Sex × dosage levels
tr_hba_age0 <- emtrends(fit, ~ Sex * APOE4_dosage_c, var = "BW_HBA1c_c",
                        at = list(Age_c = 0, APOE4_dosage_c = dos_c_levels))
write.csv(as.data.frame(summary(tr_hba_age0, infer = TRUE)),
          file.path(out_dir, "emtrends_HBA1c_by_Sex_DOSAGE_atMeanAge.csv"), row.names = FALSE)

# 2) HbA1c slope at Age mean ±1 SD (to see age dependence)
tr_hba_ageslices <- emtrends(fit, ~ Sex * APOE4_dosage_c, var = "BW_HBA1c_c",
                             at = list(Age_c = c(0, age_sd, -age_sd),
                                       APOE4_dosage_c = dos_c_levels))
write.csv(as.data.frame(summary(tr_hba_ageslices, infer = TRUE)),
          file.path(out_dir, "emtrends_HBA1c_by_Sex_DOSAGE_ageSlices.csv"), row.names = FALSE)

# 3) Age slope at mean HbA1c (BW_HBA1c_c=0) within Sex × dosage levels
tr_age_hb0 <- emtrends(fit, ~ Sex * APOE4_dosage_c, var = "Age_c",
                       at = list(BW_HBA1c_c = 0, APOE4_dosage_c = dos_c_levels))
write.csv(as.data.frame(summary(tr_age_hb0, infer = TRUE)),
          file.path(out_dir, "emtrends_Age_by_Sex_DOSAGE_atMeanHBA1c.csv"), row.names = FALSE)

# 4) Adjusted means at reference slice (Age=mean, HbA1c=mean)
emm_slice <- emmeans(fit, ~ Sex * APOE4_dosage_c,
                     at = list(Age_c = 0, BW_HBA1c_c = 0, APOE4_dosage_c = dos_c_levels))
write.csv(as.data.frame(emm_slice),
          file.path(out_dir, "emmeans_by_Sex_DOSAGE_atReferenceSlice.csv"), row.names = FALSE)

cat("✅ Saved outputs to:\n", out_dir, "\n")

# --- Quick plot: adjusted means by Sex × ε4 dosage at Age=mean & HbA1c=mean ---
suppressPackageStartupMessages({ library(ggplot2) })

# we already computed these earlier:
#   dbar  <- mean(datm$APOE4_dosage, na.rm = TRUE)
#   dos_c_levels <- c(0,1,2) - dbar

emm_slice <- emmeans(
  fit, ~ Sex * APOE4_dosage_c,
  at = list(Age_c = 0, BW_HBA1c_c = 0, APOE4_dosage_c = dos_c_levels)
)

emm_df <- as.data.frame(emm_slice)

# map centered dosage back to the labeled 0/1/2 levels for a clean x-axis
key <- data.frame(APOE4_dosage_c = dos_c_levels, dosage = factor(c(0,1,2), levels = c(0,1,2)))
emm_df <- merge(emm_df, key, by = "APOE4_dosage_c", all.x = TRUE)

pd <- position_dodge(width = 0.35)

p <- ggplot(emm_df, aes(x = dosage, y = emmean, group = Sex, shape = Sex, linetype = Sex)) +
  geom_point(size = 3, position = pd) +
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), width = 0.12, position = pd) +
  geom_line(position = pd) +
  labs(
    x = "APOE ε4 dosage (copies)",
    y = "Adjusted cBAG_AB\n(Age = mean, HbA1c = mean)",
    title = "Adjusted means by Sex × APOE ε4 dosage"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(out_dir, "emmeans_bySex_byDosage_atMeanAgeHbA1c.png"),
       p, width = 7, height = 5, dpi = 300)

# (optional) print to screen
print(p)


# --- Pairwise p-values: Sex difference at each dosage (Age=mean, HbA1c=mean) ---
# Uses the same emm_slice and emm_df from the previous block

# 1) Pairwise Sex comparison within each dosage level
pairs_sex_by_dos <- summary(pairs(emm_slice, by = "APOE4_dosage_c"), infer = TRUE) %>% 
  as.data.frame()

# Map centered dosage back to labeled 0/1/2 for joining/plotting
key <- data.frame(APOE4_dosage_c = dos_c_levels, dosage = factor(c(0,1,2), levels = c(0,1,2)))
pairs_sex_by_dos <- dplyr::left_join(pairs_sex_by_dos, key, by = "APOE4_dosage_c")

# 2) Compute a y-position a bit above the highest CI per dosage
ypos <- emm_df %>%
  dplyr::group_by(dosage) %>%
  dplyr::summarise(y_top = max(upper.CL, na.rm = TRUE), .groups = "drop")

yrange <- diff(range(emm_df$emmean, na.rm = TRUE))
annot <- dplyr::left_join(pairs_sex_by_dos, ypos, by = "dosage") %>%
  dplyr::mutate(
    label = sprintf("p = %.3g", p.value),
    y_lab = y_top + 0.06 * (ifelse(is.finite(yrange) && yrange > 0, yrange, 1))
  )

# 3) Add p-value text to the existing plot `p` and save a new PNG
p_annot <- p + 
  geom_text(data = annot, aes(x = dosage, y = y_lab, label = label),
            inherit.aes = FALSE, size = 3.6, vjust = 0)

ggsave(file.path(out_dir, "emmeans_bySex_byDosage_atMeanAgeHbA1c_withP.png"),
       p_annot, width = 7, height = 5, dpi = 300)

# 4) Also save the pairwise table
readr::write_csv(pairs_sex_by_dos, file.path(out_dir, "pairs_Sex_within_DOSAGE_atMeanAgeHbA1c.csv"))




# --- (Optional) Dosage comparisons within each Sex ---
pairs_dos_within_sex <- summary(pairs(emm_slice, by = "Sex"), infer = TRUE) %>% as.data.frame()
readr::write_csv(pairs_dos_within_sex, file.path(out_dir, "pairs_DOSAGE_within_Sex_atMeanAgeHbA1c.csv"))

# For a quick visual, annotate only 0 vs 2 per sex to keep it readable:
p02 <- pairs_dos_within_sex %>%
  dplyr::filter(grepl("0 - 2|2 - 0", contrast)) %>%
  dplyr::mutate(label = sprintf("0 vs 2: p = %.3g", p.value))

# place labels slightly above each sex's max CI across dosages
ypos_sex <- emm_df %>%
  dplyr::group_by(Sex) %>%
  dplyr::summarise(y_top = max(upper.CL, na.rm = TRUE), .groups = "drop") %>%
  dplyr::mutate(y_lab = y_top + 0.08 * (ifelse(is.finite(yrange) && yrange > 0, yrange, 1)))

annot02 <- dplyr::left_join(p02, ypos_sex, by = "Sex")

p_annot02 <- p + 
  geom_text(data = annot02, aes(x = 2, y = y_lab, label = label, group = Sex),  # x=2 near the "2 copies" group
            position = position_nudge(x = 0.1), size = 3.4, vjust = 0, inherit.aes = FALSE)

ggsave(file.path(out_dir, "emmeans_bySex_byDosage_atMeanAgeHbA1c_withP_0v2.png"),
       p_annot02, width = 7, height = 5, dpi = 300)


#counting
library(dplyr)
library(stringr)

# Parser (re-define here so it's available)
parse_apoe <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  s <- tolower(x); s <- gsub("ε","e",s); s <- gsub("[^234e/]","",s)
  digs <- stringr::str_extract_all(s, "[234]")
  vapply(digs, function(v){
    if (length(v) < 2) return(NA_character_)
    v <- sort(v[1:2]); paste0("e", v[1], "/e", v[2])
  }, character(1))
}

# Pick the genotype column name once, then use .data[[...]] inside mutate()
geno_col <- intersect(c("APOE4_Genotype","APOE_Genotype"), names(dat))[1]
stopifnot(length(geno_col) == 1)

tab <- dat %>% 
  mutate(
    geno_std = factor(
      parse_apoe(.data[[geno_col]]),
      levels = c("e2/e2","e2/e3","e2/e4","e3/e3","e3/e4","e4/e4")
    ),
    APOE4_dosage = case_when(
      geno_std %in% c("e2/e2","e2/e3","e3/e3") ~ 0L,
      geno_std %in% c("e2/e4","e3/e4")         ~ 1L,
      geno_std == "e4/e4"                      ~ 2L,
      TRUE                                     ~ NA_integer_
    )
  )

# Mapping sanity-check
tab %>% count(geno_std, APOE4_dosage) %>% arrange(geno_std)

# Overall dosage counts
tab %>% count(APOE4_dosage)

# Unique-subject counts by Sex × dosage
tab %>% distinct(SubjectID, APOE4_dosage, Sex) %>%
  count(APOE4_dosage, Sex, name = "n_subjects") %>%
  arrange(APOE4_dosage, Sex)

# Centering used in the model/plots
dbar <- if (exists("datm")) mean(datm$APOE4_dosage, na.rm = TRUE) else mean(tab$APOE4_dosage, na.rm = TRUE)
dos_c_levels <- c(0, 1, 2) - dbar
dbar; dos_c_levels

# Inspect any unparsed/raw genotype strings
bad <- is.na(tab$APOE4_dosage) & !is.na(tab[[geno_col]])
unique(tab[[geno_col]][bad])


