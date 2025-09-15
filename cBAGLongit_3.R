# ===== Compare APOE4 encodings: Genotype (6-level), Dosage (0/1/2), Positivity (0/1) =====
suppressPackageStartupMessages({
  library(tidyverse)
  library(lmerTest)   # lmer + Type III via anova(..., type=3)
  library(emmeans)
  library(broom.mixed)
  library(stringr)
})

# --- Paths ---
data_path <- "/Users/alex/AlexBadea_MyGrants/BAG_R01_100325/code/data/HABS_metadata_AB_enriched_v2.csv"
out_dir   <- file.path(dirname(data_path), "lmer_results_apoe_models")
dir.create(out_dir, showWarnings = FALSE)

# --- Load ---
dat <- read.csv(data_path, check.names = FALSE)
names(dat) <- trimws(names(dat))

# --- Required columns ---
req <- c("SubjectID","cBAG_AB","CDX_Diabetes")
miss <- setdiff(req, names(dat))
if (length(miss)) stop("Missing columns: ", paste(miss, collapse=", "))

# --- APOE parser: standardize to 'e2/e2','e2/e3','e2/e4','e3/e3','e3/e4','e4/e4' ---
parse_apoe <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  s <- tolower(x)
  s <- gsub("ε", "e", s)
  s <- gsub("[^234e/]", "", s)              # keep only e, /, and 2/3/4
  # Extract digits; if none, try just digits from original
  digs <- str_extract_all(s, "[234]")
  out <- map_chr(digs, function(v) {
    if (length(v) < 2) return(NA_character_)
    v <- sort(v[1:2])                        # order alleles
    paste0("e", v[1], "/e", v[2])
  })
  out
}

# --- Prepare variables ---
dat <- dat %>%
  mutate(
    SubjectID    = factor(SubjectID),
    CDX_Diabetes = factor(as.integer(CDX_Diabetes), levels=c(0,1), labels=c("NoT2D","T2D"))
  )

# If you already have APOE4_Genotype, still clean it; else try to build from typical columns
if ("APOE4_Genotype" %in% names(dat)) {
  ap_raw <- dat$APOE4_Genotype
} else if ("APOE_Genotype" %in% names(dat)) {
  ap_raw <- dat$APOE_Genotype
} else {
  stop("Need APOE genotype column (e.g., 'APOE4_Genotype' or 'APOE_Genotype').")
}

dat$APOE4_Genotype <- factor(parse_apoe(ap_raw),
                             levels = c("e2/e2","e2/e3","e2/e4","e3/e3","e3/e4","e4/e4")
)

# Drop rows with unknown genotype for the modeling comparisons
dat <- dat %>% filter(!is.na(APOE4_Genotype))

# ε4 dosage and positivity
dosage_vec <- ifelse(grepl("e4", dat$APOE4_Genotype), str_count(as.character(dat$APOE4_Genotype), "e4"), 0)
dat$APOE4_dosage   <- as.integer(dosage_vec)                # 0,1,2
dat$APOE4_dosage_c <- dat$APOE4_dosage - mean(dat$APOE4_dosage, na.rm=TRUE)  # center
dat$APOE4_Positivity <- factor(ifelse(dat$APOE4_dosage > 0, 1, 0), levels=c(0,1), labels=c("APOE4-","APOE4+"))

# Optional: flag ε2 carriers (can adjust in sensitivity if desired)
dat$APOE2_carrier <- factor(ifelse(grepl("e2", dat$APOE4_Genotype), 1, 0), levels=c(0,1), labels=c("noE2","E2+"))

# --- Subject counts by Diabetes × Genotype (unique subjects) ---
subj_counts_geno <- dat %>%
  distinct(SubjectID, CDX_Diabetes, APOE4_Genotype) %>%
  count(CDX_Diabetes, APOE4_Genotype, name="n_subjects") %>%
  arrange(CDX_Diabetes, APOE4_Genotype)

cat("\nSubjects per group (Diabetes × APOE genotype):\n")
print(subj_counts_geno)
write.csv(subj_counts_geno, file.path(out_dir, "subject_counts_DxGenotype.csv"), row.names = FALSE)

# Quick warning for sparsity (e.g., cells < 5)
if (any(subj_counts_geno$n_subjects < 5)) {
  message("⚠️ Sparse genotype cells detected (<5 subjects). Consider using APOE4_dosage or APOE4_Positivity.")
}

# ---- Model frames ----
base_vars <- c("cBAG_AB","SubjectID","CDX_Diabetes")
dat_geno  <- dat %>% select(all_of(c(base_vars, "APOE4_Genotype"))) %>% na.omit()
dat_dose  <- dat %>% select(all_of(c(base_vars, "APOE4_dosage_c"))) %>% na.omit()
dat_pos   <- dat %>% select(all_of(c(base_vars, "APOE4_Positivity"))) %>% na.omit()

# ---- Type III setup ----
options(contrasts = c("contr.sum","contr.poly"))
ddf_method <- if (requireNamespace("pbkrtest", quietly=TRUE)) "Kenward-Roger" else "Satterthwaite"

# =================== (1) 6-level Genotype model ===================
# Reference level common in literature is e3/e3
dat_geno$APOE4_Genotype <- relevel(dat_geno$APOE4_Genotype, ref = "e3/e3")

fit_geno <- lmer(cBAG_AB ~ CDX_Diabetes * APOE4_Genotype + (1|SubjectID), data = dat_geno, REML = TRUE)

a3_geno <- anova(fit_geno, type = 3, ddf = ddf_method) %>% as.data.frame() %>% rownames_to_column("Effect")
write.csv(a3_geno, file.path(out_dir, "anova_typeIII_GENOTYPE.csv"), row.names = FALSE)

coef_geno <- broom.mixed::tidy(fit_geno, effects="fixed", conf.int=TRUE)
write.csv(coef_geno, file.path(out_dir, "fixed_effects_GENOTYPE.csv"), row.names = FALSE)

# EMMs: 2 × 6 table (marginal means)
emm_geno <- emmeans(fit_geno, ~ CDX_Diabetes * APOE4_Genotype)
write.csv(as.data.frame(emm_geno), file.path(out_dir, "emmeans_GENOTYPE.csv"), row.names = FALSE)

# Simple comparisons: Diabetes within each genotype; APOE genotype within each Diabetes
pairs_D_wG <- summary(pairs(emm_geno, by = "APOE4_Genotype"), infer = TRUE) %>% as.data.frame()
pairs_G_wD <- summary(pairs(emm_geno, by = "CDX_Diabetes"),     infer = TRUE) %>% as.data.frame()
write.csv(pairs_D_wG, file.path(out_dir, "pairs_Diabetes_within_Genotype.csv"), row.names = FALSE)
write.csv(pairs_G_wD, file.path(out_dir, "pairs_Genotype_within_Diabetes.csv"), row.names = FALSE)

# =================== (2) Dosage (0/1/2) model ===================
fit_dose <- lmer(cBAG_AB ~ CDX_Diabetes * APOE4_dosage_c + (1|SubjectID), data = dat_dose, REML = TRUE)

a3_dose <- anova(fit_dose, type = 3, ddf = ddf_method) %>% as.data.frame() %>% rownames_to_column("Effect")
write.csv(a3_dose, file.path(out_dir, "anova_typeIII_DOSAGE.csv"), row.names = FALSE)

coef_dose <- broom.mixed::tidy(fit_dose, effects="fixed", conf.int=TRUE)
write.csv(coef_dose, file.path(out_dir, "fixed_effects_DOSAGE.csv"), row.names = FALSE)

# emtrends: slope of cBAG vs dosage within Diabetes levels
tr_dose <- emtrends(fit_dose, ~ CDX_Diabetes, var = "APOE4_dosage_c")
write.csv(as.data.frame(summary(tr_dose, infer=TRUE)), file.path(out_dir, "emtrends_DOSAGE_by_Diabetes.csv"), row.names = FALSE)

# =================== (3) Binary positivity model ===================
fit_pos <- lmer(cBAG_AB ~ CDX_Diabetes * APOE4_Positivity + (1|SubjectID), data = dat_pos, REML = TRUE)

a3_pos <- anova(fit_pos, type = 3, ddf = ddf_method) %>% as.data.frame() %>% rownames_to_column("Effect")
write.csv(a3_pos, file.path(out_dir, "anova_typeIII_POSITIVITY.csv"), row.names = FALSE)

coef_pos <- broom.mixed::tidy(fit_pos, effects="fixed", conf.int=TRUE)
write.csv(coef_pos, file.path(out_dir, "fixed_effects_POSITIVITY.csv"), row.names = FALSE)

emm_pos <- emmeans(fit_pos, ~ CDX_Diabetes * APOE4_Positivity)
write.csv(as.data.frame(emm_pos), file.path(out_dir, "emmeans_POSITIVITY.csv"), row.names = FALSE)

# =================== Model fit comparison ===================
cmp <- tibble(
  Model = c("Genotype(6L)","Dosage(0/1/2)","Positivity(0/1)"),
  n     = c(nrow(dat_geno), nrow(dat_dose), nrow(dat_pos)),
  AIC   = c(AIC(fit_geno), AIC(fit_dose), AIC(fit_pos)),
  BIC   = c(BIC(fit_geno), BIC(fit_dose), BIC(fit_pos))
)
print(cmp)
write.csv(cmp, file.path(out_dir, "model_fit_comparison.csv"), row.names = FALSE)

cat("✅ Wrote outputs to:\n", out_dir, "\n")



