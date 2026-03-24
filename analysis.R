# Ali Çağan Kaya, 2026
#
#
# The main analysis script for 
# "Evaluating the Accuracy of Speech to Text Technologies in Turkish-Accented English"
# presented at the 19th Student Conference of Linguistics at İstanbul University, Türkiye, 3 April 2026
# by Nuran Orhan & Ali Çağan Kaya
#
# - Per speaker WER binomial denominators from the fixed reference
# - Fits frequentist (lme4) and Bayesian (brms) mixed models
# - Demographic analyses based on time studied and time residency
# - Saves tidy outputs and plots
# - Visualisations (per-speaker WER CI plot, change-type proportions, token-position heatmap, top deletions/insertions bars)

# -----------------------------
# 0) Setup
# -----------------------------
required_pkgs <- c(
  "tidyverse", "readr", "janitor", "lme4", "lmerTest",
  "broom.mixed", "scales", "glue", "readxl"
)
to_install <- setdiff(required_pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, Ncpus = 2)

# brms
if (!"brms" %in% rownames(installed.packages())) install.packages("brms", Ncpus = 2)

library(tidyverse)
library(readr)
library(janitor)
library(lme4)
library(lmerTest)
library(broom.mixed)
library(scales)
library(glue)
library(readxl)
library(brms)

set.seed(42)

# -----------------------------
# 1) Paths & input
# -----------------------------
metrics_dir <- "/home/alicagank/Work/Academic/Collaborations/orhan-kaya/"
in_changes  <- file.path(metrics_dir, "all_word_changes.csv")
out_dir     <- file.path(metrics_dir, "r_outputs_only_all_changes")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

if (!file.exists(in_changes)) {
  stop(glue("File not found))
}

all_changes <- read_csv(in_changes, show_col_types = FALSE) %>% clean_names()
# Expected columns:
# speaker, change_type, sub_ref, sub_hyp, del_token, ins_token,
# ref_token_index, hyp_token_index, ref_sentence_index, hyp_sentence_index,
# ref_prev, ref_next, hyp_prev, hyp_next

# -----------------------------
# 2) Reference string & normalization
# -----------------------------
reference_text <- paste0(
  "Please call Stella. Ask her to bring these things with her from the store: ",
  "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack ",
  "for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. ",
  "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
)

normalize_text <- function(s) {
  s <- tolower(s)
  s <- gsub("[^\\w\\s]", " ", s, perl = TRUE)  # remove punctuation
  s <- gsub("\\s+", " ", s, perl = TRUE)
  trimws(s)
}

ref_tokens <- strsplit(normalize_text(reference_text), "\\s+")[[1]]
ref_len_words <- length(ref_tokens)

write_lines(glue("Reference word count used as binomial denominator: {ref_len_words}"),
            file.path(out_dir, "reference_info.txt"))

# -----------------------------
# 3) Aggregate per speaker
# -----------------------------
agg_by_speaker <- all_changes %>%
  mutate(
    is_sub = change_type == "substitution",
    is_del = change_type == "deletion",
    is_ins = change_type == "insertion"
  ) %>%
  group_by(speaker) %>%
  summarise(
    subs = sum(is_sub, na.rm = TRUE),
    dels = sum(is_del, na.rm = TRUE),
    ins  = sum(is_ins, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    # Binomial "WER" is subs + dels
    errors_on_ref = subs + dels,
    trials_ref    = ref_len_words,             # same passage for all speakers
    correct       = pmax(trials_ref - errors_on_ref, 0),
    wer           = ifelse(trials_ref > 0, errors_on_ref / trials_ref, NA_real_),
    ins_rate      = ifelse(trials_ref > 0, ins / trials_ref, NA_real_)
  )

write_csv(agg_by_speaker, file.path(out_dir, "per_speaker_aggregates.csv"))

# -----------------------------
# 4) Descriptives & global plots
# -----------------------------
desc_overall <- agg_by_speaker %>%
  summarise(
    n_speakers = n(),
    wer_mean = mean(wer, na.rm = TRUE),
    wer_sd   = sd(wer, na.rm = TRUE),
    wer_min  = min(wer, na.rm = TRUE),
    wer_max  = max(wer, na.rm = TRUE),
    ins_rate_mean = mean(ins_rate, na.rm = TRUE),
    ins_rate_sd   = sd(ins_rate, na.rm = TRUE)
  )
write_csv(desc_overall, file.path(out_dir, "overall_descriptives.csv"))

# Histogram of WER
p_hist <- ggplot(agg_by_speaker, aes(x = wer)) +
  geom_histogram(bins = 20) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "WER across speakers (from subs + dels)", x = "WER", y = "Count")
ggsave(file.path(out_dir, "wer_histogram.png"), p_hist, width = 7, height = 4.2, dpi = 160)

# Histogram of insertion rate
p_ins <- ggplot(agg_by_speaker, aes(x = ins_rate)) +
  geom_histogram(bins = 20) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "Insertion rate across speakers (insertions / ref words)", x = "Insertion rate", y = "Count")
ggsave(file.path(out_dir, "insertion_rate_histogram.png"), p_ins, width = 7, height = 4.2, dpi = 160)

# Per-speaker WER with ~95% normal CIs (quick viz)
agg_by_speaker_ci <- agg_by_speaker %>%
  arrange(desc(wer)) %>%
  mutate(
    se = sqrt(pmax(wer * (1 - wer) / trials_ref, 0)),
    ci_lo = pmax(wer - 1.96 * se, 0),
    ci_hi = pmin(wer + 1.96 * se, 1),
    speaker_f = factor(speaker, levels = rev(speaker))
  )

p_wer_ci <- ggplot(agg_by_speaker_ci, aes(x = speaker_f, y = wer)) +
  geom_pointrange(aes(ymin = ci_lo, ymax = ci_hi)) +
  coord_flip() +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "Per-speaker WER with ~95% CIs",
       x = "Speaker", y = "WER")
ggsave(file.path(out_dir, "wer_per_speaker_with_ci.png"), p_wer_ci, width = 7.5, height = max(4.5, 0.35*nrow(agg_by_speaker_ci)), dpi = 160)

p_change_mix <- all_changes %>%
  count(speaker, change_type, name = "n") %>%
  group_by(speaker) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup() %>%
  mutate(speaker_f = factor(speaker, levels = unique(agg_by_speaker_ci$speaker))) %>%
  ggplot(aes(x = speaker_f, y = prop, fill = change_type)) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "Mix of change types by speaker", x = "Speaker", y = "Proportion", fill = "Change type")
ggsave(file.path(out_dir, "change_type_mix_by_speaker.png"), p_change_mix, width = 7.8, height = max(4.8, 0.35*nrow(agg_by_speaker_ci)), dpi = 160)

# -----------------------------
# 5) Frequentist binomial GLMM (lme4)
# -----------------------------
if (nrow(agg_by_speaker) >= 2) {
  glmm_m0 <- glmer(
    cbind(errors_on_ref, correct) ~ 1 + (1 | speaker),
    data = agg_by_speaker,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
  )
  capture.output(summary(glmm_m0), file = file.path(out_dir, "glmm_m0_summary.txt"))
  broom.mixed::tidy(glmm_m0, effects = "fixed", conf.int = TRUE) %>%
    write_csv(file.path(out_dir, "glmm_m0_fixed.csv"))
  broom.mixed::tidy(glmm_m0, effects = "ran_pars", conf.int = TRUE) %>%
    write_csv(file.path(out_dir, "glmm_m0_random.csv"))
  
  #Caterpillar plot of GLMM random intercepts
  re_spk <- ranef(glmm_m0, condVar = TRUE)$speaker
  re_df <- tibble(
    speaker = rownames(re_spk),
    re = re_spk[,1],
    se = sqrt(attr(ranef(glmm_m0, condVar = TRUE)[["speaker"]], "postVar")[1,1,])
  ) %>%
    mutate(lo = re - 1.96*se, hi = re + 1.96*se,
           speaker_f = fct_reorder(speaker, re))
  
  p_cater_glmm <- ggplot(re_df, aes(x = speaker_f, y = re)) +
    geom_pointrange(aes(ymin = lo, ymax = hi)) +
    coord_flip() +
    labs(title = "GLMM random intercepts (speaker)", x = "Speaker", y = "RE (logit scale)")
  ggsave(file.path(out_dir, "glmm_random_intercepts_speaker.png"), p_cater_glmm,
         width = 7.5, height = max(4.5, 0.35*nrow(re_df)), dpi = 160)
}

# -----------------------------
# 6) Bayesian binomial mixed model (brms)
# -----------------------------
if (nrow(agg_by_speaker) >= 2) {
  bayes_df <- agg_by_speaker %>%
    mutate(y = as.integer(errors_on_ref), trials = as.integer(trials_ref))
  
  bayes_priors <- c(
    prior(student_t(3, 0, 2.5), class = "Intercept"),
    prior(student_t(3, 0, 2.5), class = "sd", group = "speaker")
  )
  
  bfit_m0 <- brm(
    formula = y | trials(trials) ~ 1 + (1 | speaker),
    data = bayes_df,
    family = binomial(link = "logit"),
    prior = bayes_priors,
    iter = 2000, warmup = 1000, chains = 4, cores = max(1, parallel::detectCores() - 1),
    seed = 42,
    control = list(adapt_delta = 0.95)
  )
  
  sink(file.path(out_dir, "brms_m0_summary.txt")); print(summary(bfit_m0)); sink()
  saveRDS(bfit_m0, file.path(out_dir, "brms_m0_fit.rds"))
  
  inv_logit <- function(x) 1 / (1 + exp(-x))
  draws <- as_draws_df(bfit_m0)
  if ("b_Intercept" %in% names(draws)) {
    rate_draws <- inv_logit(draws$b_Intercept)
    tibble(
      mean = mean(rate_draws),
      sd   = sd(rate_draws),
      q025 = quantile(rate_draws, 0.025),
      q975 = quantile(rate_draws, 0.975)
    ) %>% write_csv(file.path(out_dir, "brms_m0_error_rate_summary.csv"))
    
    p_post <- tibble(rate = rate_draws) %>%
      ggplot(aes(rate)) + geom_density() +
      scale_x_continuous(labels = percent_format(accuracy = 0.1)) +
      labs(title = "Posterior: global word error rate", x = "Error rate", y = "Density")
    ggsave(file.path(out_dir, "brms_m0_error_rate_density.png"), p_post, width = 6.2, height = 4, dpi = 160)
  }
}

# -----------------------------
# 7) Separate insertion model
# -----------------------------
if (nrow(agg_by_speaker) >= 2) {
  glmm_ins <- glmer(
    cbind(ins, trials_ref - ins) ~ 1 + (1 | speaker),
    data = agg_by_speaker,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
  )
  capture.output(summary(glmm_ins), file = file.path(out_dir, "glmm_insertions_summary.txt"))
  
  # Bayesian
  bayes_priors_ins <- c(
    prior(student_t(3, 0, 2.5), class = "Intercept"),
    prior(student_t(3, 0, 2.5), class = "sd", group = "speaker")
  )
  bfit_ins <- brm(
    formula = ins | trials(trials_ref) ~ 1 + (1 | speaker),
    data = agg_by_speaker,
    family = binomial(link = "logit"),
    prior = bayes_priors_ins,
    iter = 2000, warmup = 1000, chains = 4, cores = max(1, parallel::detectCores() - 1),
    seed = 43, control = list(adapt_delta = 0.95)
  )
  sink(file.path(out_dir, "brms_insertions_summary.txt")); print(summary(bfit_ins)); sink()
  saveRDS(bfit_ins, file.path(out_dir, "brms_insertions_fit.rds"))
}

# -----------------------------
# 8) Top change patterns (from all_word_changes.csv)
# -----------------------------
top_subs <- all_changes %>%
  filter(change_type == "substitution") %>%
  count(sub_ref, sub_hyp, name = "count") %>%
  arrange(desc(count)) %>% slice_head(n = 25)

top_dels <- all_changes %>%
  filter(change_type == "deletion") %>%
  count(del_token, name = "count") %>%
  arrange(desc(count)) %>% slice_head(n = 25)

top_ins  <- all_changes %>%
  filter(change_type == "insertion") %>%
  count(ins_token, name = "count") %>%
  arrange(desc(count)) %>% slice_head(n = 25)

write_csv(top_subs, file.path(out_dir, "top25_substitutions.csv"))
write_csv(top_dels, file.path(out_dir, "top25_deletions.csv"))
write_csv(top_ins,  file.path(out_dir, "top25_insertions.csv"))

# Bars for deletions & insertions
if (nrow(top_dels) > 0) {
  p_dels <- top_dels %>%
    mutate(tok = fct_reorder(del_token, count)) %>%
    ggplot(aes(x = tok, y = count)) +
    geom_col() + coord_flip() +
    labs(title = "Top 25 deletions (global)", x = NULL, y = "Count")
  ggsave(file.path(out_dir, "top_deletions.png"), p_dels, width = 7, height = 6.5, dpi = 160)
}
if (nrow(top_ins) > 0) {
  p_inss <- top_ins %>%
    mutate(tok = fct_reorder(ins_token, count)) %>%
    ggplot(aes(x = tok, y = count)) +
    geom_col() + coord_flip() +
    labs(title = "Top 25 insertions (global)", x = NULL, y = "Count")
  ggsave(file.path(out_dir, "top_insertions.png"), p_inss, width = 7, height = 6.5, dpi = 160)
}

# Substitutions plot
if (nrow(top_subs) > 0) {
  p_subs <- top_subs %>%
    mutate(pair = glue("{sub_ref} \u2192 {sub_hyp}"),
           pair = fct_reorder(pair, count)) %>%
    ggplot(aes(x = pair, y = count)) +
    geom_col() + coord_flip() +
    labs(title = "Top 25 substitutions (global)", x = NULL, y = "Count")
  ggsave(file.path(out_dir, "top_substitutions.png"), p_subs, width = 7, height = 6.5, dpi = 160)
}

# -----------------------------
# 9) Heatmap of change density over passage position
# -----------------------------
pos_df <- all_changes %>%
  filter(!is.na(ref_token_index), ref_token_index >= 0) %>%
  count(ref_token_index, change_type, name = "n") %>%
  group_by(change_type) %>%
  mutate(n_scaled = n / max(n, na.rm = TRUE)) %>% # scale within change type
  ungroup() %>%
  # Add the reference token labels (1-based index for vector)
  mutate(ref_token = ifelse(ref_token_index + 1 <= length(ref_tokens),
                            ref_tokens[ref_token_index + 1], NA_character_))

if (nrow(pos_df) > 0) {
  p_heat <- ggplot(pos_df, aes(x = ref_token_index, y = change_type, fill = n)) +
    geom_tile() +
    scale_fill_continuous(name = "Count") +
    labs(title = "Change density across passage (by reference token index)",
         x = "Reference token index (0-based)", y = "Change type") +
    theme(axis.text.y = element_text(size = 10))
  ggsave(file.path(out_dir, "heatmap_change_density_by_position.png"), p_heat, width = 8, height = 4.5, dpi = 160)
  
  # label the x-axis with every 5th token for reference
  tick_idx <- unique(pmin(((0:(length(ref_tokens)-1)) %/% 5) * 5, length(ref_tokens)-1))
  p_heat_labs <- p_heat + scale_x_continuous(
    breaks = tick_idx,
    labels = ref_tokens[tick_idx + 1]
  )
  ggsave(file.path(out_dir, "heatmap_change_density_by_position_labeled.png"),
         p_heat_labs, width = 9, height = 5.2, dpi = 160)
}

# =====================================================================
# 10) Demographic ingestion
# =====================================================================
demo_path <- "/Users/alicagank/Desktop/orhan-kaya/metrics/demographic-data/demographic-data.xlsx"
has_demo <- file.exists(demo_path)

if (has_demo) {
  demo_raw <- read_xlsx(demo_path, sheet = 1) %>% clean_names()
  if (!"speaker" %in% names(demo_raw)) {
    warning("Demographics file found but no 'speaker' column; skipping demographic joins.")
    has_demo <- FALSE
  } else {
    demo <- demo_raw %>%
      mutate(speaker = as.character(speaker)) %>%
      distinct(speaker, .keep_all = TRUE)
    write_csv(demo, file.path(out_dir, "demographics_cleaned.csv"))
  }
}

# =====================================================================
# 11) time_studied/time_residency
# =====================================================================
if (has_demo) {
  agg_demo <- agg_by_speaker %>% left_join(demo, by = "speaker")

  to_num <- function(x) suppressWarnings(as.numeric(x))
  agg_demo <- agg_demo %>%
    mutate(
      time_studied_raw   = if ("time_studied"   %in% names(.)) time_studied   else NA,
      time_residency_raw = if ("time_residency" %in% names(.)) time_residency else NA
    ) %>%
    mutate(
      time_studied   = to_num(time_studied_raw),
      time_residency = to_num(time_residency_raw)
    ) %>%
    mutate(
      time_studied   = ifelse(is.na(time_studied)   | time_studied   < 0, NA, time_studied),
      time_residency = ifelse(is.na(time_residency) | time_residency < 0, NA, time_residency)
    ) %>%
    mutate(
      z_time_studied   = if (all(is.na(time_studied))) NA_real_ else as.numeric(scale(time_studied)),
      z_time_residency = if (all(is.na(time_residency))) NA_real_ else as.numeric(scale(time_residency))
    )
  
  agg_demo_brm <- agg_demo %>%
    mutate(
      y = as.integer(errors_on_ref),
      trials = as.integer(trials_ref)
    )
  
  write_csv(agg_demo, file.path(out_dir, "per_speaker_aggregates_with_demographics.csv"))
  
  # Summaries
  demodesc <- agg_demo %>%
    summarise(
      n_with_study = sum(!is.na(time_studied)),
      mean_study   = mean(time_studied, na.rm = TRUE),
      sd_study     = sd(time_studied, na.rm = TRUE),
      n_with_res   = sum(!is.na(time_residency)),
      mean_res     = mean(time_residency, na.rm = TRUE),
      sd_res       = sd(time_residency, na.rm = TRUE)
    )
  write_csv(demodesc, file.path(out_dir, "demographic_descriptives_time_study_residency.csv"))
  
  # WER vs time vars
  if (sum(!is.na(agg_demo$time_studied)) >= 3) {
    p_ts <- ggplot(agg_demo, aes(x = time_studied, y = wer)) +
      geom_point(alpha = 0.7) +
      geom_smooth(method = "lm", se = TRUE) +
      scale_y_continuous(labels = percent_format(accuracy = 1)) +
      labs(title = "WER vs years studied English (time_studied)",
           x = "Years studied English", y = "WER")
    ggsave(file.path(out_dir, "wer_vs_time_studied.png"), p_ts, width = 7, height = 4.5, dpi = 160)
  }
  if (sum(!is.na(agg_demo$time_residency)) >= 3) {
    p_tr <- ggplot(agg_demo, aes(x = time_residency, y = wer)) +
      geom_point(alpha = 0.7) +
      geom_smooth(method = "lm", se = TRUE) +
      scale_y_continuous(labels = percent_format(accuracy = 1)) +
      labs(title = "WER vs years in English-speaking country (time_residency)",
           x = "Years of residency", y = "WER")
    ggsave(file.path(out_dir, "wer_vs_time_residency.png"), p_tr, width = 7, height = 4.5, dpi = 160)
  }
  
  # --------------------------
  # 12) Frequentist GLMMs
  # --------------------------
  if (sum(is.finite(agg_demo$z_time_studied), na.rm = TRUE) >= 3) {
    m_study <- glmer(
      cbind(errors_on_ref, correct) ~ z_time_studied + (1 | speaker),
      data = agg_demo %>% filter(is.finite(z_time_studied)),
      family = binomial(link = "logit"),
      control = glmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
    )
    broom.mixed::tidy(m_study, effects = "fixed", conf.int = TRUE) %>%
      write_csv(file.path(out_dir, "glmm_univar_fixed_time_studied.csv"))
    capture.output(summary(m_study), file = file.path(out_dir, "glmm_univar_time_studied_summary.txt"))
  }
  if (sum(is.finite(agg_demo$z_time_residency), na.rm = TRUE) >= 3) {
    m_res <- glmer(
      cbind(errors_on_ref, correct) ~ z_time_residency + (1 | speaker),
      data = agg_demo %>% filter(is.finite(z_time_residency)),
      family = binomial(link = "logit"),
      control = glmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
    )
    broom.mixed::tidy(m_res, effects = "fixed", conf.int = TRUE) %>%
      write_csv(file.path(out_dir, "glmm_univar_fixed_time_residency.csv"))
    capture.output(summary(m_res), file = file.path(out_dir, "glmm_univar_time_residency_summary.txt"))
  }
  if (sum(is.finite(agg_demo$z_time_studied) & is.finite(agg_demo$z_time_residency), na.rm = TRUE) >= 3) {
    m_biv <- glmer(
      cbind(errors_on_ref, correct) ~ z_time_studied + z_time_residency + (1 | speaker),
      data = agg_demo %>% filter(is.finite(z_time_studied), is.finite(z_time_residency)),
      family = binomial(link = "logit"),
      control = glmerControl(optimizer = "bobyqa", calc.derivs = FALSE)
    )
    broom.mixed::tidy(m_biv, effects = "fixed", conf.int = TRUE) %>%
      write_csv(file.path(out_dir, "glmm_bivariate_fixed_study_residency.csv"))
    capture.output(summary(m_biv), file = file.path(out_dir, "glmm_bivariate_study_residency_summary.txt"))
  }
  
  # --------------------------
  # 13) Bayesian BRMS
  # --------------------------
  priors_time <- c(
    prior(student_t(3, 0, 2.5), class = "Intercept"),
    prior(student_t(3, 0, 2.5), class = "sd", group = "speaker"),
    prior(student_t(3, 0, 2.5), class = "b")
  )
  
  # Univariate: time_studied
  if (sum(is.finite(agg_demo$z_time_studied), na.rm = TRUE) >= 3) {
    df_brm <- agg_demo_brm %>% filter(is.finite(z_time_studied))
    b_study <- brm(
      formula = y | trials(trials) ~ z_time_studied + (1 | speaker),
      data = df_brm,
      family = binomial(link = "logit"),
      prior = priors_time,
      iter = 2000, warmup = 1000, chains = 4, cores = max(1, parallel::detectCores() - 1),
      seed = 501, control = list(adapt_delta = 0.95)
    )
    saveRDS(b_study, file.path(out_dir, "brms_univar_time_studied.rds"))
    sink(file.path(out_dir, "brms_univar_time_studied_summary.txt")); print(summary(b_study)); sink()
    
    # Conditional effects plot (on z-scale)
    ce_study <- conditional_effects(b_study, "z_time_studied", prob = 0.8)
    p_ce_study <- plot(ce_study, plot = FALSE)[[1]] +
      labs(title = "BRMS effect: z_time_studied on error probability",
           y = "Predicted error rate (prob)", x = "z_time_studied")
    ggsave(file.path(out_dir, "brms_univar_time_studied_conditional_effect.png"),
           p_ce_study, width = 6.8, height = 4.3, dpi = 160)
  }
  
  # Univariate: time_residency
  if (sum(is.finite(agg_demo$z_time_residency), na.rm = TRUE) >= 3) {
    df_brm <- agg_demo_brm %>% filter(is.finite(z_time_residency))
    b_res <- brm(
      formula = y | trials(trials) ~ z_time_residency + (1 | speaker),
      data = df_brm,
      family = binomial(link = "logit"),
      prior = priors_time,
      iter = 2000, warmup = 1000, chains = 4, cores = max(1, parallel::detectCores() - 1),
      seed = 502, control = list(adapt_delta = 0.95)
    )
    saveRDS(b_res, file.path(out_dir, "brms_univar_time_residency.rds"))
    sink(file.path(out_dir, "brms_univar_time_residency_summary.txt")); print(summary(b_res)); sink()
    
    # Conditional effects plot (on z-scale)
    ce_res <- conditional_effects(b_res, "z_time_residency", prob = 0.8)
    p_ce_res <- plot(ce_res, plot = FALSE)[[1]] +
      labs(title = "BRMS effect: z_time_residency on error probability",
           y = "Predicted error rate (prob)", x = "z_time_residency")
    ggsave(file.path(out_dir, "brms_univar_time_residency_conditional_effect.png"),
           p_ce_res, width = 6.8, height = 4.3, dpi = 160)
  }
  
  # Bivariate
  if (sum(is.finite(agg_demo$z_time_studied) & is.finite(agg_demo$z_time_residency), na.rm = TRUE) >= 3) {
    df_brm <- agg_demo_brm %>% filter(is.finite(z_time_studied), is.finite(z_time_residency))
    b_biv <- brm(
      formula = y | trials(trials) ~ z_time_studied + z_time_residency + (1 | speaker),
      data = df_brm,
      family = binomial(link = "logit"),
      prior = priors_time,
      iter = 2500, warmup = 1000, chains = 4, cores = max(1, parallel::detectCores() - 1),
      seed = 503, control = list(adapt_delta = 0.97)
    )
    saveRDS(b_biv, file.path(out_dir, "brms_bivariate_study_residency.rds"))
    sink(file.path(out_dir, "brms_bivariate_study_residency_summary.txt")); print(summary(b_biv)); sink()
    
    #Conditional effects (marginals)
    ce_biv_study <- conditional_effects(b_biv, "z_time_studied", prob = 0.8)
    ce_biv_res   <- conditional_effects(b_biv, "z_time_residency", prob = 0.8)
    p_ce_biv_study <- plot(ce_biv_study, plot = FALSE)[[1]] +
      labs(title = "BRMS bivariate: effect of z_time_studied", y = "Predicted error rate", x = "z_time_studied")
    p_ce_biv_res <- plot(ce_biv_res, plot = FALSE)[[1]] +
      labs(title = "BRMS bivariate: effect of z_time_residency", y = "Predicted error rate", x = "z_time_residency")
    ggsave(file.path(out_dir, "brms_bivariate_ce_time_studied.png"), p_ce_biv_study, width = 6.8, height = 4.3, dpi = 160)
    ggsave(file.path(out_dir, "brms_bivariate_ce_time_residency.png"), p_ce_biv_res, width = 6.8, height = 4.3, dpi = 160)
  }
} else {
  message("Demographic file not found or missing 'speaker' column — demographic analyses skipped.")
}

message("Done. Outputs in: ", out_dir)

