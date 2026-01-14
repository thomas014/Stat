# 📊 Advanced Statistical Analysis Tool (Local Streamlit Version)

## 📖 Overview
This is a **privacy-first**, local statistical analysis application built with Python and Streamlit. It allows you to perform professional-grade statistical testing on your own computer without uploading your data to the cloud.

The tool bridges the gap between raw data and insight by providing an **interactive educational guide**, automatically selecting the correct visualizations, and performing advanced post-hoc analysis (like Tukey HSD) when necessary.

## ✨ Key Features
* **🔒 100% Private:** Runs locally on your machine. Your data never leaves your hard drive.
* **📘 Smart Method Guide:** Updates dynamically to explain *When*, *How*, and *Why* to use a specific test.
* **🧠 Advanced Logic & Tests:**
    * **Summary First:** Automatically calculates descriptive statistics for groups (count, mean, **median**, std, min, max).
    * **Normality Checks:** Uses **Shapiro-Wilk** (default) for typical sample sizes and **Anderson-Darling** for very large samples (N > 5000). Normality is checked per-group when grouping is detected.
    * **Reference Comparisons:** Supports tests against a reference value (enterable in the UI) including One-Sample T-Test, **Wilcoxon Signed-Rank**, and the **Sign Test** (for skewed/asymmetric data).
    * **Post-Hoc Analysis:** Automatically triggers **Tukey HSD** (for ANOVA) or **Pairwise Mann-Whitney** (for Kruskal-Wallis) if significant group differences are found (with **Bonferroni** correction for pairwise nonparametric tests).
* **📈 Automated Visualization:**
    * **Grouping Graphs:** Visualizes which groups are statistically similar using Confidence Interval plots or Heatmaps.
    * **Boxplots & Scatterplots:** Auto-generated based on the test type.

## 🛠️ Requirements
* **Python:** Version 3.8 or higher.
* **Operating System:** Windows, macOS, or Linux.

## 📦 Installation

1.  **Clone or Download** this folder to your computer.
2.  **Open your Terminal** (Command Prompt or PowerShell).
3.  **Install the required libraries** by running the following command:

```bash
pip install streamlit pandas numpy scipy matplotlib seaborn statsmodels openpyxl