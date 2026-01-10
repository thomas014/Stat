```markdown
# 📊 Advanced Statistical Analysis & Educational Tool

## Overview
This is a local, privacy-focused Python application built with **Streamlit**. It allows users to perform advanced statistical analysis on their own data without uploading files to the cloud.

Designed for both data novices and experts, the tool acts as an interactive guide: it not only calculates the math but explains **when** to use a specific test, **how** it works, and **what** the results mean in plain English.

## 🌟 Key Features
* **Privacy First:** Runs entirely on your local machine (Localhost). No data leaves your computer.
* **Smart Method Guide:** Dynamically explains the selected statistical test (Logic, Usage, Interpretation) before you run it.
* **Advanced Group Analysis:**
    * Automatically triggers **Tukey HSD** (for ANOVA) or **Pairwise Mann-Whitney** (for Kruskal-Wallis) if significant differences are found among 3+ groups.
    * Includes Bonferroni corrections for non-parametric pairwise tests.
* **Visualizations:** Auto-generates professional plots:
    * **Boxplots** (Comparison tests)
    * **Scatter Plots with Regression** (Correlation)
    * **Heatmaps** (Chi-Square)
* **Format Support:** Accepts both `.csv` and `.xlsx` (Excel) files.

---

## 🛠️ Logic & Architecture
The application follows a strict **Input $\rightarrow$ Process $\rightarrow$ Visualization** pipeline:

1.  **Ingestion:** Uses `pandas` to load data into a DataFrame.
2.  **Validation:** Checks data types (Numeric vs Categorical) and group counts (e.g., ensuring a T-Test has exactly 2 groups).
3.  **Calculation:**
    * Uses `scipy.stats` for primary hypothesis testing.
    * Uses `statsmodels` for post-hoc analysis (Tukey HSD).
4.  **Interpretation:** Converts raw P-values into human-readable verdicts ("Significant" vs "Not Significant").

---

## 📦 Installation & Requirements

### 1. Prerequisites
You need **Python 3.8+** installed on your computer.

### 2. Install Libraries
Open your terminal or command prompt and run the following command to install the necessary dependencies:

```bash
pip install streamlit pandas numpy scipy matplotlib seaborn statsmodels openpyxl

```

---

## 🚀 How to Run the App

1. Save the Python script as `app.py`.
2. Open your terminal/command prompt.
3. Navigate to the folder containing the file:
```bash
cd path/to/your/folder

```


4. Run the application:
```bash
streamlit run app.py

```


5. A new tab will automatically open in your web browser (usually at `http://localhost:8501`).

---

## 📖 How to Use

1. **Upload Data:** Click "Browse files" to upload your CSV or Excel file.
2. **Select Variables:**
* **Independent Variable (X):** The group or "cause" (e.g., *Diet Type*).
* **Dependent Variable (Y):** The value or "effect" (e.g., *Weight Loss*).


3. **Choose Method:** Select the statistical test from the dropdown.
* *Tip:* Read the "Method Guide" box that appears to confirm you chose the right test.


4. **Run Analysis:** Click the button to calculate.
* If you selected ANOVA/Kruskal-Wallis and the result is significant, scroll down to see the **Pairwise Comparison Table**.



---

## 🤖 Original AI Prompt (Reference)

*This application was generated using the following prompt:*

> **Role:** Senior Python Data Scientist.
> **Task:** Create an interactive Statistical Analysis Tool for Streamlit (Local PC).
> **Libraries:** `pandas`, `scipy.stats`, `matplotlib.pyplot`, `seaborn`, `statsmodels`.
> **Requirements:**
> 1. **Data Loading:** Upload .csv/.xlsx.
> 2. **Logic:**
> * Comparison: T-Test (2 groups), ANOVA (3+ groups), Mann-Whitney (Non-para 2), Kruskal (Non-para 3+).
> * Correlation: Pearson, Spearman.
> * Categorical: Chi-Square.
> 
> 
> 3. **Features:**
> * If ANOVA/Kruskal is significant (p<0.05), perform Post-hoc tests (Tukey HSD or Pairwise Mann-Whitney with Bonferroni).
> * Display an educational "Method Guide" explaining the selected test.
> 
> 
> 4. **Outputs:**
> * Statistics (Stat, P-value, Verdict).
> * Graphs (Boxplot, Scatter, Heatmap).
> 
> 
> 
> 

---

## ⚖️ License

This project is open-source. Feel free to modify and use it for personal or commercial analysis.

```

```