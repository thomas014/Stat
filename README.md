# 📊 Advanced Statistical Analysis Tool

## 📖 Overview
A **privacy-first**, local statistical analysis application built with Python and Streamlit. Perform professional-grade statistical testing on your own computer without uploading your data to the cloud. Your data stays on your machine, always.

This tool bridges the gap between raw data and actionable insights by providing an **interactive educational guide**, automatically selecting appropriate visualizations, and performing advanced post-hoc analysis when needed.

## ✨ Key Features

### 🔒 Privacy & Security
* **100% Local Processing:** Runs entirely on your machine. Your data never leaves your hard drive.
* **No External Dependencies:** No internet connection required after installation.

### 📘 Interactive Learning
* **Smart Method Guide:** Dynamic explanations that update based on your selected test, explaining *When*, *Why*, and *How* to use each method.
* **Comprehensive Test Information:** Includes assumptions, minimum sample sizes, null hypotheses, and interpretation guidance for every test.

### 🔬 Advanced Statistical Analysis

#### Normality Testing (Adaptive to Sample Size)
* **Small Samples (N < 50):** Shapiro-Wilk test with Z-score analysis
* **Medium Samples (50-300):** Lilliefors test (Kolmogorov-Smirnov adjusted)
* **Large Samples (N > 300):** Skewness and kurtosis evaluation
* **Per-Group Analysis:** Automatically checks normality for each group separately when applicable
* **Visual Diagnostics:** Histogram with normal curve overlay and Q-Q plots

#### Descriptive Statistics
* Automatically calculates group summaries: count, mean, **median**, standard deviation, min, and max
* Color-coded tables with gradient highlighting for easy interpretation

#### Statistical Tests Available

**One-Sample Tests (vs Reference Value):**
* One-Sample T-Test (for normal data)
* Wilcoxon Signed-Rank Test (for symmetric non-normal data)
* **Sign Test** (for skewed/asymmetric data)

**Two-Group Comparisons:**
* Independent T-Test (parametric)
* Mann-Whitney U Test (non-parametric)

**Multi-Group Comparisons (3+ groups):**
* One-Way ANOVA (parametric)
* Kruskal-Wallis Test (non-parametric)

**Correlation Analysis:**
* Pearson Correlation (linear relationships)
* Spearman Correlation (ranked/monotonic relationships)

**Categorical Analysis:**
* Chi-Square Test of Independence

#### SPSS-Style Output Tables
* **Independent T-Test:** Homogeneity of variance (Levene) and Independent Samples Test tables
* **Mann-Whitney U:** Summary table (U, W, z, SE, p)
* **Pearson & Spearman:** Correlations tables with SPSS-style layout
* **Chi-Square:** Case Processing Summary, Crosstabulation (counts + expected), Chi-Square Tests
* **P-Value Formatting:** Values < 0.001 are displayed as “<0.001” in output tables

#### Post-Hoc Analysis
* **Tukey HSD:** Automatically triggered after significant ANOVA results
* **Pairwise Mann-Whitney U:** Automatically triggered after significant Kruskal-Wallis results
* **Bonferroni Correction:** Applied to pairwise non-parametric tests to control family-wise error rate
* **Visual Results:** Confidence interval plots and heatmaps showing which groups differ

### 📈 Automated Visualizations
* **Distribution Plots:** Histograms with KDE and normal curve overlays
* **Q-Q Plots:** For assessing normality visually
* **Boxplots:** Automatically generated for group comparisons
* **Scatter Plots:** With regression lines for correlation analyses
* **Heatmaps:** For post-hoc pairwise comparison results
* **Crosstab Heatmap:** Observed count heatmap for Chi-Square analysis

### 🧾 Calculation Details Panel
* **Real-Time Logging:** See exactly what calculations are being performed
* **Transparency:** View intermediate steps, data previews, and statistical values
* **Educational:** Learn how the analysis is conducted behind the scenes
* **Collapsible:** Toggle on/off to reduce clutter when not needed

## 🛠️ System Requirements

### Software Requirements
* **Python:** Version 3.8 or higher
* **Operating System:** Windows, macOS, or Linux

### Python Dependencies
* streamlit - Web application framework
* pandas - Data manipulation and analysis
* numpy - Numerical computing
* scipy - Scientific computing and statistical functions
* matplotlib - Plotting and visualization
* seaborn - Statistical data visualization
* statsmodels - Statistical modeling and tests
* openpyxl - Excel file support

## 📦 Installation

### Quick Start

1. **Clone or Download** this repository to your computer:
   ```bash
   git clone https://github.com/thomas014/Stat.git
   cd Stat
   ```

2. **Install Dependencies:**
   
   Using pip (recommended):
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install packages individually:
   ```bash
   pip install streamlit pandas numpy scipy matplotlib seaborn statsmodels openpyxl
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the Tool:** 
   The application will automatically open in your default web browser (typically at `http://localhost:8501`)

### Troubleshooting

If you encounter issues:
* Ensure Python 3.8+ is installed: `python --version`
* Upgrade pip: `pip install --upgrade pip`
* Use virtual environment (recommended):
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

## 🚀 Usage Guide

### Step 1: Upload Your Data
* Supported formats: **CSV** or **Excel** (.xlsx)
* Your data should have:
  - One column for the independent variable (groups/categories)
  - One column for the dependent variable (measurements/values)

### Step 2: Select Variables
* **X (Independent Variable):** Choose the column that represents groups or categories
* **Y (Dependent Variable):** Choose the column with numerical measurements

### Step 3: Review Distribution & Normality
* The tool automatically displays:
  - Group summary statistics
  - Normality test results (adaptive to sample size)
  - Histogram and Q-Q plots for visual assessment

### Step 4: Choose Statistical Test
* Select from 10 different statistical methods
* The built-in guide explains when to use each test
* For reference value comparisons, enter your reference number

### Step 5: Run Analysis
* Click "Run Analysis" to see results including:
  - Test statistics and p-values
  - Automatic post-hoc tests (when applicable)
  - Publication-ready visualizations
  - Detailed calculation logs (optional)

## 📊 Example Use Cases

### Comparing Drug Efficacy
* **Scenario:** Testing if a new drug performs differently than a placebo
* **Test:** Independent T-Test (if normal) or Mann-Whitney U (if non-normal)
* **Result:** Boxplots showing group differences with p-values

### Quality Control
* **Scenario:** Checking if product measurements meet a target specification
* **Test:** One-Sample T-Test vs reference value
* **Result:** Statistical confirmation if the process is on-target

### Multi-Group Experiment
* **Scenario:** Comparing 4 different treatment conditions
* **Test:** ANOVA (if normal) or Kruskal-Wallis (if non-normal)
* **Result:** Automatic Tukey HSD or pairwise comparisons to identify which groups differ

### Relationship Analysis
* **Scenario:** Examining if two variables are related
* **Test:** Pearson (linear) or Spearman (monotonic) correlation
* **Result:** Scatter plot with regression line and correlation coefficient

## 📖 Statistical Methods Reference

### Test Selection Flowchart

**Comparing to a Reference Value?**
- Normal data → One-Sample T-Test
- Symmetric non-normal → Wilcoxon Signed-Rank
- Skewed data → Sign Test

**Comparing 2 Groups?**
- Normal data → Independent T-Test
- Non-normal data → Mann-Whitney U

**Comparing 3+ Groups?**
- Normal data → One-Way ANOVA (+ Tukey HSD if significant)
- Non-normal data → Kruskal-Wallis (+ pairwise tests if significant)

**Examining Relationships?**
- Linear relationship → Pearson Correlation
- Monotonic relationship → Spearman Correlation
- Categorical association → Chi-Square Test

## 🎓 Educational Features

### Understanding P-Values
* P < 0.05: Result is statistically significant (reject null hypothesis)
* P ≥ 0.05: Result is not significant (fail to reject null hypothesis)

### Normality Interpretation
* **Small samples:** Check if Shapiro-Wilk is >=0.05 or Z-scores are between -1.96 and +1.96
* **Medium samples:** Check if  K-S (Lilliefors) is >=0.05 or Z-scores are between -3.29 and +3.29
* **Large samples:** Check if |skewness| < 2.0 and |kurtosis| < 7.0

### Effect Size (Coming Soon)
Future versions will include effect size calculations (Cohen's d, eta-squared, etc.)

## 🔐 Privacy & Data Security

* **Local Processing:** All computations happen on your machine
* **No Data Upload:** Files are read directly from your computer
* **No Tracking:** No analytics, cookies, or external connections
* **Open Source:** Full code transparency - inspect what's happening

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🐛 Known Issues & Future Enhancements

### Planned Features
- [ ] Effect size calculations
- [ ] Power analysis tools
- [ ] Assumption violation warnings
- [ ] Export results to PDF/Word
- [ ] Multiple dependent variables analysis
- [ ] Repeated measures designs
- [ ] Mixed models support

### Current Limitations
* Maximum recommended data size: ~100,000 rows (performance considerations)
* Assumes independent observations
* Limited to univariate analyses

## 📞 Support & Contact

* **Issues:** Report bugs or request features via [GitHub Issues](https://github.com/thomas014/Stat/issues)
* **Documentation:** See this README and in-app guides
* **Owner:** thomas014

## 🙏 Acknowledgments

Built with:
* [Streamlit](https://streamlit.io/) - Application framework
* [SciPy](https://scipy.org/) - Statistical functions
* [Statsmodels](https://www.statsmodels.org/) - Advanced statistical models
* [Seaborn](https://seaborn.pydata.org/) - Statistical visualizations

---

**Made with ❤️ for researchers, students, and data analysts who value privacy and transparency in statistical analysis.**

## References
[1] H. Y. Kim, "Statistical notes for clinical researchers: Assessing normal distribution using skewness and kurtosis," Restor. Dent. Endod., vol. 38, no. 1, pp. 52–54, Feb. 2013.

[2] A. Ghasemi and S. Zahediasl, "Normality tests for statistical analysis: A guide for non-statisticians," Int. J. Endocrinol. Metab., vol. 10, no. 2, pp. 486–489, Apr. 2012.

[3] P. Mishra, C. M. Pandey, U. Singh, A. Gupta, C. Sahu, and A. Keshri, "Descriptive statistics and normality tests for statistical data," Ann. Card. Anaesth., vol. 22, no. 1, pp. 67–72, Jan. 2019.

[4] A. Field, Discovering Statistics Using IBM SPSS Statistics, 5th ed. Thousand Oaks, CA, USA: SAGE Publications, 2018.