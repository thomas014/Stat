import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Advanced Stat Tool", layout="centered")
plt.style.use('seaborn-v0_8-whitegrid')

# --- 2. KNOWLEDGE BASE (Method Explanations) ---
method_info = {
    'ttest_ind': {
        "name": "Independent T-Test",
        "when": "Comparing the averages (means) of exactly 2 independent groups (e.g., Control vs Treatment).",
        "how": "Calculates the difference between group averages relative to the variance.",
        "interpret": "If P < 0.05, the difference is real and not due to chance."
    },
    'anova': {
        "name": "One-Way ANOVA",
        "when": "Comparing averages of 3+ independent groups (e.g., Diet A vs B vs C).",
        "how": "Analyzes variance 'between' groups vs 'within' groups.",
        "interpret": "If P < 0.05, at least one group is different. (Check the Pairwise table below)."
    },
    'mannwhitney': {
        "name": "Mann-Whitney U",
        "when": "Comparing 2 groups when data is NOT normal (skewed).",
        "how": "Ranks values from low to high and compares sum of ranks.",
        "interpret": "If P < 0.05, the distributions are significantly different."
    },
    'kruskal': {
        "name": "Kruskal-Wallis",
        "when": "Comparing 3+ groups when data is NOT normal (Non-parametric ANOVA).",
        "how": "Ranks data across all groups to see if one group consistently ranks higher.",
        "interpret": "If P < 0.05, groups differ. (Check Pairwise table below)."
    },
    'pearson': {
        "name": "Pearson Correlation",
        "when": "Linear relationship between two continuous numbers.",
        "how": "Calculates 'r' (-1 to 1).",
        "interpret": "High 'r' + P < 0.05 means a strong linear trend."
    },
    'spearman': {
        "name": "Spearman Correlation",
        "when": "Ranked/Non-linear relationship between two variables.",
        "how": "Converts values to ranks, then correlates.",
        "interpret": "High 'r' + P < 0.05 means a strong monotonic trend."
    },
    'chi2': {
        "name": "Chi-Square Test",
        "when": "Checking association between two Categorical variables.",
        "how": "Compares Observed counts vs Expected counts.",
        "interpret": "If P < 0.05, variables are related/dependent."
    }
}

# --- 3. UI LAYOUT ---
st.title("📊 Advanced Statistical Analysis Tool")
st.markdown("---")

# A. File Upload
st.header("Step 1: Upload Data")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File Uploaded Successfully")
        cols = df.columns.tolist()

        # B. Variable Selection
        st.header("Step 2: Select Variables")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Independent Variable (X / Groups)", cols)
        with col2:
            y_col = st.selectbox("Dependent Variable (Y / Values)", cols)

        # C. Method Selection
        st.header("Step 3: Select Method")
        
        # Create a reverse mapping for the dropdown (Label -> Key)
        method_map = {
            "Independent T-Test (2 Groups)": "ttest_ind",
            "One-Way ANOVA (3+ Groups)": "anova",
            "Mann-Whitney U (Non-Parametric 2 Groups)": "mannwhitney",
            "Kruskal-Wallis (Non-Parametric 3+ Groups)": "kruskal",
            "Pearson Correlation (Linear)": "pearson",
            "Spearman Correlation (Ranked)": "spearman",
            "Chi-Square Test (Categorical)": "chi2"
        }
        
        selected_label = st.selectbox("Choose Method", list(method_map.keys()))
        method_key = method_map[selected_label]
        
        # --- D. EDUCATIONAL GUIDE (Expander) ---
        info = method_info[method_key]
        with st.expander(f"📘 Method Guide: {info['name']} (Click to Expand)", expanded=True):
            st.markdown(f"**When to use:** {info['when']}")
            st.markdown(f"**How it works:** {info['how']}")
            st.markdown(f"**Interpretation:** {info['interpret']}")

        # --- E. RUN ANALYSIS ---
        if st.button("Run Analysis", type="primary"):
            st.divider()
            st.header(f"🚀 Results: {info['name']}")

            # Data Prep
            df_clean = df[[x_col, y_col]].dropna()
            x_data = df_clean[x_col]
            y_data = df_clean[y_col]

            # --- PART 1: DESCRIPTIVE STATISTICS (Summary First) ---
            # Only show this for Group comparisons, not simple correlations
            if method_key in ['anova', 'kruskal', 'ttest_ind', 'mannwhitney']:
                st.subheader("1. Group Summaries")
                stats_df = df_clean.groupby(x_col)[y_col].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
                stats_df['range'] = stats_df['max'] - stats_df['min']
                # Gradient highlight on Mean
                st.dataframe(stats_df.style.background_gradient(cmap='Blues', subset=['mean']), use_container_width=True)

            # --- PART 2: GLOBAL TEST & MAIN GRAPH ---
            st.subheader("2. Global Test Result & Distribution")

            # Logic: ANOVA / KRUSKAL
            if method_key in ['anova', 'kruskal']:
                groups = [y_data[x_data == g] for g in x_data.unique()]
                
                if method_key == 'anova':
                    stat, p = stats.f_oneway(*groups)
                else:
                    stat, p = stats.kruskal(*groups)
                
                # Display Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Statistic", f"{stat:.4f}")
                c2.metric("P-Value", f"{p:.4e}")
                
                if p < 0.05:
                    c3.success("Significant Difference (p < 0.05)")
                else:
                    c3.warning("No Difference (p >= 0.05)")
                
                # Main Boxplot
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=x_col, y=y_col, data=df_clean, hue=x_col, palette="viridis", legend=False, ax=ax)
                st.pyplot(fig)

                # --- PART 3: POST-HOC (If Significant) ---
                if p < 0.05:
                    st.divider()
                    st.subheader("3. Post-Hoc Analysis (Deep Dive)")
                    st.caption("Which specific groups are different?")

                    # A. ANOVA -> TUKEY
                    if method_key == 'anova':
                        tukey = pairwise_tukeyhsd(endog=y_data, groups=x_data, alpha=0.05)
                        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
                        
                        st.dataframe(tukey_df.style.apply(lambda x: ['background-color: #d4edda' if x['reject'] else '' for i in x], axis=1))
                        
                        st.subheader("4. Grouping Graph")
                        st.info("Groups with overlapping lines are statistically similar.")
                        # Tukey Plot
                        fig = tukey.plot_simultaneous(figsize=(8, 4))
                        st.pyplot(fig)

                    # B. KRUSKAL -> PAIRWISE MANN-WHITNEY
                    else:
                        unique_groups = x_data.unique()
                        pairs = list(itertools.combinations(unique_groups, 2))
                        corr_alpha = 0.05 / len(pairs)
                        
                        results = []
                        p_matrix = pd.DataFrame(index=unique_groups, columns=unique_groups, dtype=float)

                        for g1, g2 in pairs:
                            u_stat, u_p = stats.mannwhitneyu(y_data[x_data==g1], y_data[x_data==g2])
                            sig = u_p < corr_alpha
                            results.append([g1, g2, u_p, sig])
                            p_matrix.at[g1, g2] = u_p
                            p_matrix.at[g2, g1] = u_p
                        
                        res_df = pd.DataFrame(results, columns=['Group 1', 'Group 2', 'P-Value', 'Significant?'])
                        st.dataframe(res_df.style.apply(lambda x: ['background-color: #d4edda' if x['Significant?'] else '' for i in x], axis=1))
                        st.caption(f"Bonferroni Adjusted Alpha: {corr_alpha:.4f}")
                        
                        st.subheader("4. Grouping Graph (Heatmap)")
                        st.info("Light/Blue = Similar Groups. Red = Different Groups.")
                        
                        # Heatmap
                        fig, ax = plt.subplots(figsize=(6, 5))
                        p_matrix.fillna(1.0, inplace=True)
                        sns.heatmap(p_matrix, annot=True, cmap="coolwarm_r", vmin=0, vmax=0.05, ax=ax)
                        st.pyplot(fig)

            # Logic: T-TEST / MANN-WHITNEY
            elif method_key in ['ttest_ind', 'mannwhitney']:
                groups = x_data.unique()
                if len(groups) != 2:
                    st.error(f"Error: Independent variable must have exactly 2 groups. Found {len(groups)}.")
                else:
                    g1 = y_data[x_data == groups[0]]
                    g2 = y_data[x_data == groups[1]]
                    
                    if method_key == 'ttest_ind':
                        stat, p = stats.ttest_ind(g1, g2)
                    else:
                        stat, p = stats.mannwhitneyu(g1, g2)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Statistic", f"{stat:.4f}")
                    c2.metric("P-Value", f"{p:.4e}")
                    
                    if p < 0.05: c3.success("Significant Difference")
                    else: c3.warning("No Difference")
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.boxplot(x=x_col, y=y_col, data=df_clean, hue=x_col, palette="Set2", legend=False, ax=ax)
                    st.pyplot(fig)

            # Logic: CORRELATION / CHI2
            elif method_key in ['pearson', 'spearman', 'chi2']:
                if method_key == 'chi2':
                    ct = pd.crosstab(x_data, y_data)
                    stat, p, _, _ = stats.chi2_contingency(ct)
                    st.write(f"P-Value: {p:.4e}")
                    
                    fig, ax = plt.subplots()
                    sns.heatmap(ct, annot=True, fmt='d', cmap="Blues", ax=ax)
                    st.pyplot(fig)
                else:
                    if method_key == 'pearson': stat, p = stats.pearsonr(x_data, y_data)
                    else: stat, p = stats.spearmanr(x_data, y_data)
                    
                    st.write(f"Correlation (r): {stat:.4f}, P-Value: {p:.4e}")
                    
                    fig, ax = plt.subplots()
                    sns.regplot(x=x_col, y=y_col, data=df_clean, line_kws={'color':'red'}, ax=ax)
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading file: {e}")

else:
    st.info("Please upload a CSV or Excel file to begin.")