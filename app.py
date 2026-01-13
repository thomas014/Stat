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

# --- 2. ENHANCED KNOWLEDGE BASE ---
method_info = {
    'ttest_ind': {
        "name": "Independent T-Test",
        "when": "Comparing averages of exactly 2 independent groups (e.g., Control vs Treatment).",
        "assumptions": "Normality (Shapiro > 0.05), Homogeneity of Variance (Levene > 0.05).",
        "null_hypo": "The means of the two groups are equal (µ1 = µ2).",
        "interpret": "If P < 0.05, the difference is real and not due to chance."
    },
    'anova': {
        "name": "One-Way ANOVA",
        "when": "Comparing averages of 3+ independent groups (e.g., Diet A vs B vs C).",
        "assumptions": "Normality (Residuals), Homogeneity of Variance (Levene > 0.05).",
        "null_hypo": "All group means are equal (µ1 = µ2 = µ3...).",
        "interpret": "If P < 0.05, at least one group is different. (Check the Pairwise table)."
    },
    'mannwhitney': {
        "name": "Mann-Whitney U",
        "when": "Comparing 2 groups when data is NOT normal (skewed).",
        "assumptions": "Observations are independent; ordinal or continuous scale.",
        "null_hypo": "Distributions of both groups are equal.",
        "interpret": "If P < 0.05, the distributions differ significantly."
    },
    'kruskal': {
        "name": "Kruskal-Wallis",
        "when": "Comparing 3+ groups when data is NOT normal (Non-parametric ANOVA).",
        "assumptions": "Independent samples; similar distribution shapes.",
        "null_hypo": "Population medians of all groups are equal.",
        "interpret": "If P < 0.05, groups differ. (Check Pairwise table)."
    },
    'pearson': {
        "name": "Pearson Correlation",
        "when": "Linear relationship between two continuous numbers.",
        "assumptions": "Linearity, Normality of variables, No outliers.",
        "null_hypo": "There is no linear correlation (r = 0).",
        "interpret": "High 'r' + P < 0.05 means a strong linear trend."
    },
    'spearman': {
        "name": "Spearman Correlation",
        "when": "Ranked/Non-linear relationship between two variables.",
        "assumptions": "Monotonic relationship (doesn't need normality).",
        "null_hypo": "There is no monotonic correlation.",
        "interpret": "High 'r' + P < 0.05 means a strong monotonic trend."
    },
    'chi2': {
        "name": "Chi-Square Test",
        "when": "Checking association between two Categorical variables.",
        "assumptions": "Expected count > 5 in 80% of cells.",
        "null_hypo": "Variables are independent (No relationship).",
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
        
        cols = df.columns.tolist()

        # B. Variable Selection
        st.header("Step 2: Select Variables")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Independent Variable (X / Groups)", cols)
        with col2:
            y_col = st.selectbox("Dependent Variable (Y / Values)", cols)

        # --- BETTER METHOD: REACTIVE ANALYSIS (Automatic) ---
        # No button needed. We run this block immediately when vars are selected.
        # We use an Expander so it stays visible but clean.
        st.divider()
        with st.expander("🔎 Data Distribution & Normality Check", expanded=True):
            
            df_clean = df[[x_col, y_col]].dropna()
            
            # 1. Group Summaries
            if pd.api.types.is_numeric_dtype(df_clean[y_col]) and df_clean[x_col].nunique() < 20:
                st.subheader("1. Group Summaries")
                stats_df = df_clean.groupby(x_col)[y_col].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
                st.dataframe(stats_df.style.background_gradient(cmap='Blues', subset=['mean']), use_container_width=True)
            
            # 2. Normality Tests
            st.subheader("2. Normality Tests")
            if pd.api.types.is_numeric_dtype(df_clean[y_col]):
                is_grouping = df_clean[x_col].nunique() < 20
                
                if is_grouping:
                    st.write(f"Checking Normality of **{y_col}** within each group of **{x_col}**:")
                    normality_results = []
                    groups = df_clean[x_col].unique()
                    
                    for g in groups:
                        group_data = df_clean[df_clean[x_col] == g][y_col]
                        if len(group_data) >= 3: 
                            stat, p = stats.shapiro(group_data)
                            conclusion = "Normal ✅" if p > 0.05 else "Non-Normal ⚠️"
                            normality_results.append({
                                "Group": g, "N": len(group_data), 
                                "P-Value": f"{p:.4f}", "Conclusion": conclusion
                            })
                    
                    if normality_results:
                        st.dataframe(pd.DataFrame(normality_results), use_container_width=True)
                        if any("⚠️" in r['Conclusion'] for r in normality_results):
                            st.warning("Recommendation: Data contains non-normal groups. Consider **Non-Parametric** tests (Mann-Whitney / Kruskal-Wallis).")
                        else:
                            st.success("Recommendation: Data looks Normal. You may use **Parametric** tests (T-Test / ANOVA).")
                else:
                    # Continuous X
                    stat, p = stats.shapiro(df_clean[y_col])
                    conclusion = "Normal ✅" if p > 0.05 else "Non-Normal ⚠️"
                    st.metric(label=f"Normality of {y_col}", value=conclusion, delta=f"P={p:.4f}")
            else:
                st.warning("Dependent variable is not numeric. Normality check skipped.")

        # C. Method Selection
        st.header("Step 3: Select Method")
        
        # ... [METHOD MAPPING REMIANS SAME] ...
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
        info = method_info[method_key]

        with st.expander(f"📘 Method Guide: {info['name']} (Click to Expand)", expanded=False):
            st.markdown(f"**When to use:** {info['when']}")
            st.markdown(f"**Interpretation:** {info['interpret']}")

        # --- E. RUN ANALYSIS ---
        if st.button("Run Analysis", type="primary"):
            st.divider()
            st.header(f"🚀 Results: {info['name']}")
            
            st.subheader("📋 Method Reference Card")
            ref_data = {
                "Null Hypothesis (H0)": [info['null_hypo']],
                "Key Assumptions": [info['assumptions']],
                "Interpretation Rule": [info['interpret']]
            }
            st.table(pd.DataFrame(ref_data))
            st.divider()

            df_clean = df[[x_col, y_col]].dropna()
            x_data = df_clean[x_col]
            y_data = df_clean[y_col]

            st.subheader("1. Global Test Result")

            # Logic: ANOVA / KRUSKAL
            if method_key in ['anova', 'kruskal']:
                groups = [y_data[x_data == g] for g in x_data.unique()]
                
                if method_key == 'anova':
                    stat, p = stats.f_oneway(*groups)
                else:
                    stat, p = stats.kruskal(*groups)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Statistic", f"{stat:.4f}")
                c2.metric("P-Value", f"{p:.4e}")
                
                if p < 0.05: c3.success("Significant Difference (Reject H0)")
                else: c3.warning("No Difference (Fail to Reject H0)")
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=x_col, y=y_col, data=df_clean, hue=x_col, palette="viridis", legend=False, ax=ax)
                st.pyplot(fig)

                if p < 0.05:
                    st.divider()
                    st.subheader("2. Post-Hoc Analysis")
                    if method_key == 'anova':
                        tukey = pairwise_tukeyhsd(endog=y_data, groups=x_data, alpha=0.05)
                        st.pyplot(tukey.plot_simultaneous())
                    else:
                        # [Existing Kruskal Post-hoc Logic]
                        unique_groups = x_data.unique()
                        pairs = list(itertools.combinations(unique_groups, 2))
                        p_matrix = pd.DataFrame(index=unique_groups, columns=unique_groups, dtype=float)
                        for g1, g2 in pairs:
                            u_stat, u_p = stats.mannwhitneyu(y_data[x_data==g1], y_data[x_data==g2])
                            p_matrix.at[g1, g2] = u_p
                            p_matrix.at[g2, g1] = u_p
                        
                        p_matrix.fillna(1.0, inplace=True)
                        fig, ax = plt.subplots()
                        sns.heatmap(p_matrix, annot=True, cmap="coolwarm_r", vmin=0, vmax=0.05, ax=ax)
                        st.pyplot(fig)

            # Logic: T-TEST / MANN-WHITNEY
            elif method_key in ['ttest_ind', 'mannwhitney']:
                groups = x_data.unique()
                if len(groups) != 2:
                    st.error("Error: Need exactly 2 groups.")
                else:
                    g1 = y_data[x_data == groups[0]]
                    g2 = y_data[x_data == groups[1]]
                    if method_key == 'ttest_ind': stat, p = stats.ttest_ind(g1, g2)
                    else: stat, p = stats.mannwhitneyu(g1, g2)
                    
                    st.metric("P-Value", f"{p:.4e}")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.boxplot(x=x_col, y=y_col, data=df_clean, hue=x_col, palette="Set2", legend=False, ax=ax)
                    st.pyplot(fig)

            # Logic: CORRELATIONS
            elif method_key in ['pearson', 'spearman', 'chi2']:
                if method_key == 'chi2':
                    ct = pd.crosstab(x_data, y_data)
                    stat, p, _, _ = stats.chi2_contingency(ct)
                    st.metric("P-Value", f"{p:.4e}")
                else:
                    if method_key == 'pearson': stat, p = stats.pearsonr(x_data, y_data)
                    else: stat, p = stats.spearmanr(x_data, y_data)
                    st.metric("Correlation", f"{stat:.4f}", delta=f"P={p:.4e}")
                    fig, ax = plt.subplots()
                    sns.regplot(x=x_col, y=y_col, data=df_clean, ax=ax)
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")