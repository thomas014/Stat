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
    'ttest_1samp': {
        "name": "One-Sample T-Test (vs Constant)",
        "when": "Comparing the MEAN of a group against a Reference Value (Normal Data).",
        "assumptions": "Normality (Shapiro > 0.05).",
        "min_size": "Min N=2. (No Max, but N>5000 may overpower P-values).",
        "null_hypo": "The group mean equals the Reference Value.",
        "interpret": "If P < 0.05, the mean differs from Reference."
    },
    'wilcoxon': {
        "name": "Wilcoxon Signed-Rank (vs Constant)",
        "when": "Comparing the MEDIAN of a group against a Reference Value (Symmetric Non-Normal).",
        "assumptions": "Symmetric distribution (does not need normality).",
        "min_size": "Min N=5. (No Max).",
        "null_hypo": "The group median equals the Reference Value.",
        "interpret": "If P < 0.05, the median differs from Reference."
    },
    'signtest': { ### NEW METHOD ###
        "name": "Sign Test (vs Constant)",
        "when": "Comparing the MEDIAN of a group against a Reference Value (Skewed/Asymmetric Data).",
        "assumptions": "Independent observations. Data is at least ordinal.",
        "min_size": "Min N=5. (No Max).",
        "null_hypo": "The median of differences is zero (Equal pos/neg deviations).",
        "interpret": "If P < 0.05, the median differs from Reference."
    },
    'ttest_ind': {
        "name": "Independent T-Test (2 Groups)",
        "when": "Comparing averages of exactly 2 independent groups.",
        "assumptions": "Normality, Homogeneity of Variance.",
        "min_size": "Min N=2/group. (No Max).",
        "null_hypo": "The means of the two groups are equal.",
        "interpret": "If P < 0.05, the difference is real."
    },
    'anova': {
        "name": "One-Way ANOVA (3+ Groups)",
        "when": "Comparing averages of 3+ independent groups.",
        "assumptions": "Normality, Homogeneity of Variance.",
        "min_size": "Min N=2/group. (No Max).",
        "null_hypo": "All group means are equal.",
        "interpret": "If P < 0.05, at least one group is different."
    },
    'mannwhitney': {
        "name": "Mann-Whitney U (Non-Parametric)",
        "when": "Comparing 2 groups when data is NOT normal.",
        "assumptions": "Independent samples.",
        "min_size": "Min N=5/group. (No Max).",
        "null_hypo": "Distributions of both groups are equal.",
        "interpret": "If P < 0.05, distributions differ."
    },
    'kruskal': {
        "name": "Kruskal-Wallis (Non-Parametric)",
        "when": "Comparing 3+ groups when data is NOT normal.",
        "assumptions": "Independent samples.",
        "min_size": "Min N=5/group. (No Max).",
        "null_hypo": "Population medians are equal.",
        "interpret": "If P < 0.05, groups differ."
    },
    'pearson': {
        "name": "Pearson Correlation",
        "when": "Linear relationship between two continuous numbers.",
        "assumptions": "Linearity, Normality.",
        "min_size": "Min Pairs=3. (No Max).",
        "null_hypo": "No linear correlation (r = 0).",
        "interpret": "P < 0.05 means a strong linear trend."
    },
    'spearman': {
        "name": "Spearman Correlation",
        "when": "Ranked/Non-linear relationship.",
        "assumptions": "Monotonic relationship.",
        "min_size": "Min Pairs=5. (No Max).",
        "null_hypo": "No monotonic correlation.",
        "interpret": "P < 0.05 means a strong monotonic trend."
    },
    'chi2': {
        "name": "Chi-Square Test",
        "when": "Association between two Categorical variables.",
        "assumptions": "Expected counts > 5 in 80% of cells.",
        "min_size": "Total N > 20. (No Max).",
        "null_hypo": "Variables are independent.",
        "interpret": "P < 0.05 means variables are related."
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

        # --- REACTIVE ANALYSIS: DISTRIBUTION & NORMALITY ---
        st.divider()
        with st.expander("🔎 Data Distribution & Normality Check", expanded=True):
            df_clean = df[[x_col, y_col]].dropna()
            
            # 1. Group Summaries
            if pd.api.types.is_numeric_dtype(df_clean[y_col]) and df_clean[x_col].nunique() < 20:
                st.subheader("1. Group Summaries")
                stats_df = df_clean.groupby(x_col)[y_col].agg(['count', 'mean', 'std', 'min', 'max'])
                st.dataframe(stats_df.style.background_gradient(cmap='Blues', subset=['mean']), use_container_width=True)
            
            # 2. Normality Tests & HISTOGRAM
            st.subheader("2. Normality Tests & Visualization")
            if pd.api.types.is_numeric_dtype(df_clean[y_col]):
                is_grouping = df_clean[x_col].nunique() < 20
                
                c_vis1, c_vis2 = st.columns([1, 2])
                with c_vis1:
                    st.write("**Normality Statistics:**")
                    normality_results = []
                    
                    def check_normality(data, name):
                        N = len(data)
                        if N < 3:
                            return {"Group": name, "N": N, "Test": "Too Small", "P-Val": "-", "Res": "Unknown"}
                        elif N > 5000:
                            result = stats.anderson(data, dist='norm')
                            stat = result.statistic
                            crit = result.critical_values[2] 
                            conclusion = "Normal ✅" if stat < crit else "Non-Normal ⚠️"
                            return {"Group": name, "N": N, "Test": "Anderson (Large N)", "P-Val": f"Stat={stat:.2f}", "Res": conclusion}
                        else:
                            stat, p = stats.shapiro(data)
                            conclusion = "Normal ✅" if p > 0.05 else "Non-Normal ⚠️"
                            return {"Group": name, "N": N, "Test": "Shapiro", "P-Val": f"{p:.4f}", "Res": conclusion}

                    if is_grouping:
                        groups = df_clean[x_col].unique()
                        for g in groups:
                            group_data = df_clean[df_clean[x_col] == g][y_col]
                            normality_results.append(check_normality(group_data, g))
                    else:
                        normality_results.append(check_normality(df_clean[y_col], "Global"))
                        
                    st.dataframe(pd.DataFrame(normality_results), hide_index=True)
                    st.caption("Note: For N > 5000, Anderson-Darling is used instead of Shapiro-Wilk.")

                with c_vis2:
                    st.write("**Distribution Visualizer:**")
                    if is_grouping:
                        target_group = st.selectbox("Select Group to Visualize:", df_clean[x_col].unique())
                        viz_data = df_clean[df_clean[x_col] == target_group][y_col]
                    else:
                        viz_data = df_clean[y_col]

                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.histplot(viz_data, kde=True, stat="density", color="skyblue", alpha=0.6, ax=ax)
                    xmin, xmax = ax.get_xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = stats.norm.pdf(x, viz_data.mean(), viz_data.std())
                    ax.plot(x, p, 'r', linewidth=2, label='Normal Dist')
                    ax.legend()
                    st.pyplot(fig)
            else:
                st.warning("Dependent variable is not numeric. Normality check skipped.")

        # C. Method Selection
        st.header("Step 3: Select Method")
        
        method_map = {
            "One-Sample T-Test (Compare Mean vs Ref)": "ttest_1samp",
            "Wilcoxon Signed-Rank (Compare Median vs Ref)": "wilcoxon",
            "Sign Test (Compare Median vs Ref - Skewed)": "signtest", ### NEW MAP OPTION ###
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

        ref_value = 0.0
        if method_key in ['ttest_1samp', 'wilcoxon', 'signtest']: ### UPDATED CONDITION ###
            st.info("ℹ️ This test compares your data against a fixed Reference number.")
            ref_value = st.number_input("Enter Reference Value:", value=0.0)

        with st.expander(f"📘 Method Guide: {info['name']}", expanded=False):
            st.markdown(f"**When to use:** {info['when']}")
            st.markdown(f"**Assumptions:** {info['assumptions']}")
            st.markdown(f"**Minimum Group Size:** {info['min_size']}")
            st.markdown(f"**Interpretation:** {info['interpret']}")

        # --- E. RUN ANALYSIS ---
        if st.button("Run Analysis", type="primary"):
            st.divider()
            st.header(f"🚀 Results: {info['name']}")
            
            ref_data = {
                "Null Hypothesis": [info['null_hypo']],
                "Assumptions": [info['assumptions']],
                "Sample Size Notes": [info['min_size']],
                "Interpretation": [info['interpret']]
            }
            st.table(pd.DataFrame(ref_data))
            
            df_clean = df[[x_col, y_col]].dropna()
            x_data = df_clean[x_col]
            y_data = df_clean[y_col]
            
            if len(df_clean) < 5:
                st.warning("⚠️ Warning: Small sample size (N < 5). Results may be unreliable.")

            st.subheader("1. Test Results")

            # --- ONE-SAMPLE TESTS (T-TEST, WILCOXON, SIGN) ---
            if method_key in ['ttest_1samp', 'wilcoxon', 'signtest']:
                st.write(f"Testing against Reference Value: **{ref_value}**")
                results = []
                groups = x_data.unique()
                for g in groups:
                    g_data = y_data[x_data == g]
                    
                    if method_key == 'ttest_1samp':
                        stat, p_val = stats.ttest_1samp(g_data, popmean=ref_value)
                        metric_name, metric_val = "Mean", g_data.mean()
                    elif method_key == 'wilcoxon':
                        stat, p_val = stats.wilcoxon(g_data - ref_value)
                        metric_name, metric_val = "Median", g_data.median()
                    else: # SIGN TEST
                        diffs = g_data - ref_value
                        pos = np.sum(diffs > 0)
                        neg = np.sum(diffs < 0)
                        n_valid = pos + neg # Ties (0) are ignored
                        
                        if n_valid == 0: 
                            p_val = 1.0 # Exact match
                        else:
                            # Use Binomial Test for exact P-value
                            res = stats.binomtest(k=pos, n=n_valid, p=0.5, alternative='two-sided')
                            p_val = res.pvalue
                        
                        metric_name, metric_val = "Median", g_data.median()

                    sig = "Significant 💥" if p_val < 0.05 else "Not Sig"
                    diff = metric_val - ref_value
                    
                    results.append({
                        "Group": g, "N": len(g_data),
                        f"{metric_name}": f"{metric_val:.2f}",
                        "Diff": f"{diff:.2f}",
                        "P-Value": f"{p_val:.4e}", "Result": sig
                    })
                
                res_df = pd.DataFrame(results)
                st.dataframe(res_df.style.apply(lambda x: ['background-color: #d4edda' if "💥" in x['Result'] else '' for i in x], axis=1))
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=x_col, y=y_col, data=df_clean, hue=x_col, palette="coolwarm", legend=False, ax=ax)
                ax.axhline(ref_value, color='red', linestyle='--', linewidth=2)
                st.pyplot(fig)

            # --- ANOVA / KRUSKAL ---
            elif method_key in ['anova', 'kruskal']:
                groups = [y_data[x_data == g] for g in x_data.unique()]
                if method_key == 'anova': stat, p = stats.f_oneway(*groups)
                else: stat, p = stats.kruskal(*groups)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Statistic", f"{stat:.4f}")
                c2.metric("P-Value", f"{p:.4e}")
                if p < 0.05: c3.success("Significant Difference")
                else: c3.warning("No Difference")
                
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
                        unique_groups = x_data.unique()
                        pairs = list(itertools.combinations(unique_groups, 2))
                        p_matrix = pd.DataFrame(index=unique_groups, columns=unique_groups, dtype=float)
                        for g1, g2 in pairs:
                            _, u_p = stats.mannwhitneyu(y_data[x_data==g1], y_data[x_data==g2])
                            p_matrix.at[g1, g2] = u_p
                            p_matrix.at[g2, g1] = u_p
                        p_matrix.fillna(1.0, inplace=True)
                        fig, ax = plt.subplots()
                        sns.heatmap(p_matrix, annot=True, cmap="coolwarm_r", vmin=0, vmax=0.05, ax=ax)
                        st.pyplot(fig)

            # --- T-TEST / MANN-WHITNEY ---
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

            # --- CORRELATIONS ---
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