import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import lilliefors
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

# --- 3A. CALCULATION DETAILS PANEL (RIGHT SIDE) ---
if "show_calc_panel" not in st.session_state:
    st.session_state.show_calc_panel = False
if "calc_logs" not in st.session_state:
    st.session_state.calc_logs = []

def _format_calc_value(value, max_rows=20, max_chars=1200):
    if value is None:
        return None
    try:
        if isinstance(value, (np.ndarray, list, tuple)):
            arr = np.array(value)
            text = np.array2string(arr, max_line_width=120, threshold=200)
        elif isinstance(value, pd.Series):
            text = value.head(max_rows).to_string()
        elif isinstance(value, pd.DataFrame):
            text = value.head(max_rows).to_string()
        else:
            text = str(value)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text
    except Exception:
        return str(value)

def log_step(section, message, level="info", value=None):
    st.session_state.calc_logs.append(
        {
            "section": section,
            "message": message,
            "level": level,
            "value": _format_calc_value(value),
        }
    )

def render_calc_panel():
    logs = st.session_state.calc_logs
    if not logs:
        content = "<p>No calculations yet.</p>"
    else:
        sections = {}
        for entry in logs:
            sections.setdefault(entry["section"], []).append(entry)
        content = ""
        for section, entries in sections.items():
            content += f"<h4>{section}</h4><ul>"
            for e in entries:
                cls = "calc-log-error" if e["level"] == "error" else "calc-log-info"
                content += f"<li class='{cls}'>{e['message']}"
                if e.get("value"):
                    content += f"<pre class='calc-log-data'>{e['value']}</pre>"
                content += "</li>"
            content += "</ul>"
    st.markdown(
        f"""
        <div id="calc-panel">
            <div class="calc-panel-header">Calculation Details</div>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <style>
    #calc-panel {
        position: fixed;
        top: 90px;
        right: 12px;
        width: 340px;
        max-height: 80vh;
        overflow-y: auto;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 12px 14px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        border-radius: 10px;
        z-index: 9999;
    }
    .calc-panel-header {
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 8px;
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 6px;
    }
    #calc-panel, #calc-panel * { color: #000000; }
    .calc-log-info { color: #000000; }
    .calc-log-error { color: #b91c1c; font-weight: 600; }
    .calc-log-data {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        padding: 6px 8px;
        border-radius: 6px;
        margin-top: 6px;
        white-space: pre-wrap;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.expander("🧾 Calculation Details (Toggle)", expanded=st.session_state.show_calc_panel):
    st.toggle("Show calculation details", key="show_calc_panel")
    if st.session_state.show_calc_panel:
        render_calc_panel()

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

        log_step("Inputs", f"Selected X column: {x_col}")
        log_step("Inputs", f"Selected Y column: {y_col}")

        # --- REACTIVE ANALYSIS: DISTRIBUTION & NORMALITY ---
        st.divider()
        with st.expander("🔎 Data Distribution & Normality Check", expanded=True):
            df_clean = df[[x_col, y_col]].dropna()
            log_step("Data Prep", f"Rows after dropna: {len(df_clean)}")
            log_step("Data Prep", "Preview (head)", value=df_clean.head(10))
            
            # 1. Group Summaries
            if pd.api.types.is_numeric_dtype(df_clean[y_col]) and df_clean[x_col].nunique() < 20:
                st.subheader("1. Group Summaries")
                # Added 'median' to the aggregation and included in styling
                stats_df = df_clean.groupby(x_col)[y_col].agg(
                    count='count',
                    mean='mean',
                    median='median',
                    std='std',
                    min='min',
                    q25=lambda s: s.quantile(0.25),
                    q75=lambda s: s.quantile(0.75),
                    max='max',
                )
                st.dataframe(
                    stats_df.style.background_gradient(cmap='Blues', subset=['mean', 'median', 'q25', 'q75']),
                    use_container_width=True,
                )
                log_step("Group Summaries", "Computed count/mean/median/std/min/q25/q75/max by group", value=stats_df)
            
            # 2. Normality Tests & HISTOGRAM
            st.subheader("2. Normality Tests & Visualization")
            if pd.api.types.is_numeric_dtype(df_clean[y_col]):
                is_grouping = df_clean[x_col].nunique() < 20

                st.write("**Normality Statistics:**")
                normality_results = []

                def check_normality(data, name):
                    N = len(data)
                    if N < 3:
                        return {
                            "Group": name,
                            "N": N,
                            "Test Used": "Too Small",
                            "Shapiro-Wilk P": "-",
                            "Lilliefors P": "-",
                            "Skew": "-",
                            "Kurtosis": "-",
                            "Z-Skew": "-",
                            "Z-Kurt": "-",
                            "Res": "Unknown",
                        }

                    if np.isclose(np.std(data, ddof=1), 0.0, atol=1e-12):
                        return {
                            "Group": name,
                            "N": N,
                            "Test Used": "Constant",
                            "Shapiro-Wilk P": "-",
                            "Lilliefors P": "-",
                            "Skew": "0.000",
                            "Kurtosis": "0.000",
                            "Z-Skew": "-",
                            "Z-Kurt": "-",
                            "Res": "Unknown",
                        }

                    skew = stats.skew(data, bias=False)
                    kurt = stats.kurtosis(data, fisher=True, bias=False)
                    # SPSS/SAS standard errors for skewness and kurtosis
                    if N > 3:
                        se_skew = np.sqrt((6 * N * (N - 1)) / ((N - 2) * (N + 1) * (N + 3)))
                        se_kurt = 2 * se_skew * np.sqrt((N**2 - 1) / ((N - 3) * (N + 5)))
                    else:
                        se_skew = np.nan
                        se_kurt = np.nan
                    z_skew = skew / se_skew if np.isfinite(se_skew) and se_skew != 0 else np.nan
                    z_kurt = kurt / se_kurt if np.isfinite(se_kurt) and se_kurt != 0 else np.nan

                    shapiro_p = "-"
                    lillie_p = "-"
                    if N <= 2000:
                        try:
                            _, p = stats.shapiro(data)
                            shapiro_p = f"{p:.4f}"
                        except Exception:
                            shapiro_p = "-"

                        try:
                            _, p = lilliefors(data, dist="norm")
                            lillie_p = f"{p:.4f}"
                        except Exception:
                            lillie_p = "-"

                        z_limit = 1.96 if N < 50 else 3.29
                        z_ok = (abs(z_skew) <= z_limit) and (abs(z_kurt) <= z_limit)
                        shapiro_ok = (shapiro_p != "-") and (float(shapiro_p) > 0.05)
                        lillie_ok = (lillie_p != "-") and (float(lillie_p) > 0.05)
                        test_ok = shapiro_ok and lillie_ok
                        conclusion = "Normal ✅" if (test_ok and z_ok) else "Non-Normal ⚠️"
                        test_name = "Shapiro & Lilliefors"
                    else:
                        normal_enough = (abs(skew) < 2.0) and (abs(kurt) < 7.0)
                        conclusion = "Normal ✅" if normal_enough else "Non-Normal ⚠️"
                        test_name = "Skew/Kurtosis"

                    return {
                        "Group": name,
                        "N": N,
                        "Test Used": test_name,
                        "Shapiro-Wilk P": shapiro_p,
                        "Lilliefors P": lillie_p,
                        "Res": conclusion,
                        "Skew": f"{skew:.3f}",
                        "Kurtosis": f"{kurt:.3f}",
                        "Z-Skew": f"{z_skew:.3f}" if np.isfinite(z_skew) else "-",
                        "Z-Kurt": f"{z_kurt:.3f}" if np.isfinite(z_kurt) else "-",
                    }

                if is_grouping:
                    groups = df_clean[x_col].unique()
                    for g in groups:
                        group_data = df_clean[df_clean[x_col] == g][y_col]
                        normality_results.append(check_normality(group_data, g))
                else:
                    normality_results.append(check_normality(df_clean[y_col], "Global"))

                log_step("Normality", f"Normality checks completed for {len(normality_results)} group(s)", value=pd.DataFrame(normality_results))

                normality_df = pd.DataFrame(normality_results)
                normality_tests_df = normality_df[["Group", "N", "Shapiro-Wilk P", "Lilliefors P", "Test Used", "Res"]]
                normality_shape_df = normality_df[["Group", "Skew", "Kurtosis", "Z-Skew", "Z-Kurt", "Res"]]

                st.write("**Normality Test P-Values:**")
                st.dataframe(normality_tests_df, hide_index=True)
                st.write("**Shape Statistics (Skew/Kurtosis):**")
                st.dataframe(normality_shape_df, hide_index=True)

                st.caption("Normality method adapts to sample size: Shapiro-Wilk and Lilliefors (N≤2000), Skew/Kurtosis (N>2000).")
                st.markdown(
                    """
<div style="border: 1px solid #e5e7eb; background: #ffffff; padding: 12px 14px; border-radius: 8px; color: #000000;">
<strong>Normality & Visualization Guidelines (How to use the results):</strong>
<ul>
    <li>While Shapiro-Wilk is generally more powerful (better at detecting non-normality), there are rare "pathological" distributions where K-S might behave differently</li>
    <li><strong>Low N ($<50$): </strong> Shapiro-Wilk is king. K-S is too weak.</li>
    <li><strong>Medium N ($50-300$):</strong> Both usually agree, but K-S (Lilliefors) is the traditional choice in older textbooks.</li>
    <li><strong>High N ($1000-2000$): </strong> Both will likely be "over-sensitive," but seeing that both reject normality ($p < 0.001$) gives the user confidence that the data is genuinely not perfectly normal (mathematically speaking).</li>
</ul>
<br/>
<strong>Thresholds for Normality (How to read the results):</strong>
<ul>
    <li><strong>Small Samples (n &lt; 50):</strong> If the Z-score is between -1.96 and +1.96, the data is normal.</li>
    <li><strong>Medium Samples (50 &lt; n &lt; 300):</strong> If the Z-score is between -3.29 and +3.29, the data is normal.</li>
    <li><strong>Large Samples (n &gt; 300):</strong> Look at absolute skewness rather than Z-score. If |skewness| &lt; 2.0 and |kurtosis| &lt; 7.0, the data is likely normal enough for parametric tests.</li>
</ul>
</div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("**Distribution Visualizer:**")
                if is_grouping:
                    target_group = st.selectbox("Select Group to Visualize:", df_clean[x_col].unique())
                    viz_data = df_clean[df_clean[x_col] == target_group][y_col]
                else:
                    viz_data = df_clean[y_col]

                st.write("**Histogram:**")
                fig, ax = plt.subplots(figsize=(6, 3))
                sns.histplot(viz_data, kde=True, stat="density", color="skyblue", alpha=0.6, ax=ax)
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, viz_data.mean(), viz_data.std())
                ax.plot(x, p, 'r', linewidth=2, label='Normal Dist')
                ax.legend()
                ax.set_title("Histogram")
                st.pyplot(fig)

                st.write("**Q-Q Plot:**")
                fig, ax = plt.subplots(figsize=(6, 3))
                stats.probplot(viz_data, dist="norm", plot=ax)
                ax.set_title("Q-Q Plot")
                st.pyplot(fig)

                st.write("**Detrended Q-Q Plot (SPSS Standard):**")
                fig, ax = plt.subplots(figsize=(6, 3))
                (osm, osr), (slope, intercept, _) = stats.probplot(viz_data, dist="norm", fit=True)
                expected = slope * osm + intercept
                detrended = osr - expected
                ax.axhline(0, color="gray", linestyle="--", linewidth=1)
                ax.scatter(expected, detrended, color="tab:blue", s=18, alpha=0.8)
                ax.set_title("Detrended Q-Q Plot (SPSS Standard)")
                ax.set_xlabel("Expected Normal Value")
                ax.set_ylabel("Observed - Expected")
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
        log_step("Method Selection", f"Selected method: {info['name']}")

        ref_value = 0.0
        if method_key in ['ttest_1samp', 'wilcoxon', 'signtest']: ### UPDATED CONDITION ###
            st.info("ℹ️ This test compares your data against a fixed Reference number.")
            ref_value = st.number_input("Enter Reference Value:", value=0.0)
            log_step("Inputs", f"Reference value: {ref_value}")

        with st.expander(f"📘 Method Guide: {info['name']}", expanded=False):
            st.markdown(f"**When to use:** {info['when']}")
            st.markdown(f"**Assumptions:** {info['assumptions']}")
            st.markdown(f"**Minimum Group Size:** {info['min_size']}")
            st.markdown(f"**Interpretation:** {info['interpret']}")

        # --- E. RUN ANALYSIS ---
        if st.button("Run Analysis", type="primary"):
            st.divider()
            st.header(f"🚀 Results: {info['name']}")

            st.session_state.calc_logs = []
            log_step("Run Analysis", "Initialized calculation logs")
            log_step("Inputs", f"Method: {info['name']}")
            log_step("Inputs", f"X column: {x_col}")
            log_step("Inputs", f"Y column: {y_col}")
            if method_key in ['ttest_1samp', 'wilcoxon', 'signtest']:
                log_step("Inputs", f"Reference value: {ref_value}")
            
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

            log_step("Data Prep", f"Final dataset size: {len(df_clean)}")
            log_step("Data Prep", "Preview (head)", value=df_clean.head(10))
            
            if len(df_clean) < 5:
                st.warning("⚠️ Warning: Small sample size (N < 5). Results may be unreliable.")
                log_step("Warnings", "Small sample size (N < 5)", level="error")

            st.subheader("1. Test Results")

            # --- ONE-SAMPLE TESTS (T-TEST, WILCOXON, SIGN) ---
            if method_key in ['ttest_1samp', 'wilcoxon', 'signtest']:
                st.write(f"Testing against Reference Value: **{ref_value}**")
                log_step("One-Sample Tests", f"Reference value: {ref_value}")
                results = []
                groups = x_data.unique()
                for g in groups:
                    g_data = y_data[x_data == g]
                    log_step("One-Sample Tests", f"Group {g} data (head)", value=g_data.head(10))
                    
                    if method_key == 'ttest_1samp':
                        stat, p_val = stats.ttest_1samp(g_data, popmean=ref_value)
                        metric_name, metric_val = "Mean", g_data.mean()
                        log_step("One-Sample Tests", f"Group {g} t-test stats", value={"stat": stat, "p": p_val})
                    elif method_key == 'wilcoxon':
                        stat, p_val = stats.wilcoxon(g_data - ref_value)
                        metric_name, metric_val = "Median", g_data.median()
                        log_step("One-Sample Tests", f"Group {g} wilcoxon stats", value={"stat": stat, "p": p_val})
                    else: # SIGN TEST
                        diffs = g_data - ref_value
                        pos = np.sum(diffs > 0)
                        neg = np.sum(diffs < 0)
                        n_valid = pos + neg # Ties (0) are ignored
                        log_step("One-Sample Tests", f"Group {g} diffs (head)", value=diffs.head(10))
                        log_step("One-Sample Tests", f"Group {g} sign counts", value={"pos": int(pos), "neg": int(neg), "n_valid": int(n_valid)})
                        
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
                    log_step("One-Sample Tests", f"Group {g}: {metric_name}={metric_val:.4f}, P={p_val:.4e}")
                
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

                log_step("Group Comparison", f"Statistic={stat:.4f}, P={p:.4e}")
                log_step("Group Comparison", "Group sizes", value=[len(g) for g in groups])
                
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
                        log_step("Post-Hoc", "Tukey HSD computed", value=str(tukey))
                    else:
                        unique_groups = x_data.unique()
                        pairs = list(itertools.combinations(unique_groups, 2))
                        num_pairs = len(pairs)
                        p_matrix = pd.DataFrame(index=unique_groups, columns=unique_groups, dtype=float)
                        for g1, g2 in pairs:
                            _, u_p = stats.mannwhitneyu(y_data[x_data==g1], y_data[x_data==g2])
                            u_p_adjusted = min(u_p * num_pairs, 1.0)
                            p_matrix.at[g1, g2] = u_p_adjusted
                            p_matrix.at[g2, g1] = u_p_adjusted
                        p_matrix.fillna(1.0, inplace=True)
                        st.write(f"**Note:** P-values are Bonferroni-corrected for {num_pairs} comparisons.")
                        fig, ax = plt.subplots()
                        sns.heatmap(p_matrix, annot=True, cmap="coolwarm_r", vmin=0, vmax=0.05, ax=ax)
                        st.pyplot(fig)
                        log_step("Post-Hoc", f"Pairwise Mann-Whitney U computed (Bonferroni n={num_pairs})", value=p_matrix)

            # --- T-TEST / MANN-WHITNEY ---
            elif method_key in ['ttest_ind', 'mannwhitney']:
                groups = x_data.unique()
                if len(groups) != 2:
                    st.error("Error: Need exactly 2 groups.")
                    log_step("Errors", "Need exactly 2 groups for this test", level="error")
                else:
                    g1 = y_data[x_data == groups[0]]
                    g2 = y_data[x_data == groups[1]]
                    log_step("Two-Group Tests", f"Group {groups[0]} data (head)", value=g1.head(10))
                    log_step("Two-Group Tests", f"Group {groups[1]} data (head)", value=g2.head(10))
                    if method_key == 'ttest_ind':
                        stat, p = stats.ttest_ind(g1, g2)
                    else:
                        stat, p = stats.mannwhitneyu(g1, g2)
                    st.metric("P-Value", f"{p:.4e}")
                    log_step("Two-Group Tests", f"Statistic={stat:.4f}, P={p:.4e}", value={"stat": stat, "p": p})
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.boxplot(x=x_col, y=y_col, data=df_clean, hue=x_col, palette="Set2", legend=False, ax=ax)
                    st.pyplot(fig)

            # --- CORRELATIONS ---
            elif method_key in ['pearson', 'spearman', 'chi2']:
                if method_key == 'chi2':
                    ct = pd.crosstab(x_data, y_data)
                    stat, p, _, _ = stats.chi2_contingency(ct)
                    st.metric("P-Value", f"{p:.4e}")
                    log_step("Chi-Square", f"Statistic={stat:.4f}, P={p:.4e}", value=ct)
                else:
                    log_step("Correlation", "X data (head)", value=x_data.head(10))
                    log_step("Correlation", "Y data (head)", value=y_data.head(10))
                    if method_key == 'pearson':
                        stat, p = stats.pearsonr(x_data, y_data)
                    else:
                        stat, p = stats.spearmanr(x_data, y_data)
                    st.metric("Correlation", f"{stat:.4f}", delta=f"P={p:.4e}")
                    log_step("Correlation", f"Statistic={stat:.4f}, P={p:.4e}", value={"stat": stat, "p": p})
                    fig, ax = plt.subplots()
                    sns.regplot(x=x_col, y=y_col, data=df_clean, ax=ax)
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
        log_step("Errors", f"{e}", level="error")