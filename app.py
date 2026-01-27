import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.oneway import effectsize_oneway
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
                    q25=lambda s: np.percentile(s, 25, method='weibull'),
                    q75=lambda s: np.percentile(s, 75, method='weibull'),
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

        test_prop = 0.5
        if method_key == 'signtest':
            test_prop = st.number_input(
                "Enter Test Proportion for Sign Test:",
                value=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            )
            log_step("Inputs", f"Sign test proportion: {test_prop}")

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
            if method_key == 'signtest':
                log_step("Inputs", f"Sign test proportion: {test_prop}")
            
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
                binom_tables = []
                wilcoxon_tables = []
                groups = x_data.unique()
                for g in groups:
                    g_data = y_data[x_data == g]
                    log_step("One-Sample Tests", f"Group {g} data (head)", value=g_data.head(10))
                    
                    if method_key == 'ttest_1samp':
                        stat, p_val = stats.ttest_1samp(g_data, popmean=ref_value)
                        metric_name, metric_val = "Mean", g_data.mean()
                        log_step("One-Sample Tests", f"Group {g} t-test stats", value={"stat": stat, "p": p_val})
                    elif method_key == 'wilcoxon':
                        diffs = g_data - ref_value
                        diffs_nz = diffs[diffs != 0]
                        stat, p_val = stats.wilcoxon(diffs_nz)

                        abs_diffs = np.abs(diffs_nz.to_numpy())
                        ranks = stats.rankdata(abs_diffs, method="average")
                        pos_mask = diffs_nz.to_numpy() > 0
                        neg_mask = diffs_nz.to_numpy() < 0
                        w_plus = float(np.sum(ranks[pos_mask]))
                        w_minus = float(np.sum(ranks[neg_mask]))
                        w_stat = min(w_plus, w_minus)

                        n_total = int(len(g_data))
                        n_valid = int(len(diffs_nz))
                        mean_w = n_valid * (n_valid + 1) / 4
                        _, tie_counts = np.unique(abs_diffs, return_counts=True)
                        tie_correction = np.sum(tie_counts**3 - tie_counts) / 48
                        var_w = (n_valid * (n_valid + 1) * (2 * n_valid + 1)) / 24 - tie_correction
                        se_w = np.sqrt(var_w) if var_w > 0 else np.nan
                        z_w = (w_stat - mean_w) / se_w if np.isfinite(se_w) and se_w > 0 else np.nan
                        p_z = 2 * (1 - stats.norm.cdf(abs(z_w))) if np.isfinite(z_w) else 1.0

                        wilcoxon_table = pd.DataFrame([
                            {"": "Total N", "Value": n_total},
                            {"": "Test Statistic", "Value": w_stat},
                            {"": "Standard Error", "Value": se_w},
                            {"": "Standardized Test Statistic", "Value": z_w},
                            {"": "Asymptotic Sig.(2-sided test)", "Value": p_z},
                        ])
                        wilcoxon_tables.append((g, wilcoxon_table))

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
                            res = stats.binomtest(k=pos, n=n_valid, p=test_prop, alternative='two-sided')
                            p_val = res.pvalue
                        
                        metric_name, metric_val = "Median", g_data.median()

                        n_total = len(g_data)
                        count_le = int(np.sum(g_data <= ref_value))
                        count_gt = int(np.sum(g_data > ref_value))
                        obs_prop_le = count_le / n_total if n_total > 0 else np.nan
                        obs_prop_gt = count_gt / n_total if n_total > 0 else np.nan

                        binom_table = pd.DataFrame([
                            {
                                "Category": f"Group 1 (<= {ref_value})",
                                "N": count_le,
                                "Observed Prop.": obs_prop_le,
                                "Test Prop.": test_prop,
                                "Exact Sig. (2-tailed)": p_val,
                            },
                            {
                                "Category": f"Group 2 (> {ref_value})",
                                "N": count_gt,
                                "Observed Prop.": obs_prop_gt,
                                "Test Prop.": "",
                                "Exact Sig. (2-tailed)": "",
                            },
                            {
                                "Category": "Total",
                                "N": n_total,
                                "Observed Prop.": 1.00 if n_total > 0 else np.nan,
                                "Test Prop.": "",
                                "Exact Sig. (2-tailed)": "",
                            },
                        ])
                        binom_tables.append((g, binom_table))

                    sig = "Significant 💥" if p_val < 0.05 else "Not Sig"
                    diff = metric_val - ref_value
                    
                    results.append({
                        "Group": g, "N": len(g_data),
                        f"{metric_name}": f"{metric_val:.2f}",
                        "Diff": f"{diff:.2f}",
                        "P-Value": f"{p_val:.4e}", "Result": sig
                    })
                    log_step("One-Sample Tests", f"Group {g}: {metric_name}={metric_val:.4f}, P={p_val:.4e}")

                if method_key == 'signtest':
                    for g, table in binom_tables:
                        st.write(f"**Binomial Test ({g})**")
                        display_table = table.copy()
                        display_table["Observed Prop."] = display_table["Observed Prop."].map(
                            lambda v: f"{v:.2f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else v
                        )
                        display_table["Test Prop."] = display_table["Test Prop."].map(
                            lambda v: f"{v:.2f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else v
                        )
                        display_table["Exact Sig. (2-tailed)"] = display_table["Exact Sig. (2-tailed)"].map(
                            lambda v: f"{v:.3f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else v
                        )
                        st.dataframe(display_table, hide_index=True)
                elif method_key == 'wilcoxon':
                    for g, table in wilcoxon_tables:
                        st.write(f"**One-Sample Wilcoxon Signed Rank Test Summary ({g})**")
                        display_table = table.copy()
                        display_table["Value"] = display_table["Value"].map(
                            lambda v: f"{v:.3f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else v
                        )
                        st.dataframe(display_table, hide_index=True)
                else:
                    res_df = pd.DataFrame(results)
                    st.dataframe(res_df.style.apply(lambda x: ['background-color: #d4edda' if "💥" in x['Result'] else '' for i in x], axis=1))
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=x_col, y=y_col, data=df_clean, hue=x_col, palette="coolwarm", legend=False, ax=ax)
                ax.axhline(ref_value, color='red', linestyle='--', linewidth=2)
                st.pyplot(fig)

            # --- ANOVA / KRUSKAL ---
            elif method_key in ['anova', 'kruskal']:
                groups = [y_data[x_data == g] for g in x_data.unique()]
                if method_key == 'anova':
                    # --- Homogeneity of Variances (Levene/Brown-Forsythe) ---
                    g_labels = list(x_data.unique())
                    g_values = [y_data[x_data == g].to_numpy() for g in g_labels]
                    n_total = int(np.sum([len(g) for g in g_values]))
                    k_groups = len(g_values)

                    lev_mean_stat, lev_mean_p = stats.levene(*g_values, center='mean')
                    lev_med_stat, lev_med_p = stats.levene(*g_values, center='median')
                    lev_trim_stat, lev_trim_p = stats.levene(*g_values, center='trimmed', proportiontocut=0.05)

                    # Adjusted df (Brown-Forsythe with Welch correction) on absolute deviations from median
                    med_centers = [np.median(g) for g in g_values]
                    z_groups = [np.abs(g - med_centers[i]) for i, g in enumerate(g_values)]

                    # 2. Calculate Welch-Satterthwaite df for the deviations (SPSS Style)
                    z_vars = [np.var(g, ddof=1) for g in z_groups]
                    z_ns = [len(g) for g in z_groups]
                    numerator = sum((n - 1) * v for n, v in zip(z_ns, z_vars))**2
                    denominator = sum(((n - 1) * v)**2 / (n - 1) for n, v in zip(z_ns, z_vars))
                    bf_df1 = k_groups - 1 
                    bf_df2 = numerator / denominator
                    bf_stat, _ = stats.f_oneway(*z_groups)
                    bf_p = 1 - stats.f.cdf(bf_stat, k_groups - 1, bf_df2)

                    homogeneity_table = pd.DataFrame([
                        {
                            "": "Based on Mean",
                            "Levene Statistic": lev_mean_stat,
                            "df1": k_groups - 1,
                            "df2": n_total - k_groups,
                            "Sig.": lev_mean_p,
                        },
                        {
                            "": "Based on Median",
                            "Levene Statistic": lev_med_stat,
                            "df1": k_groups - 1,
                            "df2": n_total - k_groups,
                            "Sig.": lev_med_p,
                        },
                        {
                            "": "Based on Median and with adjusted df",
                            "Levene Statistic": bf_stat,
                            "df1": bf_df1,
                            "df2": bf_df2,
                            "Sig.": bf_p,
                        },
                        {
                            "": "Based on trimmed mean",
                            "Levene Statistic": lev_trim_stat,
                            "df1": k_groups - 1,
                            "df2": n_total - k_groups,
                            "Sig.": lev_trim_p,
                        },
                    ])

                    homogeneity_display = homogeneity_table.copy()
                    for col in ["Levene Statistic", "df1", "df2", "Sig."]:
                        homogeneity_display[col] = homogeneity_display[col].map(
                            lambda v: f"{v:.3f}" if pd.notna(v) else "-"
                        )

                    st.write("**Tests of Homogeneity of Variances**")
                    st.dataframe(homogeneity_display, hide_index=True)
                    log_step("ANOVA", "Homogeneity of variances computed", value=homogeneity_table)

                    stat, p = stats.f_oneway(*groups)
                else:
                    stat, p = stats.kruskal(*groups)

                log_step("Group Comparison", f"Statistic={stat:.4f}, P={p:.4e}")
                log_step("Group Comparison", "Group sizes", value=[len(g) for g in groups])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Statistic", f"{stat:.4f}")
                c2.metric("P-Value", f"{p:.4e}")
                if p < 0.05: c3.success("Significant Difference")
                else: c3.warning("No Difference")

                if method_key == 'anova':
                    g_labels = list(x_data.unique())
                    g_values = [y_data[x_data == g].to_numpy() for g in g_labels]
                    n_total = int(np.sum([len(g) for g in g_values]))
                    k_groups = len(g_values)
                    overall_mean = float(np.mean(y_data))
                    group_means = [float(np.mean(g)) for g in g_values]

                    ss_between = float(np.sum([len(g_values[i]) * (group_means[i] - overall_mean) ** 2 for i in range(k_groups)]))
                    ss_within = float(np.sum([np.sum((g_values[i] - group_means[i]) ** 2) for i in range(k_groups)]))
                    ss_total = ss_between + ss_within

                    df_between = k_groups - 1
                    df_within = n_total - k_groups
                    df_total = n_total - 1
                    ms_between = ss_between / df_between if df_between > 0 else np.nan
                    ms_within = ss_within / df_within if df_within > 0 else np.nan
                    f_stat = ms_between / ms_within if ms_within > 0 else np.nan
                    p_val = 1 - stats.f.cdf(f_stat, df_between, df_within) if np.isfinite(f_stat) else 1.0

                    anova_table = pd.DataFrame([
                        {
                            "": "Between Groups",
                            "Sum of Squares": ss_between,
                            "df": df_between,
                            "Mean Square": ms_between,
                            "F": f_stat,
                            "Sig.": p_val,
                        },
                        {
                            "": "Within Groups",
                            "Sum of Squares": ss_within,
                            "df": df_within,
                            "Mean Square": ms_within,
                            "F": "",
                            "Sig.": "",
                        },
                        {
                            "": "Total",
                            "Sum of Squares": ss_total,
                            "df": df_total,
                            "Mean Square": "",
                            "F": "",
                            "Sig.": "",
                        },
                    ])

                    anova_display = anova_table.copy()
                    for col in ["Sum of Squares", "Mean Square", "F", "Sig."]:
                        anova_display[col] = anova_display[col].map(
                            lambda v: f"{v:.3f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else v
                        )
                    anova_display["df"] = anova_display["df"].map(
                        lambda v: f"{int(v)}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else v
                    )

                    st.write("**ANOVA**")
                    st.dataframe(anova_display, hide_index=True)
                    log_step("ANOVA", "ANOVA table computed", value=anova_table)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=x_col, y=y_col, data=df_clean, hue=x_col, palette="viridis", legend=False, ax=ax)
                st.pyplot(fig)

                if p < 0.05:
                    st.divider()
                    st.subheader("2. Post-Hoc Analysis")
                    if method_key == 'anova':
                        equal_var = homogeneity_table.loc[homogeneity_table[""] == "Based on Mean", "Sig."].iloc[0] >= 0.05

                        tukey = pairwise_tukeyhsd(endog=y_data, groups=x_data, alpha=0.05)
                        tukey_df = pd.DataFrame(
                            tukey.summary().data[1:],
                            columns=tukey.summary().data[0],
                        )
                        tukey_df.rename(
                            columns={
                                "group1": "(I) Group",
                                "group2": "(J) Group",
                                "meandiff": "Mean Difference (I-J)",
                                "p-adj": "Sig.",
                                "lower": "Lower Bound",
                                "upper": "Upper Bound",
                            },
                            inplace=True,
                        )

                        tukey_display = tukey_df.copy()
                        for col in ["Mean Difference (I-J)", "Sig.", "Lower Bound", "Upper Bound"]:
                            tukey_display[col] = tukey_display[col].map(
                                lambda v: f"{v:.3f}" if pd.notna(v) else "-"
                            )

                        st.write("**Multiple Comparisons (Tukey HSD)**")
                        if not equal_var:
                            st.caption("Variances unequal: Tukey shown for reference; Games-Howell is preferred.")
                        st.dataframe(tukey_display, hide_index=True)
                        st.pyplot(tukey.plot_simultaneous())
                        log_step("Post-Hoc", "Tukey HSD computed", value=tukey_df)

                        if not equal_var:
                            g_labels = list(x_data.unique())
                            g_values = [y_data[x_data == g].to_numpy() for g in g_labels]
                            k_groups = len(g_labels)

                            means = {g_labels[i]: float(np.mean(g_values[i])) for i in range(k_groups)}
                            variances = {g_labels[i]: float(np.var(g_values[i], ddof=1)) for i in range(k_groups)}
                            ns = {g_labels[i]: int(len(g_values[i])) for i in range(k_groups)}

                            rows = []
                            alpha = 0.05
                            for i in range(k_groups):
                                for j in range(i + 1, k_groups):
                                    g1 = g_labels[i]
                                    g2 = g_labels[j]
                                    mean_diff = means[g1] - means[g2]
                                    se_mean = np.sqrt(variances[g1] / ns[g1] + variances[g2] / ns[g2])
                                    se_q = se_mean * np.sqrt(0.5)
                                    df_num = (variances[g1] / ns[g1] + variances[g2] / ns[g2]) ** 2
                                    df_den = ((variances[g1] / ns[g1]) ** 2) / (ns[g1] - 1) + ((variances[g2] / ns[g2]) ** 2) / (ns[g2] - 1)
                                    df = df_num / df_den if df_den > 0 else np.nan

                                    q = abs(mean_diff) / se_q if se_q > 0 else 0.0
                                    p_val = stats.studentized_range.sf(q, k_groups, df) if np.isfinite(df) else 1.0
                                    q_crit = stats.studentized_range.ppf(1 - alpha, k_groups, df) if np.isfinite(df) else np.nan
                                    margin = q_crit * se_q if np.isfinite(q_crit) else np.nan
                                    ci_low = mean_diff - margin if np.isfinite(margin) else np.nan
                                    ci_high = mean_diff + margin if np.isfinite(margin) else np.nan

                                    rows.append({
                                        "(I) Group": g1,
                                        "(J) Group": g2,
                                        "Mean Difference (I-J)": mean_diff,
                                        "Std. Error": se_mean,
                                        "Sig.": p_val,
                                        "Lower Bound": ci_low,
                                        "Upper Bound": ci_high,
                                    })

                            gh_df = pd.DataFrame(rows)
                            gh_display = gh_df.copy()
                            for col in ["Mean Difference (I-J)", "Std. Error", "Sig.", "Lower Bound", "Upper Bound"]:
                                gh_display[col] = gh_display[col].map(
                                    lambda v: f"{v:.3f}" if pd.notna(v) else "-"
                                )

                            st.write("**Multiple Comparisons (Games-Howell)**")
                            st.dataframe(gh_display, hide_index=True)
                            st.write("**Mean Difference Confidence Interval Plot (Games-Howell)**")

                            gh_plot_df = gh_df.copy()
                            gh_plot_df["Comparison"] = gh_plot_df["(I) Group"].astype(str) + " - " + gh_plot_df["(J) Group"].astype(str)
                            gh_plot_df = gh_plot_df.sort_values("Mean Difference (I-J)")

                            fig_h = max(3.5, 0.4 * len(gh_plot_df))
                            fig, ax = plt.subplots(figsize=(8, fig_h))
                            y_pos = np.arange(len(gh_plot_df))

                            ax.errorbar(
                                gh_plot_df["Mean Difference (I-J)"],
                                y_pos,
                                xerr=[
                                    gh_plot_df["Mean Difference (I-J)"] - gh_plot_df["Lower Bound"],
                                    gh_plot_df["Upper Bound"] - gh_plot_df["Mean Difference (I-J)"]
                                ],
                                fmt='o',
                                color='#1f77b4',
                                ecolor='#1f77b4',
                                elinewidth=2,
                                capsize=3,
                            )
                            ax.axvline(0, color='gray', linestyle='--', linewidth=1)
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(gh_plot_df["Comparison"])
                            ax.set_xlabel("Mean Difference (I-J) with 95% CI")
                            ax.set_title("Games-Howell Mean Differences")
                            ax.invert_yaxis()
                            ax.grid(axis='x', linestyle=':', alpha=0.6)
                            st.pyplot(fig)
                            log_step("Post-Hoc", "Games-Howell computed", value=gh_df)
                    else:
                        unique_groups = x_data.unique()
                        pairs = list(itertools.combinations(unique_groups, 2))
                        num_pairs = len(pairs)

                        values = y_data.to_numpy()
                        ranks = stats.rankdata(values, method="average")
                        n_total = len(values)
                        _, tie_counts = np.unique(values, return_counts=True)
                        tie_sum = np.sum(tie_counts**3 - tie_counts)
                        tie_correction = (tie_sum / (12 * (n_total - 1))) if n_total > 1 else 0.0
                        rank_var = (n_total * (n_total + 1)) / 12 - tie_correction

                        mean_ranks = {}
                        group_sizes = {}
                        for g in unique_groups:
                            mask = (x_data == g).to_numpy()
                            mean_ranks[g] = float(np.mean(ranks[mask]))
                            group_sizes[g] = int(np.sum(mask))

                        results_rows = []
                        p_matrix_raw = pd.DataFrame(index=unique_groups, columns=unique_groups, dtype=float)
                        p_matrix_display = pd.DataFrame(index=unique_groups, columns=unique_groups, dtype=float)
                        for g1, g2 in pairs:
                            n1 = group_sizes[g1]
                            n2 = group_sizes[g2]
                            diff = mean_ranks[g1] - mean_ranks[g2]
                            se = np.sqrt(rank_var * (1.0 / n1 + 1.0 / n2)) if rank_var > 0 else np.nan
                            z = diff / se if np.isfinite(se) and se > 0 else np.nan
                            p_val = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else 1.0
                            p_adj = min(p_val * num_pairs, 1.0)
                            p_adj_display = float(f"{p_adj:.3f}")

                            results_rows.append({
                                "Sample 1-Sample 2": f"{g1}-{g2}",
                                "Test Statistic": diff,
                                "Std. Error": se,
                                "Std. Test Statistic": z,
                                "Sig.": p_val,
                                "Adj. Sig.": p_adj,
                            })

                            p_matrix_raw.at[g1, g2] = p_adj
                            p_matrix_raw.at[g2, g1] = p_adj
                            p_matrix_display.at[g1, g2] = p_adj_display
                            p_matrix_display.at[g2, g1] = p_adj_display

                        p_matrix_raw.fillna(1.0, inplace=True)
                        p_matrix_display.fillna(1.0, inplace=True)

                        results_df = pd.DataFrame(results_rows)
                        results_df_display = results_df.copy()
                        for col in ["Test Statistic", "Std. Error", "Std. Test Statistic", "Sig.", "Adj. Sig."]:
                            results_df_display[col] = results_df_display[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")

                        st.write("**Pairwise Comparisons (Dunn's Test - Global Ranks):**")
                        st.dataframe(results_df_display, hide_index=True)
                        st.write(f"**Note:** P-values are Bonferroni-corrected for {num_pairs} comparisons.")

                        fig, ax = plt.subplots()
                        sns.heatmap(p_matrix_display, annot=True, cmap="coolwarm_r", vmin=0, vmax=0.05, ax=ax)
                        st.pyplot(fig)

                        st.write("**Pairwise Comparison Network (Adj. Sig.)**")
                        groups_list = list(p_matrix_display.index)
                        n_groups = len(groups_list)
                        angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False)
                        positions = {
                            groups_list[i]: (np.cos(angles[i]), np.sin(angles[i]))
                            for i in range(n_groups)
                        }

                        fig_net, ax_net = plt.subplots(figsize=(6, 6))
                        ax_net.set_title(f"Pairwise Comparisons of {y_col}")

                        for i in range(n_groups):
                            for j in range(i + 1, n_groups):
                                g1 = groups_list[i]
                                g2 = groups_list[j]
                                p_adj_disp = float(p_matrix_display.at[g1, g2]) if pd.notna(p_matrix_display.at[g1, g2]) else 1.0
                                color = "#2b83ba" if p_adj_disp <= 0.05 else "#c51b7d"
                                x1, y1 = positions[g1]
                                x2, y2 = positions[g2]
                                ax_net.plot([x1, x2], [y1, y2], color=color, linewidth=1.8, alpha=0.9)

                        for g in groups_list:
                            x, y = positions[g]
                            ax_net.scatter([x], [y], s=220, color="#8ecae6", edgecolor="#023047", zorder=3)
                            label = f"{g}\n{mean_ranks[g]:.2f}"
                            ax_net.text(x, y, label, ha="center", va="center", fontsize=8, zorder=4)

                        legend_items = [
                            Line2D([0], [0], color="#2b83ba", lw=2, label="< 0.05"),
                            Line2D([0], [0], color="#c51b7d", lw=2, label=">= 0.05"),
                        ]
                        ax_net.legend(handles=legend_items, title="Adj. Sig.", loc="upper right")
                        ax_net.set_aspect("equal")
                        ax_net.set_xlim(-1.2, 1.2)
                        ax_net.set_ylim(-1.2, 1.2)
                        ax_net.axis("off")
                        st.pyplot(fig_net)
                        st.caption("Each node shows the sample average rank of the selected dependent variable.")

                        log_step(
                            "Post-Hoc",
                            f"Dunn's test computed (Bonferroni n={num_pairs})",
                            value=results_df,
                        )

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