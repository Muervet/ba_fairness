import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Fairness in Risk Assessment", layout="wide")

# Initialize session state for page navigation
if 'show_fairness_analysis' not in st.session_state:
    st.session_state.show_fairness_analysis = False


# Toggle function for fairness analysis
def toggle_fairness_analysis():
    st.session_state.show_fairness_analysis = not st.session_state.show_fairness_analysis


# Title and description
st.title("Fairness in Criminal Risk Assessment")

# Add button for fairness analysis at the top
if not st.session_state.show_fairness_analysis:
    if st.button("🔍 Show Fairness Criteria Analysis", type="primary", use_container_width=True):
        toggle_fairness_analysis()
        st.rerun()
else:
    if st.button("⬅️ Back to Main Analysis", use_container_width=True):
        toggle_fairness_analysis()
        st.rerun()


# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv('datasets/compas-scores-two-years.csv', sep=',')

    # Create binary target
    def create_binary_target(score_text):
        if pd.isna(score_text):
            return np.nan
        score_text = str(score_text).lower().strip()
        if 'low' in score_text:
            return 0  # Low risk
        else:
            return 1  # Medium/High risk

    data['target_binary'] = data['score_text'].apply(create_binary_target)
    data = data.dropna(subset=['target_binary'])

    # Keep only African-American and Caucasian for fairness analysis
    # Other races will be filtered out
    return data


data = load_data()

# FAIRNESS ANALYSIS PAGE
if st.session_state.show_fairness_analysis:
    st.header("Fairness Criteria Analysis: African-American vs Caucasian")

    # Create two comparison groups - SADECE African-American ve Caucasian
    data_african_american = data[data['race'] == 'African-American'].copy()
    data_caucasian = data[data['race'] == 'Caucasian'].copy()

    # Filter out other races for this analysis
    data_african_american = data_african_american.copy()
    data_caucasian = data_caucasian.copy()

    # Display sample sizes
    st.sidebar.header("Comparison Groups")
    st.sidebar.info(f"""
    **Group 1 - African-American:**
    - Samples: {len(data_african_american):,}
    - Positive cases: {data_african_american['target_binary'].sum():,} ({data_african_american['target_binary'].mean():.1%})

    **Group 2 - Caucasian:**
    - Samples: {len(data_caucasian):,}
    - Positive cases: {data_caucasian['target_binary'].sum():,} ({data_caucasian['target_binary'].mean():.1%})
    """)

    # Sidebar for fairness controls
    with st.sidebar:
        st.header("Fairness Controls")

        # Fairness criterion selection
        fairness_criterion = st.radio(
            "Choose Fairness Criterion:",
            ["Independence (Demographic Parity)",
             "Separation (Equalized Odds)",
             "Sufficiency (Calibration)"],
            help="""Select which fairness criterion to optimize:
            - Independence: Equal positive prediction rates across groups
            - Separation: Equal TPR and FPR across groups  
            - Sufficiency: Same score means same actual risk probability across groups"""
        )

        # Trade-off slider for balanced optimization
        fairness_weight = st.slider(
            "Fairness vs Practicality Balance:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="0.0 = Maximize practicality (balanced thresholds), 1.0 = Maximize fairness (may give extreme thresholds)"
        )

        st.markdown("---")
        st.markdown("### Fairness Definitions")
        st.markdown("""
        **Independence**: Both groups have same percentage classified as high risk
        \n**Separation**: Equal true positive rates and false positive rates
        \n**Sufficiency**: Same risk score predicts same actual risk probability
        """)


    # Prepare features function
    def prepare_features(df):
        features_to_drop = [
            'target_binary', 'score_text', 'decile_score', 'v_decile_score',
            'v_score_text', 'decile_score.1', 'id', 'violent_recid',
            'r_days_from_arrest', 'days_b_screening_arrest', 'name',
            'first', 'last', 'sex', 'dob', 'age', 'age_cat', 'race'
        ]

        features_to_drop = [col for col in features_to_drop if col in df.columns]
        X = df.drop(features_to_drop, axis=1, errors='ignore')
        y = df['target_binary']

        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) > 0:
            imputer_num = SimpleImputer(strategy='median')
            X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])

            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        return X, y


    # Train models for both groups
    @st.cache_resource
    def train_model_for_group(df, group_name):
        X, y = prepare_features(df)

        if len(X) < 30:
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            class_weight='balanced',
            min_samples_split=5,
            min_samples_leaf=2
        )

        classifier.fit(X_train, y_train)
        y_scores = classifier.predict_proba(X_test)[:, 1]

        return {
            'X_test': X_test,
            'y_test': y_test,
            'y_scores': y_scores,
            'model': classifier
        }


    # Calculate metrics for different thresholds
    def calculate_metrics_for_thresholds(y_test, y_scores):
        thresholds = np.linspace(0, 1, 101)
        metrics = []

        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)

            tp = np.sum((y_pred == 1) & (y_test == 1))
            fp = np.sum((y_pred == 1) & (y_test == 0))
            tn = np.sum((y_pred == 0) & (y_test == 0))
            fn = np.sum((y_pred == 0) & (y_test == 1))

            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            positive_rate = np.mean(y_pred)
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

            metrics.append({
                'threshold': thresh,
                'accuracy': accuracy,
                'tpr': tpr,
                'fpr': fpr,
                'precision': precision,
                'positive_rate': positive_rate,
                'f1': f1
            })

        return pd.DataFrame(metrics)


    # Calculate calibration metrics
    def calculate_calibration_metrics(y_test, y_scores, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_scores, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        calibration_data = []
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                avg_score = np.mean(y_scores[mask])
                actual_pos_rate = np.mean(y_test[mask])
                calibration_data.append({
                    'bin': i,
                    'avg_score': avg_score,
                    'actual_pos_rate': actual_pos_rate,
                    'count': np.sum(mask)
                })
        return pd.DataFrame(calibration_data)


    # SMART OPTIMIZATION FUNCTION (with practicality constraint)
    def find_optimal_threshold_smart(metrics_df1, metrics_df2, calibration_df1, calibration_df2,
                                     criterion, fairness_weight=0.7):
        thresholds = metrics_df1['threshold'].values

        if criterion == "Independence (Demographic Parity)":
            # 1. Fairness component: minimize positive rate difference
            diff_pos_rate = np.abs(metrics_df1['positive_rate'].values - metrics_df2['positive_rate'].values)
            fairness_component = diff_pos_rate

            # 2. Practicality component: prefer thresholds around 0.3-0.7
            practicality = np.abs(thresholds - 0.5)  # 0 is best at 0.5

            # 3. Performance component: maximize average accuracy
            avg_accuracy = (metrics_df1['accuracy'].values + metrics_df2['accuracy'].values) / 2
            performance_component = 1 - avg_accuracy  # We want to minimize this

            # Combine with weights
            combined_score = (fairness_weight * fairness_component +
                              (1 - fairness_weight) * (0.6 * practicality + 0.4 * performance_component))

            idx = np.argmin(combined_score)
            optimal_thresh = thresholds[idx]

            # Ensure it's not too extreme
            if optimal_thresh < 0.1 or optimal_thresh > 0.9:
                # Find best threshold in reasonable range
                reasonable_mask = (thresholds >= 0.3) & (thresholds <= 0.7)
                if np.any(reasonable_mask):
                    reasonable_scores = combined_score[reasonable_mask]
                    reasonable_thresholds = thresholds[reasonable_mask]
                    idx_reasonable = np.argmin(reasonable_scores)
                    optimal_thresh = reasonable_thresholds[idx_reasonable]
                    reason = "Minimizes positive rate difference with practical constraint"
                else:
                    reason = "Minimizes positive rate difference (extreme threshold warning)"
            else:
                reason = "Minimizes positive rate difference"

            return optimal_thresh, reason

        elif criterion == "Separation (Equalized Odds)":
            # Minimize differences in TPR and FPR
            diff_tpr = np.abs(metrics_df1['tpr'].values - metrics_df2['tpr'].values)
            diff_fpr = np.abs(metrics_df1['fpr'].values - metrics_df2['fpr'].values)
            fairness_component = (diff_tpr + diff_fpr) / 2

            # Practicality component
            practicality = np.abs(thresholds - 0.5)

            # Performance component
            avg_accuracy = (metrics_df1['accuracy'].values + metrics_df2['accuracy'].values) / 2
            performance_component = 1 - avg_accuracy

            # Combine
            combined_score = (fairness_weight * fairness_component +
                              (1 - fairness_weight) * (0.6 * practicality + 0.4 * performance_component))

            idx = np.argmin(combined_score)
            optimal_thresh = thresholds[idx]

            # Ensure reasonable
            if optimal_thresh < 0.1 or optimal_thresh > 0.9:
                reasonable_mask = (thresholds >= 0.3) & (thresholds <= 0.7)
                if np.any(reasonable_mask):
                    reasonable_scores = combined_score[reasonable_mask]
                    reasonable_thresholds = thresholds[reasonable_mask]
                    idx_reasonable = np.argmin(reasonable_scores)
                    optimal_thresh = reasonable_thresholds[idx_reasonable]
                    reason = "Minimizes TPR/FPR differences with practical constraint"
                else:
                    reason = "Minimizes TPR/FPR differences (extreme threshold warning)"
            else:
                reason = "Minimizes TPR and FPR differences"

            return optimal_thresh, reason

        else:  # Sufficiency
            # Minimize calibration differences
            cal_errors = []
            for i, thresh in enumerate(thresholds):
                ppv1 = metrics_df1.iloc[i]['precision']
                ppv2 = metrics_df2.iloc[i]['precision']
                cal_diff = np.abs(ppv1 - ppv2)
                cal_errors.append(cal_diff)

            fairness_component = np.array(cal_errors)

            # Practicality
            practicality = np.abs(thresholds - 0.5)

            # Performance
            avg_accuracy = (metrics_df1['accuracy'].values + metrics_df2['accuracy'].values) / 2
            performance_component = 1 - avg_accuracy

            # Combine
            combined_score = (fairness_weight * fairness_component +
                              (1 - fairness_weight) * (0.6 * practicality + 0.4 * performance_component))

            idx = np.argmin(combined_score)
            optimal_thresh = thresholds[idx]

            # Ensure reasonable
            if optimal_thresh < 0.1 or optimal_thresh > 0.9:
                reasonable_mask = (thresholds >= 0.3) & (thresholds <= 0.7)
                if np.any(reasonable_mask):
                    reasonable_scores = combined_score[reasonable_mask]
                    reasonable_thresholds = thresholds[reasonable_mask]
                    idx_reasonable = np.argmin(reasonable_scores)
                    optimal_thresh = reasonable_thresholds[idx_reasonable]
                    reason = "Minimizes calibration differences with practical constraint"
                else:
                    reason = "Minimizes calibration differences (extreme threshold warning)"
            else:
                reason = "Minimizes calibration differences"

            return optimal_thresh, reason


    # Train models
    with st.spinner("Training models for African-American and Caucasian..."):
        model_aa = train_model_for_group(data_african_american, "African-American")
        model_caucasian = train_model_for_group(data_caucasian, "Caucasian")

    if model_aa is None or model_caucasian is None:
        st.error("Insufficient data for one or both groups.")
        if st.button("Return to Main Analysis"):
            toggle_fairness_analysis()
            st.rerun()
    else:
        # Calculate metrics
        metrics_aa = calculate_metrics_for_thresholds(model_aa['y_test'], model_aa['y_scores'])
        metrics_caucasian = calculate_metrics_for_thresholds(model_caucasian['y_test'], model_caucasian['y_scores'])

        # Calculate calibration
        calibration_aa = calculate_calibration_metrics(model_aa['y_test'], model_aa['y_scores'])
        calibration_caucasian = calculate_calibration_metrics(model_caucasian['y_test'], model_caucasian['y_scores'])

        # Find optimal threshold
        optimal_thresh, reason = find_optimal_threshold_smart(
            metrics_aa, metrics_caucasian, calibration_aa, calibration_caucasian,
            fairness_criterion, fairness_weight
        )
        optimal_decile = max(1, min(10, int(np.ceil(optimal_thresh * 10))))

        # DISPLAY OPTIMAL THRESHOLD AND SLIDER
        st.subheader("Threshold Control")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Selected Fairness", fairness_criterion.split('(')[0].strip())
        with col2:
            st.metric("Optimal Threshold", f"Decile {optimal_decile}")
        with col3:
            st.metric("Probability", f"{optimal_thresh:.3f}")

        st.info(f"**Optimization Goal:** {reason}")

        # THRESHOLD SLIDER (like in main page)
        threshold = st.slider(
            "Select Risk Threshold (Decile Score):",
            min_value=1,
            max_value=10,
            value=optimal_decile,  # Default to optimal
            step=1,
            help="Scores at or above this threshold will be classified as 'High Risk'"
        )

        # Convert decile to probability for calculations
        threshold_prob = threshold / 10.0

        # Create visualization with current threshold
        st.subheader(f"Comparison at Threshold: Decile {threshold}")

        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Performance Metrics", "ROC Curves", "Calibration"])

        with tab1:
            # Find metrics at current threshold
            idx_aa = np.argmin(np.abs(metrics_aa['threshold'] - threshold_prob))
            idx_caucasian = np.argmin(np.abs(metrics_caucasian['threshold'] - threshold_prob))

            metrics_at_threshold = pd.DataFrame({
                'Metric': ['Accuracy', 'TPR (Recall)', 'FPR', 'Precision (PPV)', 'Positive Rate', 'F1 Score'],
                'African-American': [
                    f"{metrics_aa.iloc[idx_aa]['accuracy']:.3f}",
                    f"{metrics_aa.iloc[idx_aa]['tpr']:.3f}",
                    f"{metrics_aa.iloc[idx_aa]['fpr']:.3f}",
                    f"{metrics_aa.iloc[idx_aa]['precision']:.3f}",
                    f"{metrics_aa.iloc[idx_aa]['positive_rate']:.3f}",
                    f"{metrics_aa.iloc[idx_aa]['f1']:.3f}"
                ],
                'Caucasian': [
                    f"{metrics_caucasian.iloc[idx_caucasian]['accuracy']:.3f}",
                    f"{metrics_caucasian.iloc[idx_caucasian]['tpr']:.3f}",
                    f"{metrics_caucasian.iloc[idx_caucasian]['fpr']:.3f}",
                    f"{metrics_caucasian.iloc[idx_caucasian]['precision']:.3f}",
                    f"{metrics_caucasian.iloc[idx_caucasian]['positive_rate']:.3f}",
                    f"{metrics_caucasian.iloc[idx_caucasian]['f1']:.3f}"
                ],
                'Difference': [
                    f"{abs(metrics_aa.iloc[idx_aa]['accuracy'] - metrics_caucasian.iloc[idx_caucasian]['accuracy']):.3f}",
                    f"{abs(metrics_aa.iloc[idx_aa]['tpr'] - metrics_caucasian.iloc[idx_caucasian]['tpr']):.3f}",
                    f"{abs(metrics_aa.iloc[idx_aa]['fpr'] - metrics_caucasian.iloc[idx_caucasian]['fpr']):.3f}",
                    f"{abs(metrics_aa.iloc[idx_aa]['precision'] - metrics_caucasian.iloc[idx_caucasian]['precision']):.3f}",
                    f"{abs(metrics_aa.iloc[idx_aa]['positive_rate'] - metrics_caucasian.iloc[idx_caucasian]['positive_rate']):.3f}",
                    f"{abs(metrics_aa.iloc[idx_aa]['f1'] - metrics_caucasian.iloc[idx_caucasian]['f1']):.3f}"
                ]
            })


            # Color coding based on selected fairness criterion
            def highlight_fairness_metric(row):
                styles = [''] * len(row)
                if fairness_criterion == "Independence (Demographic Parity)" and row.name == 4:  # Positive Rate
                    styles[-1] = 'background-color: #ffd700; font-weight: bold'
                elif fairness_criterion == "Separation (Equalized Odds)" and row.name in [1, 2]:  # TPR and FPR
                    styles[-1] = 'background-color: #ffd700; font-weight: bold'
                elif fairness_criterion == "Sufficiency (Calibration)" and row.name == 3:  # Precision
                    styles[-1] = 'background-color: #ffd700; font-weight: bold'
                return styles


            styled_df = metrics_at_threshold.style.apply(highlight_fairness_metric, axis=1)
            st.dataframe(styled_df, use_container_width=True)

            # Visualization of metrics across thresholds
            fig_metrics = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Accuracy", "Positive Rate", "TPR", "FPR"),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )

            color_aa = '#1f77b4'
            color_caucasian = '#ff7f0e'

            # Accuracy
            fig_metrics.add_trace(
                go.Scatter(x=metrics_aa['threshold'], y=metrics_aa['accuracy'],
                           mode='lines', name='African-American',
                           line=dict(color=color_aa, width=2)),
                row=1, col=1
            )
            fig_metrics.add_trace(
                go.Scatter(x=metrics_caucasian['threshold'], y=metrics_caucasian['accuracy'],
                           mode='lines', name='Caucasian',
                           line=dict(color=color_caucasian, width=2)),
                row=1, col=1
            )

            # Positive Rate
            fig_metrics.add_trace(
                go.Scatter(x=metrics_aa['threshold'], y=metrics_aa['positive_rate'],
                           mode='lines', showlegend=False,
                           line=dict(color=color_aa, width=2)),
                row=1, col=2
            )
            fig_metrics.add_trace(
                go.Scatter(x=metrics_caucasian['threshold'], y=metrics_caucasian['positive_rate'],
                           mode='lines', showlegend=False,
                           line=dict(color=color_caucasian, width=2)),
                row=1, col=2
            )

            # TPR
            fig_metrics.add_trace(
                go.Scatter(x=metrics_aa['threshold'], y=metrics_aa['tpr'],
                           mode='lines', showlegend=False,
                           line=dict(color=color_aa, width=2)),
                row=2, col=1
            )
            fig_metrics.add_trace(
                go.Scatter(x=metrics_caucasian['threshold'], y=metrics_caucasian['tpr'],
                           mode='lines', showlegend=False,
                           line=dict(color=color_caucasian, width=2)),
                row=2, col=1
            )

            # FPR
            fig_metrics.add_trace(
                go.Scatter(x=metrics_aa['threshold'], y=metrics_aa['fpr'],
                           mode='lines', showlegend=False,
                           line=dict(color=color_aa, width=2)),
                row=2, col=2
            )
            fig_metrics.add_trace(
                go.Scatter(x=metrics_caucasian['threshold'], y=metrics_caucasian['fpr'],
                           mode='lines', showlegend=False,
                           line=dict(color=color_caucasian, width=2)),
                row=2, col=2
            )

            # Add vertical line at current threshold to all subplots
            for row in [1, 2]:
                for col in [1, 2]:
                    fig_metrics.add_vline(
                        x=threshold_prob, line_dash="dash",
                        line_color="red", line_width=2,
                        row=row, col=col,
                        annotation_text=f"Current: {threshold_prob:.2f}",
                        annotation_position="top right"
                    )

            # Add vertical line at optimal threshold
            for row in [1, 2]:
                for col in [1, 2]:
                    fig_metrics.add_vline(
                        x=optimal_thresh, line_dash="dot",
                        line_color="green", line_width=2,
                        row=row, col=col,
                        annotation_text=f"Optimal: {optimal_thresh:.2f}",
                        annotation_position="bottom right"
                    )

            fig_metrics.update_layout(
                height=600,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )

            fig_metrics.update_xaxes(title_text="Threshold", row=1, col=1)
            fig_metrics.update_yaxes(title_text="Accuracy", row=1, col=1)
            fig_metrics.update_xaxes(title_text="Threshold", row=1, col=2)
            fig_metrics.update_yaxes(title_text="Positive Rate", row=1, col=2)
            fig_metrics.update_xaxes(title_text="Threshold", row=2, col=1)
            fig_metrics.update_yaxes(title_text="TPR", row=2, col=1)
            fig_metrics.update_xaxes(title_text="Threshold", row=2, col=2)
            fig_metrics.update_yaxes(title_text="FPR", row=2, col=2)

            st.plotly_chart(fig_metrics, use_container_width=True)

        with tab2:
            # ROC Curves
            from sklearn.metrics import roc_curve, auc

            fpr_aa, tpr_aa, _ = roc_curve(model_aa['y_test'], model_aa['y_scores'])
            fpr_caucasian, tpr_caucasian, _ = roc_curve(model_caucasian['y_test'], model_caucasian['y_scores'])
            auc_aa = auc(fpr_aa, tpr_aa)
            auc_caucasian = auc(fpr_caucasian, tpr_caucasian)

            # Find points on ROC curve at current threshold
            idx_aa_roc = np.argmin(np.abs(metrics_aa['threshold'] - threshold_prob))
            idx_caucasian_roc = np.argmin(np.abs(metrics_caucasian['threshold'] - threshold_prob))

            fig_roc = go.Figure()

            fig_roc.add_trace(
                go.Scatter(x=fpr_aa, y=tpr_aa, mode='lines',
                           name=f'African-American (AUC={auc_aa:.3f})',
                           line=dict(color=color_aa, width=3))
            )
            fig_roc.add_trace(
                go.Scatter(x=fpr_caucasian, y=tpr_caucasian, mode='lines',
                           name=f'Caucasian (AUC={auc_caucasian:.3f})',
                           line=dict(color=color_caucasian, width=3))
            )

            # Add current operating point for AA
            fig_roc.add_trace(
                go.Scatter(x=[metrics_aa.iloc[idx_aa_roc]['fpr']],
                           y=[metrics_aa.iloc[idx_aa_roc]['tpr']],
                           mode='markers',
                           name=f'AA at Decile {threshold}',
                           marker=dict(size=15, color=color_aa, symbol='circle'),
                           hovertemplate=f"AA: FPR={metrics_aa.iloc[idx_aa_roc]['fpr']:.3f}, TPR={metrics_aa.iloc[idx_aa_roc]['tpr']:.3f}<br>Threshold: {threshold_prob:.3f}")
            )

            # Add current operating point for Caucasian
            fig_roc.add_trace(
                go.Scatter(x=[metrics_caucasian.iloc[idx_caucasian_roc]['fpr']],
                           y=[metrics_caucasian.iloc[idx_caucasian_roc]['tpr']],
                           mode='markers',
                           name=f'Caucasian at Decile {threshold}',
                           marker=dict(size=15, color=color_caucasian, symbol='square'),
                           hovertemplate=f"Caucasian: FPR={metrics_caucasian.iloc[idx_caucasian_roc]['fpr']:.3f}, TPR={metrics_caucasian.iloc[idx_caucasian_roc]['tpr']:.3f}<br>Threshold: {threshold_prob:.3f}")
            )

            fig_roc.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                           name='Random', line=dict(color='gray', dash='dash'))
            )

            fig_roc.update_layout(
                title=f"ROC Curves Comparison (Current: Decile {threshold})",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=500,
                showlegend=True
            )

            st.plotly_chart(fig_roc, use_container_width=True)

        with tab3:
            # Calibration plots
            fig_calibration = make_subplots(
                rows=1, cols=2,
                subplot_titles=("African-American", "Caucasian"),
                horizontal_spacing=0.15
            )

            perfect_line = np.linspace(0, 1, 100)

            # African-American calibration
            fig_calibration.add_trace(
                go.Scatter(x=calibration_aa['avg_score'], y=calibration_aa['actual_pos_rate'],
                           mode='markers+lines', name='African-American',
                           marker=dict(size=10, color=color_aa),
                           line=dict(color=color_aa, width=2)),
                row=1, col=1
            )
            fig_calibration.add_trace(
                go.Scatter(x=perfect_line, y=perfect_line,
                           mode='lines', name='Perfect',
                           line=dict(color='gray', dash='dash')),
                row=1, col=1
            )

            # Caucasian calibration
            fig_calibration.add_trace(
                go.Scatter(x=calibration_caucasian['avg_score'], y=calibration_caucasian['actual_pos_rate'],
                           mode='markers+lines', name='Caucasian',
                           marker=dict(size=10, color=color_caucasian),
                           line=dict(color=color_caucasian, width=2),
                           showlegend=False),
                row=1, col=2
            )
            fig_calibration.add_trace(
                go.Scatter(x=perfect_line, y=perfect_line,
                           mode='lines', name='Perfect',
                           line=dict(color='gray', dash='dash'),
                           showlegend=False),
                row=1, col=2
            )

            fig_calibration.update_layout(
                height=500,
                showlegend=True,
                title_text=f"Calibration Analysis (Current Threshold: {threshold_prob:.2f})"
            )

            fig_calibration.update_xaxes(title_text="Predicted Score", row=1, col=1)
            fig_calibration.update_yaxes(title_text="Actual Positive Rate", row=1, col=1)
            fig_calibration.update_xaxes(title_text="Predicted Score", row=1, col=2)
            fig_calibration.update_yaxes(title_text="Actual Positive Rate", row=1, col=2)

            st.plotly_chart(fig_calibration, use_container_width=True)

        # Fairness Analysis Summary
        st.markdown("---")
        st.subheader(f"Fairness Analysis Summary")

        # Calculate fairness metrics at current threshold
        pos_rate_diff = abs(
            metrics_aa.iloc[idx_aa]['positive_rate'] - metrics_caucasian.iloc[idx_caucasian]['positive_rate'])
        tpr_diff = abs(metrics_aa.iloc[idx_aa]['tpr'] - metrics_caucasian.iloc[idx_caucasian]['tpr'])
        fpr_diff = abs(metrics_aa.iloc[idx_aa]['fpr'] - metrics_caucasian.iloc[idx_caucasian]['fpr'])
        precision_diff = abs(metrics_aa.iloc[idx_aa]['precision'] - metrics_caucasian.iloc[idx_caucasian]['precision'])

        fairness_metrics = pd.DataFrame({
            'Criterion': ['Independence', 'Separation', 'Sufficiency'],
            'Key Metric': ['Positive Rate Difference', 'TPR & FPR Differences', 'Precision Difference'],
            'Current Value': [f"{pos_rate_diff:.3f}", f"{(tpr_diff + fpr_diff) / 2:.3f}", f"{precision_diff:.3f}"],
            'Interpretation': [
                "Lower is better (≤0.1 good)" if pos_rate_diff <= 0.1 else "High disparity (>0.1)",
                "Lower is better (≤0.1 good)" if (tpr_diff + fpr_diff) / 2 <= 0.1 else "High disparity (>0.1)",
                "Lower is better (≤0.1 good)" if precision_diff <= 0.1 else "High disparity (>0.1)"
            ]
        })

        st.dataframe(fairness_metrics, use_container_width=True)

        # Interpretation
        with st.expander("📊 How to Interpret These Results"):
            st.markdown(f"""
            ### At Threshold Decile {threshold}:

            **For {fairness_criterion.split('(')[0].strip()}:**
            - **Current fairness level**: {fairness_metrics[fairness_metrics['Criterion'] == fairness_criterion.split('(')[0].strip()]['Interpretation'].values[0]}
            - **Optimal threshold would be**: Decile {optimal_decile}

            **Trade-off Analysis:**
            - Moving threshold **up** (higher decile): Fewer people jailed, higher precision, lower positive rates
            - Moving threshold **down** (lower decile): More people jailed, lower precision, higher positive rates

            **Practical Implications:**
            - Current threshold jails approximately:
              - **African-American**: {metrics_aa.iloc[idx_aa]['positive_rate']:.1%}
              - **Caucasian**: {metrics_caucasian.iloc[idx_caucasian]['positive_rate']:.1%}
            - Accuracy trade-off: {abs(metrics_aa.iloc[idx_aa]['accuracy'] - metrics_caucasian.iloc[idx_caucasian]['accuracy']):.3f} difference
            """)

        # Button to go back
        if st.button("Return to Main Analysis", type="primary"):
            toggle_fairness_analysis()
            st.rerun()

# MAIN ANALYSIS PAGE (YOUR ORIGINAL CODE - KEEP AS IS)
else:
    # Select race group
    race_options = ['All'] + sorted(data['race'].unique().tolist())
    selected_race = st.segmented_control("Select Race Group:", race_options,
                                         selection_mode="single", default='All')
    # Filter data
    if selected_race != 'All':
        filtered_data = data[data['race'] == selected_race].copy()
    else:
        filtered_data = data.copy()


    # Prepare features
    def prepare_features(df):
        features_to_drop = [
            'target_binary', 'score_text', 'decile_score', 'v_decile_score',
            'v_score_text', 'decile_score.1', 'id', 'violent_recid',
            'r_days_from_arrest', 'days_b_screening_arrest', 'name',
            'first', 'last', 'sex', 'dob', 'age', 'age_cat', 'race'
        ]

        features_to_drop = [col for col in features_to_drop if col in df.columns]
        X = df.drop(features_to_drop, axis=1, errors='ignore')
        y = df['target_binary']

        # Handle missing values
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) > 0:
            imputer_num = SimpleImputer(strategy='median')
            X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])

            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        return X, y


    # Train model and get scores
    @st.cache_resource
    def train_model(X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        # Train model
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            class_weight='balanced',
            min_samples_split=5,
            min_samples_leaf=2
        )

        classifier.fit(X_train, y_train)

        # Get scores
        y_scores = classifier.predict_proba(X_test)[:, 1]

        return X_test, y_test, y_scores, classifier


    # Main visualization
    st.header("Risk Score Distribution")

    # Prepare data for selected group
    X, y = prepare_features(filtered_data)
    if len(X) > 50:  # Minimum samples needed
        X_test, y_test, y_scores, model = train_model(X, y)

        # Create decile scores (1-10) from probabilities
        decile_scores = np.digitize(y_scores, bins=np.linspace(0, 1, 11))
        # Adjust so 0% -> 1 (low risk), 10% -> 1, ..., 90% -> 9, 100% -> 10(high risk)
        decile_scores = np.clip(decile_scores, 1, 10)

        # Create DataFrame for visualization
        viz_df = pd.DataFrame({
            'Decile Score': decile_scores,
            'Probability': y_scores,
            'Actual Outcome': y_test,
            'Predicted Risk': ['High' if score > 7 else 'Low' for score in decile_scores]
        })

        # Add jitter for better visualization
        viz_df['Jittered Score'] = viz_df['Decile Score'] + np.random.uniform(-0.2, 0.2, len(viz_df))

        # Threshold slider
        threshold = st.slider(
            "Select Risk Threshold (Decile Score):",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Scores at or above this threshold will be classified as 'High Risk'"
        )

        # Calculate metrics at current threshold
        predicted_high_risk = decile_scores >= threshold
        predicted_low_risk = decile_scores < threshold

        tp = np.sum((predicted_high_risk == 1) & (y_test == 1))
        fp = np.sum((predicted_high_risk == 1) & (y_test == 0))
        tn = np.sum((predicted_low_risk == 1) & (y_test == 0))
        fn = np.sum((predicted_low_risk == 1) & (y_test == 1))

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("False Positive Rate", f"{fpr:.2%}")
        with col3:
            st.metric("True Positive Rate", f"{tpr:.2%}")
        with col4:
            st.metric("Precision", f"{precision:.2%}")

        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Risk Score Distribution", "Performance Metrics by Threshold"),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )

        # Create colors for actual outcomes
        colors = ['forestgreen', 'red']  # Forestgreen-->not re-arrested, Red-->re-arrested

        # Add scatter plot for decile scores
        for outcome, color in zip([0, 1], colors):
            mask = viz_df['Actual Outcome'] == outcome
            outcome_label = 'Not Re-arrested' if outcome == 0 else 'Re-arrested'

            fig.add_trace(
                go.Scatter(
                    x=viz_df.loc[mask, 'Jittered Score'],
                    y=np.random.uniform(0, 1, mask.sum()),  # Random y for jitter
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=8,
                        opacity=0.6,
                        symbol='circle' if outcome == 0 else 'triangle-up',
                        line=dict(width=1, color='black')
                    ),
                    name=outcome_label,
                    text=[f"Score: {s}<br>Actual: {outcome_label}"
                          for s in viz_df.loc[mask, 'Decile Score']],
                    hoverinfo='text'
                ),
                row=1, col=1
            )

        # Add threshold line
        fig.add_vline(
            x=threshold - 0.5,  # Center between scores
            line_dash="dash",
            line_color="black",
            line_width=2,
            row=1, col=1
        )

        # Add annotations for jailed/released
        fig.add_annotation(
            x=threshold - 1.3,
            y=1.2,
            text="⬅️ Released",
            showarrow=False,
            font=dict(size=15, color="green"),
            row=1, col=1
        )

        fig.add_annotation(
            x=threshold + 0.2,
            y=1.2,
            text="Jailed ➡️",
            showarrow=False,
            font=dict(size=15, color="red"),
            row=1, col=1
        )

        # Update x-axis for first plot
        fig.update_xaxes(
            title_text="Risk Score (1-10)",
            tickmode='array',
            tickvals=list(range(1, 11)),
            ticktext=[f"{i}" for i in range(1, 11)],
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="",
            showticklabels=False,
            range=[-0.3, 1.3],
            row=1, col=1
        )

        # Calculate metrics across all thresholds
        thresholds = list(range(1, 11))
        accuracies = []
        fprs = []
        tprs = []

        for t in thresholds:
            pred_high = decile_scores >= t
            pred_low = decile_scores < t

            tp_t = np.sum((pred_high == 1) & (y_test == 1))
            fp_t = np.sum((pred_high == 1) & (y_test == 0))
            tn_t = np.sum((pred_low == 1) & (y_test == 0))
            fn_t = np.sum((pred_low == 1) & (y_test == 1))

            acc = (tp_t + tn_t) / len(y_test) if len(y_test) > 0 else 0
            fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
            tpr_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0

            accuracies.append(acc)
            fprs.append(fpr_t)
            tprs.append(tpr_t)

        # Add accuracy line
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=accuracies,
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )

        # Add FPR line
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=fprs,
                mode='lines+markers',
                name='False Positive Rate',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )

        # Add TPR line
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=tprs,
                mode='lines+markers',
                name='True Positive Rate',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )

        # Add vertical line at current threshold
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="black",
            line_width=2,
            row=2, col=1
        )

        # Update axes for second plot
        fig.update_xaxes(
            title_text="Threshold (Decile Score)",
            tickmode='array',
            tickvals=list(range(1, 11)),
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Metric Value",
            range=[0, 1],
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            ),
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show confusion matrix
        st.subheader("Confusion Matrix at Current Threshold")

        confusion_data = pd.DataFrame({
            '': ['Actual Re-arrested', 'Actual Not Re-arrested'],
            'Predicted High Risk': [fp, tp],
            'Predicted Low Risk': [tn, fn]
        }).set_index('')

        st.dataframe(confusion_data.style.format("{:,.0f}").background_gradient(cmap='GnBu'))