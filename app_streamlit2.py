import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Fairness in Risk Assessment", layout="wide")

# Initialize session state for page navigation
if 'show_fairness_analysis' not in st.session_state:
    st.session_state.show_fairness_analysis = False


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

    def create_binary_target(score_text):
        if pd.isna(score_text):
            return np.nan
        score_text = str(score_text).lower().strip()
        if 'low' in score_text:
            return 0
        else:
            return 1

    data['target_binary'] = data['score_text'].apply(create_binary_target)
    data = data.dropna(subset=['target_binary'])
    return data


data = load_data()

# FAIRNESS ANALYSIS PAGE
if st.session_state.show_fairness_analysis:
    st.header("Fairness Criteria Visualization on ROC Curves")

    data_african_american = data[data['race'] == 'African-American'].copy()
    data_caucasian = data[data['race'] == 'Caucasian'].copy()

    st.sidebar.header("Comparison Groups")
    st.sidebar.info(f"""
    **African-American:**
    - Samples: {len(data_african_american):,}
    - Positive cases: {data_african_american['target_binary'].sum():,} ({data_african_american['target_binary'].mean():.1%})

    **Caucasian:**
    - Samples: {len(data_caucasian):,}
    - Positive cases: {data_caucasian['target_binary'].sum():,} ({data_caucasian['target_binary'].mean():.1%})
    """)


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


    def calculate_roc_curve(y_test, y_scores, n_points=100):
        thresholds = np.linspace(0, 1, n_points)
        tpr_values = []
        fpr_values = []
        precision_values = []
        positive_rate_values = []

        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_test == 1))
            fp = np.sum((y_pred == 1) & (y_test == 0))
            tn = np.sum((y_pred == 0) & (y_test == 0))
            fn = np.sum((y_pred == 0) & (y_test == 1))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            positive_rate = np.mean(y_pred)

            tpr_values.append(tpr)
            fpr_values.append(fpr)
            precision_values.append(precision)
            positive_rate_values.append(positive_rate)

        return (np.array(fpr_values), np.array(tpr_values), np.array(thresholds),
                np.array(precision_values), np.array(positive_rate_values))


    def find_fairness_points(fpr_curve, tpr_curve, precision_curve, positive_rate_curve,
                             target_fpr=None, target_tpr=None,
                             target_precision=None, target_positive_rate=None):
        points = {}

        if target_fpr is not None:
            idx = np.argmin(np.abs(fpr_curve - target_fpr))
            points['fpr_match'] = (fpr_curve[idx], tpr_curve[idx])

        if target_tpr is not None:
            idx = np.argmin(np.abs(tpr_curve - target_tpr))
            points['tpr_match'] = (fpr_curve[idx], tpr_curve[idx])

        if target_precision is not None:
            idx = np.argmin(np.abs(precision_curve - target_precision))
            points['precision_match'] = (fpr_curve[idx], tpr_curve[idx])

        if target_positive_rate is not None:
            idx = np.argmin(np.abs(positive_rate_curve - target_positive_rate))
            points['positive_rate_match'] = (fpr_curve[idx], tpr_curve[idx])

        return points


    with st.spinner("Training models..."):
        model_aa = train_model_for_group(data_african_american, "African-American")
        model_caucasian = train_model_for_group(data_caucasian, "Caucasian")

    if model_aa is None or model_caucasian is None:
        st.error("Insufficient data for one or both groups.")
        if st.button("Return to Main Analysis"):
            toggle_fairness_analysis()
            st.rerun()
    else:
        # Calculate ROC curves with additional metrics
        (fpr_aa, tpr_aa, thresholds_aa,
         precision_aa, positive_rate_aa) = calculate_roc_curve(model_aa['y_test'], model_aa['y_scores'])
        (fpr_caucasian, tpr_caucasian, thresholds_caucasian,
         precision_caucasian, positive_rate_caucasian) = calculate_roc_curve(model_caucasian['y_test'],
                                                                             model_caucasian['y_scores'])

        # Threshold selection
        st.subheader("Threshold Selection")
        threshold_decile = st.slider(
            "Select Risk Threshold (Decile Score):",
            min_value=1, max_value=10, value=5, step=1
        )
        threshold_prob = threshold_decile / 10.0

        # Find current threshold points
        idx_aa = np.argmin(np.abs(thresholds_aa - threshold_prob))
        idx_caucasian = np.argmin(np.abs(thresholds_caucasian - threshold_prob))

        current_point_aa = (fpr_aa[idx_aa], tpr_aa[idx_aa])
        current_point_caucasian = (fpr_caucasian[idx_caucasian], tpr_caucasian[idx_caucasian])

        # Calculate current metrics
        current_precision_aa = precision_aa[idx_aa]
        current_precision_caucasian = precision_caucasian[idx_caucasian]
        current_positive_rate_aa = positive_rate_aa[idx_aa]
        current_positive_rate_caucasian = positive_rate_caucasian[idx_caucasian]

        # Find fairness constraint points for each group
        # For Independence: Equal positive rate
        target_positive_rate = (current_positive_rate_aa + current_positive_rate_caucasian) / 2

        # For Separation: Equal TPR and FPR
        target_tpr = (tpr_aa[idx_aa] + tpr_caucasian[idx_caucasian]) / 2
        target_fpr = (fpr_aa[idx_aa] + fpr_caucasian[idx_caucasian]) / 2

        # For Sufficiency: Equal precision
        target_precision = (current_precision_aa + current_precision_caucasian) / 2

        # Find points on each curve that satisfy fairness constraints
        points_aa = find_fairness_points(
            fpr_aa, tpr_aa, precision_aa, positive_rate_aa,
            target_fpr=target_fpr, target_tpr=target_tpr,
            target_precision=target_precision, target_positive_rate=target_positive_rate
        )

        points_caucasian = find_fairness_points(
            fpr_caucasian, tpr_caucasian, precision_caucasian, positive_rate_caucasian,
            target_fpr=target_fpr, target_tpr=target_tpr,
            target_precision=target_precision, target_positive_rate=target_positive_rate
        )

        # DISPLAY SINGLE ROC PLOT WITH ALL FAIRNESS CRITERIA
        st.subheader("ROC with All Fairness Criteria")

        fig = go.Figure()

        # ROC curves
        fig.add_trace(go.Scatter(
            x=fpr_aa, y=tpr_aa, mode='lines',
            name='African-American', line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=fpr_caucasian, y=tpr_caucasian, mode='lines',
            name='Caucasian', line=dict(color='orange', width=2)
        ))

        # Current threshold points
        fig.add_trace(go.Scatter(
            x=[current_point_aa[0]], y=[current_point_aa[1]],
            mode='markers', name='Current Threshold (AA)',
            marker=dict(size=15, color='blue', symbol='circle', line=dict(width=2, color='black'))
        ))
        fig.add_trace(go.Scatter(
            x=[current_point_caucasian[0]], y=[current_point_caucasian[1]],
            mode='markers', name='Current Threshold (Cauc)',
            marker=dict(size=15, color='orange', symbol='circle', line=dict(width=2, color='black'))
        ))

        # Independence constraint points (equal positive rate)
        if 'positive_rate_match' in points_aa and 'positive_rate_match' in points_caucasian:
            ind_point_aa = points_aa['positive_rate_match']
            ind_point_caucasian = points_caucasian['positive_rate_match']

            # Mark independence points with diamond symbol
            fig.add_trace(go.Scatter(
                x=[ind_point_aa[0]], y=[ind_point_aa[1]],
                mode='markers', name='Independence (AA)',
                marker=dict(size=12, color='blue', symbol='diamond', line=dict(width=2, color='black'))
            ))
            fig.add_trace(go.Scatter(
                x=[ind_point_caucasian[0]], y=[ind_point_caucasian[1]],
                mode='markers', name='Independence (Cauc)',
                marker=dict(size=12, color='orange', symbol='diamond', line=dict(width=2, color='black'))
            ))

        # Separation constraint points (equal TPR/FPR)
        if 'tpr_match' in points_aa and 'tpr_match' in points_caucasian:
            sep_point_aa = points_aa['tpr_match']
            sep_point_caucasian = points_caucasian['tpr_match']

            # Mark separation points with cross symbol
            fig.add_trace(go.Scatter(
                x=[sep_point_aa[0]], y=[sep_point_aa[1]],
                mode='markers', name='Separation (AA)',
                marker=dict(size=12, color='blue', symbol='cross', line=dict(width=2, color='black'))
            ))
            fig.add_trace(go.Scatter(
                x=[sep_point_caucasian[0]], y=[sep_point_caucasian[1]],
                mode='markers', name='Separation (Cauc)',
                marker=dict(size=12, color='orange', symbol='cross', line=dict(width=2, color='black'))
            ))

        # Sufficiency constraint points (equal precision) - Added to the same figure
        if 'precision_match' in points_aa and 'precision_match' in points_caucasian:
            suff_point_aa = points_aa['precision_match']
            suff_point_caucasian = points_caucasian['precision_match']

            # Mark sufficiency points with star symbol
            fig.add_trace(go.Scatter(
                x=[suff_point_aa[0]], y=[suff_point_aa[1]],
                mode='markers', name='Sufficiency (AA)',
                marker=dict(size=12, color='blue', symbol='star', line=dict(width=2, color='black'))
            ))
            fig.add_trace(go.Scatter(
                x=[suff_point_caucasian[0]], y=[suff_point_caucasian[1]],
                mode='markers', name='Sufficiency (Cauc)',
                marker=dict(size=12, color='orange', symbol='star', line=dict(width=2, color='black'))
            ))

        # Random classifier
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Random', line=dict(color='gray', dash='dash', width=1)
        ))

        fig.update_layout(
            title="ROC with All Fairness Criteria (Independence, Separation, Sufficiency)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="Black",
                borderwidth=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Explanation
        st.info("""
        **Fairness Criteria:**
        - **Current (Circles):** Current threshold points for each group
        - **Independence (Diamonds):** Points where both groups have equal positive rates
        - **Separation (Crosses):** Points where both groups have equal TPR/FPR  
        - **Sufficiency (Stars):** Points where both groups have equal precision

        **Note:** All three fairness criteria cannot be satisfied simultaneously at the same threshold.
        """)
        # Display current metrics
        st.subheader("Current Metrics at Selected Threshold")

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

        with metrics_col1:
            st.metric("Current Threshold", f"Decile {threshold_decile}")
            st.metric("AA - FPR", f"{current_point_aa[0]:.3f}")
            st.metric("Caucasian - FPR", f"{current_point_caucasian[0]:.3f}")

        with metrics_col2:
            st.metric("Probability Threshold", f"{threshold_prob:.3f}")
            st.metric("AA - TPR", f"{current_point_aa[1]:.3f}")
            st.metric("Caucasian - TPR", f"{current_point_caucasian[1]:.3f}")

        with metrics_col3:
            st.metric("AA - Precision", f"{current_precision_aa:.3f}")
            st.metric("Caucasian - Precision", f"{current_precision_caucasian:.3f}")
            st.metric("AA - Positive Rate", f"{current_positive_rate_aa:.3f}")
            st.metric("Caucasian - Positive Rate", f"{current_positive_rate_caucasian:.3f}")

        if st.button("Return to Main Analysis", type="primary"):
            toggle_fairness_analysis()
            st.rerun()

# MAIN ANALYSIS PAGE
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

    # Display sample info
    st.sidebar.header("Dataset Information")
    st.sidebar.info(f"""
    **Selected Group:** {selected_race}
    **Total Samples:** {len(filtered_data):,}
    **Positive Cases:** {filtered_data['target_binary'].sum():,} ({filtered_data['target_binary'].mean():.1%})
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

    # Train model and get scores
    @st.cache_resource
    def train_model(X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
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
    st.header("Risk Score Distribution Analysis")

    # Prepare data for selected group
    X, y = prepare_features(filtered_data)

    if len(X) > 50:
        X_test, y_test, y_scores, model = train_model(X, y)

        # Create decile scores (1-10) from probabilities
        decile_scores = np.digitize(y_scores, bins=np.linspace(0, 1, 11))
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
        colors = ['forestgreen', 'red']

        # Add scatter plot for decile scores
        for outcome, color in zip([0, 1], colors):
            mask = viz_df['Actual Outcome'] == outcome
            outcome_label = 'Not Re-arrested' if outcome == 0 else 'Re-arrested'

            fig.add_trace(
                go.Scatter(
                    x=viz_df.loc[mask, 'Jittered Score'],
                    y=np.random.uniform(0, 1, mask.sum()),
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
            x=threshold - 0.5,
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

        # Button for fairness analysis
        if st.button("🔍 Show Fairness Analysis", type="primary", use_container_width=True):
            toggle_fairness_analysis()
            st.rerun()

    else:
        st.warning(f"Insufficient data for analysis. Need at least 50 samples, but only have {len(X)}.")
        st.info("Please select a different race group or use 'All' for more data.")