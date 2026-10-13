# Import the libraries required for the interactive dashboard.
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# Define the project and report directories relative to this script.
BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"

# Define the available CSV reports and their display names.
REPORT_FILES = {
    "معیارهای مدل روی تست": "multilabel_svm_metrics.csv",
    "پیش‌بینی نمونه‌های تست": "multilabel_svm_test_predictions.csv",
    "معیارهای اعتبارسنجی داخلی": "multilabel_svm_cv_metrics.csv",
    "پیش‌بینی‌های اعتبارسنجی داخلی": "multilabel_svm_cv_predictions.csv",
    "خلاصه پایداری مناطق طیفی": "spectral_region_lopo_summary.csv",
    "جزئیات مناطق طیفی": "spectral_region_lopo_details.csv",
    "خلاصه پایداری رنگ‌ها": "svm_cv_stability_summary.csv",
    "تحلیل امتیازهای مدل": "svm_score_analysis.csv",
    "خلاصه آستانه‌ها": "svm_threshold_summary.csv",
    "تحلیل مناطق طیفی": "dye_spectral_region_analysis.csv",
    "خلاصه عددی مناطق کاندید": "candidate_region_numeric_summary.csv",
    "فراوانی برچسب‌ها": "label_counts.csv",
    "فراداده نمونه‌ها": "sample_metadata.csv",
}


# Configure the Streamlit page before rendering dashboard elements.
st.set_page_config(
    page_title="Saffron Adulteration Intelligence",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply Persian right-to-left styling to the dashboard.
st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            direction: rtl;
            text-align: right;
        }
        [data-testid="stAppViewContainer"] {
            direction: rtl;
        }
        [data-testid="stMetricValue"] {
            direction: ltr;
            text-align: right;
        }
        div[data-testid="stDataFrame"] {
            direction: ltr;
        }
        .dashboard-note {
            padding: 12px;
            border: 1px solid #88888855;
            border-radius: 8px;
            margin-bottom: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load a CSV report and return an empty DataFrame when the file is unavailable.
@st.cache_data
def load_report(file_name):
    # Resolve the requested report path inside the reports directory.
    file_path = REPORTS_DIR / file_name

    # Return an empty table when the report does not exist.
    if not file_path.exists():
        return pd.DataFrame()

    # Read the CSV while tolerating UTF-8 BOM and common encoding differences.
    try:
        return pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin-1")


# Parse a semicolon-separated list of dye names into a normalized set.
def parse_dyes(value):
    # Treat missing values and empty strings as having no detected dyes.
    if pd.isna(value) or not str(value).strip():
        return set()

    # Split the value and remove whitespace around each dye name.
    return {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }


# Format numeric values as readable percentages without changing missing values.
def format_percent(value):
    # Return a dash when the supplied value cannot be interpreted as a number.
    if pd.isna(value):
        return "—"

    # Convert a metric between zero and one to a percentage string.
    return f"{value:.1%}"


# Load all reports used by the dashboard.
test_metrics = load_report("multilabel_svm_metrics.csv")
test_predictions = load_report("multilabel_svm_test_predictions.csv")
cv_metrics = load_report("multilabel_svm_cv_metrics.csv")
region_summary = load_report("spectral_region_lopo_summary.csv")
region_details = load_report("spectral_region_lopo_details.csv")
stability_summary = load_report("svm_cv_stability_summary.csv")
threshold_summary = load_report("svm_threshold_summary.csv")
label_counts = load_report("label_counts.csv")
sample_metadata = load_report("sample_metadata.csv")


# Display the dashboard title and its scientific purpose.
st.title("🧪 سامانه هوشمند بررسی تقلب زعفران")
st.caption(
    "داشبورد ارزیابی مدل شناسایی رنگ‌های مصنوعی در نمونه‌های زعفران "
    "| پروژه Saffron Adulteration Intelligence"
)


# Show a useful warning if the main test report is missing.
if test_predictions.empty:
    st.error(
        "گزارش پیش‌بینی تست پیدا نشد. ابتدا اسکریپت آموزش مدل را اجرا کنید "
        "و مطمئن شوید reports/multilabel_svm_test_predictions.csv ساخته شده است."
    )
    st.stop()


# Keep only known-label test samples when the expected metadata columns exist.
known_test = test_predictions.copy()

# Exclude unknown-label samples from the known-label evaluation.
if "Dataset_Role" in known_test.columns:
    known_test = known_test[
        known_test["Dataset_Role"].astype(str).str.casefold() == "test"
    ].copy()

# Exclude samples whose adulterant type is explicitly unknown.
if "Adulterant_Type" in known_test.columns:
    known_test = known_test[
        known_test["Adulterant_Type"].fillna("").astype(str).str.casefold()
        != "unknown"
    ].copy()


# Derive sample-level evaluation results from actual and predicted dye labels.
if {"Actual_Dyes", "Predicted_Dyes"}.issubset(known_test.columns):
    # Convert each sample's actual dye list into a set.
    known_test["Actual_Set"] = known_test["Actual_Dyes"].apply(parse_dyes)

    # Convert each sample's predicted dye list into a set.
    known_test["Predicted_Set"] = known_test["Predicted_Dyes"].apply(parse_dyes)

    # Count the actual dyes correctly identified in each sample.
    known_test["Correct_Dye_Count"] = known_test.apply(
        lambda row: len(row["Actual_Set"] & row["Predicted_Set"]),
        axis=1,
    )

    # Flag whether the model predicted at least one dye for a sample.
    known_test["Any_Dye_Predicted"] = known_test["Predicted_Set"].apply(bool)

    # Flag whether the sample contains any known actual dye label.
    known_test["Has_Actual_Dye"] = known_test["Actual_Set"].apply(bool)

    # Flag whether at least one actual dye was correctly identified.
    known_test["Any_Correct_Dye"] = known_test["Correct_Dye_Count"] > 0

    # Count test samples and sample-level prediction outcomes.
    test_sample_count = len(known_test)
    predicted_sample_count = int(known_test["Any_Dye_Predicted"].sum())
    correctly_identified_samples = int(known_test["Any_Correct_Dye"].sum())
    actual_adulterated_samples = int(known_test["Has_Actual_Dye"].sum())
    missed_adulterated_samples = int(
        (
            known_test["Has_Actual_Dye"]
            & ~known_test["Any_Dye_Predicted"]
        ).sum()
    )
else:
    # Use safe defaults if the expected label columns are absent.
    test_sample_count = len(known_test)
    predicted_sample_count = 0
    correctly_identified_samples = 0
    actual_adulterated_samples = 0
    missed_adulterated_samples = 0


# Display headline metrics for the known-label test samples.
st.subheader("۱. نمای کلی ارزیابی تست")

metric_columns = st.columns(5)

# Show the number of known-label test samples.
metric_columns[0].metric("نمونه‌های تست", test_sample_count)

# Show how many samples received at least one dye prediction.
metric_columns[1].metric("نمونه با پیش‌بینی رنگ", predicted_sample_count)

# Show the number of samples with at least one correctly identified dye.
metric_columns[2].metric(
    "نمونه با تشخیص صحیح رنگ",
    correctly_identified_samples,
)

# Show the number of actually adulterated samples in the known-label test set.
metric_columns[3].metric(
    "نمونه‌های دارای رنگ واقعی",
    actual_adulterated_samples,
)

# Show how many adulterated samples received no dye prediction.
metric_columns[4].metric(
    "نمونه‌های تقلبی بدون تشخیص رنگ",
    missed_adulterated_samples,
)


# Explain the limitations of the current external test results.
st.warning(
    "تفسیر مهم: این شاخص‌ها از برچسب‌های واقعی و پیش‌بینی‌های ثبت‌شده "
    "محاسبه می‌شوند. تشخیص یک رنگ اشتباه، تشخیص صحیح محسوب نمی‌شود. "
    "تعداد کم نمونه‌های تست نیز مانع نتیجه‌گیری قطعی درباره عملکرد عمومی مدل است."
)


# Display detailed actual-versus-predicted results for test samples.
st.subheader("۲. مقایسه نمونه‌های تست")

# Select only columns that are present in the report.
test_display_columns = [
    column
    for column in [
        "Sample_ID",
        "Target_Label",
        "Adulterant_Type",
        "Actual_Dyes",
        "Predicted_Dyes",
        "Correct_Dye_Count",
    ]
    if column in known_test.columns
]

# Present the test predictions in a searchable, scrollable table.
if test_display_columns:
    st.dataframe(
        known_test[test_display_columns],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.dataframe(known_test, use_container_width=True, hide_index=True)


# Provide a per-sample interpretation of whether any dye was correctly identified.
if "Any_Correct_Dye" in known_test.columns:
    # Count samples where at least one actual dye was correctly identified.
    correct_count = int(known_test["Any_Correct_Dye"].sum())

    # Count samples where actual dyes exist but none were correctly identified.
    incorrect_or_missed_count = int(
        (
            known_test["Has_Actual_Dye"]
            & ~known_test["Any_Correct_Dye"]
        ).sum()
    )

    # Count genuine samples incorrectly assigned at least one dye label.
    false_alarm_count = int(
        (
            ~known_test["Has_Actual_Dye"]
            & known_test["Any_Dye_Predicted"]
        ).sum()
    )

    # Build a chart of sample-level test outcomes.
    outcome_data = pd.DataFrame(
        {
            "وضعیت": [
                "حداقل یک رنگ صحیح",
                "رنگ واقعی بدون تشخیص صحیح",
                "نمونه بدون رنگ واقعی با پیش‌بینی رنگ",
            ],
            "تعداد": [
                correct_count,
                incorrect_or_missed_count,
                false_alarm_count,
            ],
        }
    )

    # Draw the test outcome chart.
    outcome_chart = px.bar(
        outcome_data,
        x="وضعیت",
        y="تعداد",
        text="تعداد",
        title="نتیجه پیش‌بینی در سطح نمونه",
    )
    outcome_chart.update_layout(
        xaxis_title="وضعیت نمونه",
        yaxis_title="تعداد نمونه",
    )
    st.plotly_chart(outcome_chart, use_container_width=True)


# Display internal cross-validation metrics.
st.subheader("۳. اعتبارسنجی داخلی مدل (Cross-validation)")

if not cv_metrics.empty and "Dye" in cv_metrics.columns:
    # Identify the metrics that are actually present in the report.
    cv_metric_columns = [
        column
        for column in [
            "Dye",
            "Positive_Samples",
            "Negative_Samples",
            "Precision",
            "Recall",
            "F1",
            "Average_Precision",
            "Predicted_Positive",
        ]
        if column in cv_metrics.columns
    ]

    # Display the original internal-validation metrics.
    st.dataframe(
        cv_metrics[cv_metric_columns],
        use_container_width=True,
        hide_index=True,
    )

    # Select available numeric metrics for the comparison chart.
    chart_metric_columns = [
        column
        for column in ["Precision", "Recall", "F1", "Average_Precision"]
        if column in cv_metrics.columns
    ]

    # Plot metrics side by side for each dye.
    if chart_metric_columns:
        # Convert the metric table into a long format for Plotly.
        cv_chart_data = cv_metrics[["Dye"] + chart_metric_columns].melt(
            id_vars="Dye",
            value_vars=chart_metric_columns,
            var_name="معیار",
            value_name="مقدار",
        )

        # Remove unavailable values from the chart.
        cv_chart_data = cv_chart_data.dropna(subset=["مقدار"])

        # Draw a grouped bar chart comparing the CV metrics.
        cv_chart = px.bar(
            cv_chart_data,
            x="Dye",
            y="مقدار",
            color="معیار",
            barmode="group",
            title="مقایسه معیارهای اعتبارسنجی داخلی به تفکیک رنگ",
            hover_data=["معیار", "مقدار"],
        )
        cv_chart.update_layout(
            xaxis_title="نوع رنگ",
            yaxis_title="مقدار معیار",
            yaxis_range=[0, 1],
            legend_title="معیار ارزیابی",
        )
        st.plotly_chart(cv_chart, use_container_width=True)

    # Show the number of training positives to highlight sparse classes.
    if "Positive_Samples" in cv_metrics.columns:
        # Sort dyes by their available positive training examples.
        positive_counts = cv_metrics[
            ["Dye", "Positive_Samples"]
        ].sort_values("Positive_Samples")

        # Draw the positive-sample count chart.
        positive_chart = px.bar(
            positive_counts,
            x="Dye",
            y="Positive_Samples",
            title="تعداد نمونه‌های مثبت هر رنگ در اعتبارسنجی داخلی",
            text="Positive_Samples",
        )
        positive_chart.update_layout(
            xaxis_title="نوع رنگ",
            yaxis_title="تعداد نمونه مثبت",
        )
        st.plotly_chart(positive_chart, use_container_width=True)

    # Clarify what internal validation metrics mean.
    st.info(
        "Average Precision کیفیت رتبه‌بندی امتیازها را ارزیابی می‌کند؛ "
        "به‌تنهایی به معنی تشخیص صحیح در آستانه فعلی نیست. "
        "اعتبارسنجی داخلی جایگزین تست مستقل نیست."
    )
else:
    st.info("گزارش اعتبارسنجی داخلی در دسترس نیست.")


# Display the final model metrics report when it is available.
st.subheader("۴. معیارهای نهایی مدل روی تست")

if not test_metrics.empty:
    # Display the report without assuming every metric is evaluable.
    st.dataframe(
        test_metrics,
        use_container_width=True,
        hide_index=True,
    )

    # Plot F1 only for dyes with an available F1 value.
    if {"Dye", "F1"}.issubset(test_metrics.columns):
        # Keep rows where F1 was actually computed.
        test_f1_data = test_metrics.dropna(subset=["F1"]).copy()

        # Draw the F1 chart when at least one dye can be evaluated.
        if not test_f1_data.empty:
            test_f1_chart = px.bar(
                test_f1_data,
                x="Dye",
                y="F1",
                title="F1 رنگ‌ها در تست مستقل",
                text="F1",
            )
            test_f1_chart.update_layout(
                xaxis_title="نوع رنگ",
                yaxis_title="F1",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(test_f1_chart, use_container_width=True)

    # Explain why some test metrics may be blank.
    st.caption(
        "اگر تست مستقل برای یک رنگ نمونه مثبت یا منفی کافی نداشته باشد، "
        "معیارهایی مانند Precision، Recall و F1 ممکن است قابل ارزیابی نباشند."
    )


# Display exploratory candidate spectral regions.
st.subheader("۵. مناطق طیفی کاندید")

if not region_summary.empty and "Dye" in region_summary.columns:
    # Select aggregate rows that summarize all leave-one-positive-out runs.
    if "Omitted_Sample_ID" in region_summary.columns:
        # Keep the aggregate summary rows when the report contains them.
        aggregate_regions = region_summary[
            region_summary["Omitted_Sample_ID"]
            .astype(str)
            .str.upper()
            .eq("ALL_HOLDOUTS")
        ].copy()
    else:
        # Fall back to displaying the full report if no aggregate marker exists.
        aggregate_regions = region_summary.copy()

    # Prefer the aggregate summary table when it contains records.
    region_display = (
        aggregate_regions
        if not aggregate_regions.empty
        else region_summary
    )

    # Select available columns for the region stability table.
    region_columns = [
        column
        for column in [
            "Dye",
            "Positive_Samples",
            "Evaluated_Holdouts",
            "Matched_Reference_Regions",
            "Reference_Regions",
            "Mean_Region_Overlap",
            "Median_Region_Overlap",
            "Assessment",
        ]
        if column in region_display.columns
    ]

    # Show the candidate-region stability summary.
    st.dataframe(
        region_display[region_columns],
        use_container_width=True,
        hide_index=True,
    )

    # Plot mean region overlap when available.
    if {"Dye", "Mean_Region_Overlap"}.issubset(region_display.columns):
        # Keep rows with a numeric overlap value.
        overlap_data = region_display.dropna(
            subset=["Mean_Region_Overlap"]
        ).copy()

        # Draw the region overlap chart if aggregate values are available.
        if not overlap_data.empty:
            overlap_chart = px.bar(
                overlap_data,
                x="Dye",
                y="Mean_Region_Overlap",
                title="پایداری میانگین رتبه‌بندی مناطق طیفی",
                text="Mean_Region_Overlap",
            )
            overlap_chart.update_layout(
                xaxis_title="نوع رنگ",
                yaxis_title="میانگین هم‌پوشانی مناطق",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(overlap_chart, use_container_width=True)

    # Explain the scientific limits of candidate spectral regions.
    st.warning(
        "این مناطق کاندید، حاصل تحلیل اکتشافی داده‌ها هستند. "
        "پایداری رتبه‌بندی یا جدایی عددی در همین مجموعه، "
        "اثبات اختصاصی‌بودن شیمیایی منطقه یا شناسایی قطعی رنگ نیست."
    )
else:
    st.info("گزارش مناطق طیفی در دسترس نیست.")


# Display the threshold analysis report if it exists.
st.subheader("۶. تحلیل آستانه تصمیم")

if not threshold_summary.empty:
    # Display threshold summaries without assuming a fixed schema.
    st.dataframe(
        threshold_summary,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("گزارش خلاصه آستانه‌ها در این اجرا موجود نیست.")


# Provide access to the available CSV reports.
st.subheader("۷. مشاهده و دانلود گزارش‌ها")

# Let the user select a report by its human-readable name.
selected_report_name = st.selectbox(
    "گزارش موردنظر را انتخاب کنید",
    options=list(REPORT_FILES.keys()),
)

# Resolve the selected report filename.
selected_file_name = REPORT_FILES[selected_report_name]

# Load the selected report.
selected_report = load_report(selected_file_name)

# Display and provide a download button for the selected report.
if not selected_report.empty:
    # Show the selected CSV as a table.
    st.dataframe(
        selected_report,
        use_container_width=True,
        hide_index=True,
    )

    # Read the original report bytes for an exact CSV download.
    selected_file_path = REPORTS_DIR / selected_file_name

    # Offer the selected report as a downloadable CSV file.
    st.download_button(
        label="دانلود گزارش CSV",
        data=selected_file_path.read_bytes(),
        file_name=selected_file_name,
        mime="text/csv",
    )
else:
    st.info(
        f"فایل {selected_file_name} موجود نیست یا داده‌ای برای نمایش ندارد."
    )


# Finish with a concise scientific interpretation of the current results.
st.divider()
st.subheader("جمع‌بندی وضعیت فعلی")

st.markdown(
    """
    - مدل فعلی برای استفاده عملی و اعلام قطعی نوع رنگ آماده نیست.
    - نتیجه اعتبارسنجی داخلی باید جدا از نتیجه تست مستقل تفسیر شود.
    - تعداد نمونه‌های مثبت برخی رنگ‌ها بسیار کم است.
    - پودر خالص رنگ‌ها نباید بدون بررسی سازگاری، معادل طیف رنگ در مخلوط زعفران فرض شود.
    - برای ارزیابی قابل اتکا، به نمونه‌های مستقل بیشتر و برچسب‌های آزمایشگاهی معتبر نیاز داریم.
    """
)