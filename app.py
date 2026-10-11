
from pathlib import Path
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "expenses.csv"
BUDGET_FILE = APP_DIR / "budget.csv"

CATEGORIES = [
    "Food & Dining", "Transport", "Shopping",
    "Bills & Utilities", "Health", "Education",
    "Entertainment", "Travel", "Home", "Other"
]

PAYMENT_METHODS = [
    "UPI", "Cash", "Debit Card",
    "Credit Card", "Bank Transfer", "Other"
]

COLUMNS = [
    "id", "date", "amount",
    "category", "description", "payment_method"
]

st.set_page_config(
    page_title="SmartSpend AI",
    page_icon="💸",
    layout="wide"
)

# --------------------------------------------------
# 2. LIGHT THEME
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: #f6f8fc;
}
.block-container {
    padding-top: 1.6rem;
    max-width: 1450px;
}
.hero {
    background: linear-gradient(120deg, #ffffff, #eaf7f5);
    border: 1px solid #dcebe8;
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 18px;
}
.hero h1 {
    margin: 0;
    color: #163b42;
    font-size: 2rem;
}
.hero p {
    color: #587078;
    margin-top: 7px;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5eaf1;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 2px 8px rgba(20,40,70,.035);
}
section[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #e7edf3;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. CSV STORAGE
# --------------------------------------------------

def ensure_files():
    if not DATA_FILE.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(
            DATA_FILE, index=False
        )

    if not BUDGET_FILE.exists():
        pd.DataFrame(
            columns=["month", "budget"]
        ).to_csv(BUDGET_FILE, index=False)


def load_expenses():
    ensure_files()

    try:
        df = pd.read_csv(DATA_FILE)

        for col in COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        df = df[COLUMNS]

        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )
        df["amount"] = pd.to_numeric(
            df["amount"], errors="coerce"
        )

        df = df.dropna(subset=["date", "amount"])

        df["category"] = (
            df["category"].fillna("Other").astype(str)
        )
        df["description"] = (
            df["description"].fillna("").astype(str)
        )
        df["payment_method"] = (
            df["payment_method"].fillna("Other").astype(str)
        )
        df["id"] = df["id"].fillna("").astype(str)

        return df

    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame(columns=COLUMNS)


def save_expenses(df):
    output = df.copy()

    if not output.empty:
        output["date"] = (
            pd.to_datetime(output["date"])
            .dt.strftime("%Y-%m-%d")
        )

    output.to_csv(DATA_FILE, index=False)


def load_budgets():
    ensure_files()

    try:
        budgets = pd.read_csv(BUDGET_FILE)

        if budgets.empty:
            return {}

        budgets["month"] = budgets["month"].astype(str)
        budgets["budget"] = pd.to_numeric(
            budgets["budget"], errors="coerce"
        )

        return dict(
            zip(budgets["month"], budgets["budget"].fillna(0))
        )

    except (pd.errors.EmptyDataError, OSError, ValueError):
        return {}


def save_budgets(budgets):
    rows = [
        {"month": month, "budget": amount}
        for month, amount in sorted(budgets.items())
    ]

    pd.DataFrame(
        rows, columns=["month", "budget"]
    ).to_csv(BUDGET_FILE, index=False)


def money(value):
    return f"₹{value:,.2f}"


# --------------------------------------------------
# 4. DATA ANALYSIS
# --------------------------------------------------

def monthly_totals(df):
    if df.empty:
        return pd.DataFrame(columns=["month", "amount"])

    result = df.assign(
        month=df["date"].dt.to_period("M").dt.to_timestamp()
    )

    return (
        result.groupby("month", as_index=False)["amount"]
        .sum()
        .sort_values("month")
    )


def add_expense(
    df, amount, expense_date,
    category, description, payment_method
):
    new_row = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "date": pd.Timestamp(expense_date),
        "amount": float(amount),
        "category": category,
        "description": description.strip(),
        "payment_method": payment_method
    }

    return pd.concat(
        [df, pd.DataFrame([new_row])],
        ignore_index=True
    )


# --------------------------------------------------
# 5. SPENDING PREDICTION
# --------------------------------------------------

def forecast_next_month(df):
    monthly = monthly_totals(df)

    if len(monthly) == 0:
        return None, "Add expenses to generate a forecast."

    if len(monthly) < 3:
        average = float(monthly["amount"].mean())

        return average, (
            "Based on the average of available months. "
            "Add at least 3 months for a trend model."
        )

    # Convert months into sequential numeric values.
    x = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly["amount"].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(x, y)

    prediction = float(
        model.predict(
            np.array([[len(monthly)]])
        )[0]
    )

    return max(0.0, prediction), (
        "Linear regression estimates next month's spending. "
        "This is a forecast, not a guarantee."
    )


# --------------------------------------------------
# 6. UNUSUAL-EXPENSE DETECTION
# --------------------------------------------------

def find_unusual_expenses(df):
    result = df.copy()
    result["unusual"] = False

    if len(result) < 12 or result["amount"].nunique() < 4:
        return result, (
            "Add at least 12 transactions with varied amounts "
            "to enable anomaly detection."
        )

    # Learn which transaction amounts are unusual.
    features = result[["amount"]].astype(float).to_numpy()

    model = IsolationForest(
        n_estimators=150,
        contamination=0.08,
        random_state=42
    )

    result["unusual"] = model.fit_predict(features) == -1

    return result, (
        "Alerts are statistical outliers based on amount only. "
        "Review each transaction in context."
    )


# --------------------------------------------------
# 7. INITIALIZE APPLICATION
# --------------------------------------------------

ensure_files()

df = load_expenses()
budgets = load_budgets()

today = date.today()
this_month = today.strftime("%Y-%m")


# --------------------------------------------------
# 8. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 💸 SmartSpend AI")
    st.caption("Personal finance insights")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Expenses",
            "Budgets",
            "Predictions & Alerts"
        ]
    )

    st.divider()
    st.markdown("**Your data**")
    st.caption("Transactions are stored locally in CSV files.")

    st.download_button(
        "Download expense CSV",
        data=DATA_FILE.read_bytes(),
        file_name="smartspend_expenses.csv",
        mime="text/csv",
        use_container_width=True
    )


# --------------------------------------------------
# 9. APPLICATION HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>SmartSpend AI</h1>
    <p>
        Understand your spending, stay on budget,
        and make more informed money decisions.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 10. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    current = (
        df[df["date"].dt.strftime("%Y-%m") == this_month]
        if not df.empty else df
    )

    total_month = (
        float(current["amount"].sum())
        if not current.empty else 0.0
    )

    budget = float(budgets.get(this_month, 0.0))
    remaining = budget - total_month if budget > 0 else 0.0

    avg_transaction = (
        float(current["amount"].mean())
        if not current.empty else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Spent This Month", money(total_month))
    c2.metric(
        "Monthly Budget",
        money(budget) if budget > 0 else "Not set"
    )
    c3.metric(
        "Budget Remaining",
        money(remaining) if budget > 0 else "—",
        delta=(
            "Over budget" if remaining < 0
            else "Within budget"
        ) if budget > 0 else None,
        delta_color="inverse"
    )
    c4.metric(
        "Average Transaction",
        money(avg_transaction)
    )

    if budget > 0:
        progress = min(max(total_month / budget, 0), 1)

        st.progress(
            progress,
            text=f"{total_month / budget * 100:.1f}% of monthly budget used"
        )

        if total_month > budget:
            st.error(
                f"You exceeded your budget by "
                f"{money(total_month - budget)}."
            )
        elif total_month >= budget * 0.8:
            st.warning(
                "You have used at least 80% of your monthly budget."
            )

    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Spending Over Time")

        monthly = monthly_totals(df)

        if monthly.empty:
            st.info(
                "No expenses yet. Add your first transaction "
                "from the Expenses page."
            )
        else:
            fig = px.line(
                monthly,
                x="month",
                y="amount",
                markers=True,
                labels={
                    "month": "Month",
                    "amount": "Spending (₹)"
                },
                template="plotly_white"
            )

            fig.update_layout(height=330)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Spending by Category")

        if current.empty:
            st.info(
                "Your category chart will appear after you "
                "add expenses this month."
            )
        else:
            by_category = (
                current.groupby("category", as_index=False)["amount"]
                .sum()
                .sort_values("amount", ascending=False)
            )

            fig = px.pie(
                by_category,
                names="category",
                values="amount",
                hole=0.58,
                template="plotly_white"
            )

            fig.update_layout(height=330)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Transactions")

    if df.empty:
        st.caption("Your recent transactions will appear here.")
    else:
        recent = df.sort_values(
            "date", ascending=False
        ).head(7).copy()

        recent["date"] = recent["date"].dt.strftime("%d %b %Y")
        recent["amount"] = recent["amount"].map(money)

        st.dataframe(
            recent[
                [
                    "date", "description", "category",
                    "payment_method", "amount"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# --------------------------------------------------
# 11. EXPENSE MANAGER
# --------------------------------------------------

elif page == "Expenses":

    st.subheader("Expense Manager")
    st.caption("Record and manage your day-to-day spending.")

    with st.expander(
        "＋ Add a New Expense",
        expanded=df.empty
    ):
        with st.form("add_expense_form", clear_on_submit=True):

            col1, col2 = st.columns(2)

            amount = col1.number_input(
                "Amount (₹)",
                min_value=0.01,
                value=100.0,
                step=50.0
            )

            expense_date = col2.date_input(
                "Date",
                value=today,
                max_value=today
            )

            category = col1.selectbox(
                "Category", CATEGORIES
            )

            payment_method = col2.selectbox(
                "Payment Method", PAYMENT_METHODS
            )

            description = st.text_input(
                "Description",
                placeholder="e.g. Lunch, bus ticket, groceries"
            )

            submitted = st.form_submit_button(
                "Add Expense",
                type="primary",
                use_container_width=True
            )

            if submitted:
                df = add_expense(
                    df, amount, expense_date,
                    category, description, payment_method
                )

                save_expenses(df)

                st.success("Expense saved successfully.")
                st.rerun()

    if df.empty:
        st.info("No expenses recorded yet.")

    else:
        f1, f2, f3 = st.columns(3)

        category_filter = f1.multiselect(
            "Filter Categories",
            sorted(df["category"].unique())
        )

        start_date = f2.date_input(
            "From Date",
            value=df["date"].min().date(),
            key="from_date"
        )

        end_date = f3.date_input(
            "To Date",
            value=max(today, df["date"].max().date()),
            key="to_date"
        )

        filtered = df.copy()

        if category_filter:
            filtered = filtered[
                filtered["category"].isin(category_filter)
            ]

        if start_date <= end_date:
            filtered = filtered[
                (filtered["date"].dt.date >= start_date)
                & (filtered["date"].dt.date <= end_date)
            ]
        else:
            st.warning("The start date must be before the end date.")
            filtered = filtered.iloc[0:0]

        display_df = filtered.sort_values(
            "date", ascending=False
        ).copy()

        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        display_df["amount"] = display_df["amount"].map(money)

        st.dataframe(
            display_df[
                [
                    "id", "date", "amount", "category",
                    "description", "payment_method"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Export Filtered Expenses",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="smartspend_filtered_expenses.csv",
            mime="text/csv"
        )

        with st.expander("Delete an Expense"):
            ids = filtered["id"].astype(str).tolist()

            if ids:
                selected_id = st.selectbox(
                    "Choose Transaction ID", ids
                )

                if st.button("Delete Selected Expense"):
                    df = df[
                        df["id"].astype(str) != selected_id
                    ]

                    save_expenses(df)
                    st.success("Expense deleted.")
                    st.rerun()


# --------------------------------------------------
# 12. BUDGET MANAGEMENT
# --------------------------------------------------

elif page == "Budgets":

    st.subheader("Monthly Budget")
    st.write(
        "Set a spending limit for a month and track your progress."
    )

    selected_month = st.date_input(
        "Choose a date in the month",
        value=today,
        key="budget_month"
    ).strftime("%Y-%m")

    current_budget = float(
        budgets.get(selected_month, 0.0)
    )

    new_budget = st.number_input(
        "Budget Amount (₹)",
        min_value=0.0,
        value=current_budget,
        step=500.0
    )

    if st.button("Save Budget", type="primary"):
        budgets[selected_month] = float(new_budget)
        save_budgets(budgets)
        st.success("Budget saved successfully.")
        st.rerun()

    month_expenses = (
        df[df["date"].dt.strftime("%Y-%m") == selected_month]
        if not df.empty else df
    )

    spent = (
        float(month_expenses["amount"].sum())
        if not month_expenses.empty else 0.0
    )

    active_budget = float(budgets.get(selected_month, 0.0))

    a, b, c = st.columns(3)

    a.metric("Budget", money(active_budget))
    b.metric("Spent", money(spent))
    c.metric(
        "Remaining",
        money(active_budget - spent)
        if active_budget > 0 else "Set a budget"
    )

    if active_budget > 0:
        st.progress(
            min(spent / active_budget, 1.0),
            text=f"{spent / active_budget * 100:.1f}% used"
        )

        if spent > active_budget:
            st.error(
                f"Over budget by {money(spent - active_budget)}."
            )

    if budgets:
        st.subheader("Saved Budgets")

        budget_table = pd.DataFrame([
            {"Month": month, "Budget": money(value)}
            for month, value in sorted(
                budgets.items(), reverse=True
            )
        ])

        st.dataframe(
            budget_table,
            use_container_width=True,
            hide_index=True
        )


# --------------------------------------------------
# 13. PREDICTIONS AND ALERTS
# --------------------------------------------------

elif page == "Predictions & Alerts":

    st.subheader("Spending Prediction")

    forecast, explanation = forecast_next_month(df)

    if forecast is None:
        st.info(explanation)

    else:
        next_month = (
            pd.Timestamp(today).to_period("M") + 1
        ).strftime("%B %Y")

        st.metric(
            f"Estimated Spending for {next_month}",
            money(forecast)
        )

        st.caption(explanation)
        st.caption(
            "Forecast quality depends on the amount and "
            "consistency of your historical data."
        )

        monthly = monthly_totals(df)

        if not monthly.empty:
            chart_data = monthly.copy()
            chart_data["Type"] = "Historical"

            next_start = (
                monthly["month"].max().to_period("M") + 1
            ).to_timestamp()

            forecast_row = pd.DataFrame([{
                "month": next_start,
                "amount": forecast,
                "Type": "Forecast"
            }])

            chart_data = pd.concat(
                [chart_data, forecast_row],
                ignore_index=True
            )

            fig = px.line(
                chart_data,
                x="month",
                y="amount",
                color="Type",
                markers=True,
                labels={
                    "month": "Month",
                    "amount": "Spending (₹)"
                },
                template="plotly_white"
            )

            fig.update_layout(height=330)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Unusual-Expense Alerts")

    flagged, alert_note = find_unusual_expenses(df)

    st.caption(alert_note)

    if flagged["unusual"].any():

        unusual = flagged[
            flagged["unusual"]
        ].sort_values("date", ascending=False).copy()

        st.warning(
            f"{len(unusual)} transaction(s) were flagged for review."
        )

        unusual["date"] = unusual["date"].dt.strftime("%Y-%m-%d")
        unusual["amount"] = unusual["amount"].map(money)

        st.dataframe(
            unusual[
                [
                    "date", "amount", "category",
                    "description", "payment_method"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Alerts use transaction amount only. "
            "Check the context before drawing conclusions."
        )

    else:
        st.success(
            "No unusual transactions were flagged, or there "
            "is not enough data to run the detector."
        )


# --------------------------------------------------
# 14. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SmartSpend AI • Local prototype • "
    "Forecasts are estimates, not financial advice."
)
