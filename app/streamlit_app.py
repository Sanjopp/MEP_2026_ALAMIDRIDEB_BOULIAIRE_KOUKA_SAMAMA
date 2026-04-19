import streamlit as st
import mlflow
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="BatchNorm Dashboard", layout="wide")

# -----------------------------
# Style
# -----------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .stApp {
        background-color: #f5f7fb;
        color: #111827;
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #e9eef6;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #0f172a;
        font-family: "Inter", "Segoe UI", sans-serif;
        font-weight: 700;
    }

    p, label, div, span {
        color: #1f2937;
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }

    .info-box {
        background: #ffffff;
        border: 1px solid #dbe3ef;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbe3ef;
        padding: 0.8rem 1rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }

    div[data-baseweb="tab-list"] {
        gap: 1rem;
        border-bottom: 1px solid #dbe3ef;
    }

    button[data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        color: #475569;
        background: transparent;
        border: none;
        padding: 0.6rem 0.2rem;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom: 3px solid #3b82f6 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #dbe3ef;
        border-radius: 12px;
        overflow: hidden;
        background: white;
    }

    /* Tags multiselect à gauche */
    [data-baseweb="tag"] {
        background-color: #60a5fa !important;
        color: white !important;
        border-radius: 8px !important;
    }

    [data-baseweb="tag"] span {
        color: white !important;
    }

    [data-baseweb="tag"] svg {
        fill: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# -----------------------------
# Helpers
# -----------------------------
def to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def to_str(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


@st.cache_data(ttl=10)
def load_runs(experiment_name: str) -> pd.DataFrame:
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return pd.DataFrame()
    return mlflow.search_runs(experiment_ids=[exp.experiment_id])


# -----------------------------
# Header
# -----------------------------
st.title("Batch Normalization Dashboard")
st.markdown(
    '<div class="small-muted">Exploration des runs MLflow et lancement interactif '
    "des entraînements via FastAPI.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
    <b>Projet :</b> mise en production d’un projet de Data Science autour de l’impact du Batch Normalization
    sur les performances d’entraînement, avec suivi des expériences via MLflow et pilotage via FastAPI.
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Experiment Explorer", "Train via API"])

# =========================================================
# TAB 1 - MLFLOW EXPLORER
# =========================================================
with tab1:
    experiment_name = "cifar10_cnn"
    runs = load_runs(experiment_name)

    if runs.empty:
        st.error(f"Aucun run trouvé pour l'expérience '{experiment_name}' ou expérience introuvable.")
        st.stop()

    metric_cols = [
        "metrics.train_acc",
        "metrics.val_acc",
        "metrics.test_acc",
        "metrics.train_loss",
        "metrics.val_loss",
        "metrics.test_loss",
        "metrics.grad_norm",
    ]

    runs = to_float(runs, metric_cols + ["params.learning_rate"])
    runs = to_str(
        runs,
        [
            "params.dataset",
            "params.model_type",
            "params.use_bn",
            "params.optimizer",
            "params.batch_size",
            "params.num_epochs",
            "params.seed",
        ],
    )

    st.sidebar.header("Filtres MLflow")

    filtered_runs = runs.copy()

    if "params.use_bn" in filtered_runs.columns:
        bn_values = sorted(filtered_runs["params.use_bn"].dropna().unique().tolist())
        selected_bn = st.sidebar.multiselect("BatchNorm", bn_values, default=bn_values)
        filtered_runs = filtered_runs[filtered_runs["params.use_bn"].isin(selected_bn)]

    if "params.model_type" in filtered_runs.columns:
        model_values = sorted(filtered_runs["params.model_type"].dropna().unique().tolist())
        selected_models = st.sidebar.multiselect("Modèle", model_values, default=model_values)
        filtered_runs = filtered_runs[filtered_runs["params.model_type"].isin(selected_models)]

    if "params.dataset" in filtered_runs.columns:
        dataset_values = sorted(filtered_runs["params.dataset"].dropna().unique().tolist())
        selected_datasets = st.sidebar.multiselect("Dataset", dataset_values, default=dataset_values)
        filtered_runs = filtered_runs[filtered_runs["params.dataset"].isin(selected_datasets)]

    if "params.optimizer" in filtered_runs.columns:
        opt_values = sorted(filtered_runs["params.optimizer"].dropna().unique().tolist())
        selected_opts = st.sidebar.multiselect("Optimiseur", opt_values, default=opt_values)
        filtered_runs = filtered_runs[filtered_runs["params.optimizer"].isin(selected_opts)]

    if filtered_runs.empty:
        st.warning("Aucun run ne correspond aux filtres sélectionnés.")
        st.stop()

    st.subheader("Vue d’ensemble")

    num_runs = len(filtered_runs)
    best_test_acc = filtered_runs["metrics.test_acc"].max() if "metrics.test_acc" in filtered_runs.columns else None
    best_val_acc = filtered_runs["metrics.val_acc"].max() if "metrics.val_acc" in filtered_runs.columns else None
    mean_train_acc = filtered_runs["metrics.train_acc"].mean() if "metrics.train_acc" in filtered_runs.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nombre de runs", f"{num_runs}")
    c2.metric("Meilleure test accuracy", f"{best_test_acc:.4f}" if pd.notna(best_test_acc) else "N/A")
    c3.metric("Meilleure val accuracy", f"{best_val_acc:.4f}" if pd.notna(best_val_acc) else "N/A")
    c4.metric("Train accuracy moyenne", f"{mean_train_acc:.4f}" if pd.notna(mean_train_acc) else "N/A")

    if "metrics.test_acc" in filtered_runs.columns and filtered_runs["metrics.test_acc"].notna().any():
        best_idx = filtered_runs["metrics.test_acc"].idxmax()
        best_row = filtered_runs.loc[best_idx]
        st.success(
            f"Meilleur run : dataset={best_row.get('params.dataset', 'N/A')} | "
            f"modèle={best_row.get('params.model_type', 'N/A')} | "
            f"BatchNorm={best_row.get('params.use_bn', 'N/A')} | "
            f"optimiseur={best_row.get('params.optimizer', 'N/A')} | "
            f"test_acc={best_row.get('metrics.test_acc', float('nan')):.4f}"
        )

    st.subheader("Analyse visuelle")

    plot_df = filtered_runs.copy()
    if "run_id" in plot_df.columns:
        plot_df["run_short"] = plot_df["run_id"].str[:8]

    g1, g2 = st.columns(2)

    if {"run_short", "metrics.test_acc"}.issubset(plot_df.columns):
        fig_test = px.bar(
            plot_df,
            x="run_short",
            y="metrics.test_acc",
            color="params.use_bn" if "params.use_bn" in plot_df.columns else None,
            barmode="group",
            title="Test accuracy par run",
            labels={
                "run_short": "Run",
                "metrics.test_acc": "Test accuracy",
                "params.use_bn": "BatchNorm",
            },
        )
        fig_test.update_layout(
            height=420,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="#f5f5f5",
        )
        g1.plotly_chart(fig_test, use_container_width=True)

    if {"metrics.train_acc", "metrics.test_acc"}.issubset(plot_df.columns):
        fig_scatter = px.scatter(
            plot_df,
            x="metrics.train_acc",
            y="metrics.test_acc",
            color="params.use_bn" if "params.use_bn" in plot_df.columns else None,
            symbol="params.model_type" if "params.model_type" in plot_df.columns else None,
            hover_name="run_id" if "run_id" in plot_df.columns else None,
            title="Train accuracy vs Test accuracy",
            labels={
                "metrics.train_acc": "Train accuracy",
                "metrics.test_acc": "Test accuracy",
                "params.use_bn": "BatchNorm",
                "params.model_type": "Modèle",
            },
        )
        fig_scatter.update_layout(
            height=420,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="#f5f5f5",
        )
        g2.plotly_chart(fig_scatter, use_container_width=True)

    if {"params.use_bn", "metrics.test_acc"}.issubset(plot_df.columns):
        st.subheader("Comparaison BatchNorm")
        bn_summary = (
            plot_df.groupby("params.use_bn", dropna=False)["metrics.test_acc"]
            .agg(["mean", "max", "count"])
            .reset_index()
        )
        bn_summary = bn_summary.rename(
            columns={
                "params.use_bn": "use_bn",
                "mean": "mean_test_acc",
                "max": "best_test_acc",
                "count": "nb_runs",
            }
        )

        fig_bn = px.bar(
            bn_summary,
            x="use_bn",
            y="mean_test_acc",
            hover_data=["best_test_acc", "nb_runs"],
            title="Test accuracy moyenne selon BatchNorm",
            labels={
                "use_bn": "BatchNorm",
                "mean_test_acc": "Test accuracy moyenne",
            },
        )
        fig_bn.update_layout(
            height=380,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="#f5f5f5",
        )
        st.plotly_chart(fig_bn, use_container_width=True)

    st.subheader("Tableau des runs")

    display_cols = [
        "run_id",
        "params.dataset",
        "params.model_type",
        "params.use_bn",
        "params.optimizer",
        "params.learning_rate",
        "params.batch_size",
        "params.num_epochs",
        "metrics.train_acc",
        "metrics.val_acc",
        "metrics.test_acc",
        "metrics.train_loss",
        "metrics.val_loss",
        "metrics.test_loss",
        "metrics.grad_norm",
    ]

    available_cols = [c for c in display_cols if c in filtered_runs.columns]
    runs_display = filtered_runs[available_cols].copy()

    rename_map = {
        "run_id": "run_id",
        "params.dataset": "dataset",
        "params.model_type": "model_type",
        "params.use_bn": "use_bn",
        "params.optimizer": "optimizer",
        "params.learning_rate": "learning_rate",
        "params.batch_size": "batch_size",
        "params.num_epochs": "num_epochs",
        "metrics.train_acc": "train_acc",
        "metrics.val_acc": "val_acc",
        "metrics.test_acc": "test_acc",
        "metrics.train_loss": "train_loss",
        "metrics.val_loss": "val_loss",
        "metrics.test_loss": "test_loss",
        "metrics.grad_norm": "grad_norm",
    }

    runs_display = runs_display.rename(columns=rename_map)

    for col in ["train_acc", "val_acc", "test_acc", "train_loss", "val_loss", "test_loss", "grad_norm", "learning_rate"]:
        if col in runs_display.columns:
            runs_display[col] = pd.to_numeric(runs_display[col], errors="coerce").round(4)

    if "test_acc" in runs_display.columns:
        runs_display = runs_display.sort_values(by="test_acc", ascending=False)

    st.dataframe(runs_display, use_container_width=True, hide_index=True)

# =========================================================
# TAB 2 - FASTAPI TRAINING
# =========================================================
with tab2:
    st.subheader("Pilotage via FastAPI")
    st.markdown(
        '<div class="small-muted">Tester l’API et lancer une nouvelle expérience depuis l’interface.</div>',
        unsafe_allow_html=True,
    )

    health_col, refresh_col = st.columns([1, 1])

    with health_col:
        if st.button("Tester /health"):
            try:
                response = requests.get("http://127.0.0.1:8000/health", timeout=5)
                if response.ok:
                    st.success(response.json())
                else:
                    st.warning(f"Réponse API : {response.status_code}")
            except Exception as exc:
                st.error(f"API inaccessible : {exc}")

    with refresh_col:
        if st.button("Rappel : rafraîchir les runs MLflow"):
            st.info("Après un entraînement réussi, retourne sur l’onglet Experiment Explorer et rafraîchis la page.")

    st.markdown("---")
    st.subheader("Lancer une nouvelle expérience")

    form_col1, form_col2 = st.columns(2)

    with form_col1:
        dataset = st.selectbox("Dataset", ["cifar10", "fashionmnist"])
        model_type = st.selectbox("Model type", ["cnn", "mlp"])
        use_bn = st.selectbox("Use BatchNorm", [True, False])
        optimizer = st.selectbox("Optimizer", ["adam", "sgd"])

    with form_col2:
        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.0001,
            max_value=1.0,
            value=0.001,
            step=0.0001,
            format="%.4f",
        )
        num_epochs = st.slider("Epochs", min_value=1, max_value=20, value=1)
        batch_size = st.selectbox("Batch size", [32, 64, 128, 256], index=1)
        seed = st.number_input("Seed", min_value=0, max_value=10000, value=42, step=1)

    if st.button("Lancer l'entraînement via API", type="primary"):
        payload = {
            "dataset": dataset,
            "model_type": model_type,
            "use_bn": use_bn,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "optimizer": optimizer,
            "batch_size": batch_size,
            "seed": int(seed),
            "verbose": True,
        }

        with st.spinner("Entraînement en cours..."):
            try:
                response = requests.post("http://127.0.0.1:8000/train", json=payload, timeout=600)
                data = response.json()

                if response.status_code == 200 and data.get("status") == "success":
                    st.success("Entraînement terminé avec succès.")
                else:
                    st.warning("Le training s'est terminé avec un problème.")

                st.code(data.get("command", ""), language="bash")

                with st.expander("stdout"):
                    st.text(data.get("stdout", ""))

                with st.expander("stderr"):
                    st.text(data.get("stderr", ""))

                st.info("Le nouveau run devrait maintenant apparaître dans MLflow.")
            except Exception as exc:
                st.error(f"Erreur lors de l'appel API /train : {exc}")