#!/usr/bin/env python3
"""Streamlit visualization tool for EVA-A Task Completion analysis.

Deep-dive into task completion results: scenario-level pass/fail heatmaps,
failure category breakdowns, DB diff inspection, EVA-A component correlation,
and cross-system comparison of accuracy metrics.

Usage:
    streamlit run apps/task_completion_analysis.py

    # Or point at a custom output directory:
    EVA_OUTPUT_DIR=path/to/results streamlit run apps/task_completion_analysis.py
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = os.environ.get("EVA_OUTPUT_DIR", "output")

# Scenario category mapping inferred from ID prefix
_SCENARIO_CATEGORIES: dict[str, str] = {
    "1.1": "Voluntary Date Change",
    "1.2": "Same-Day Change",
    "1.3": "Route Change",
    "2.1": "IRROPS Cancellation",
    "2.2": "IRROPS Delay",
    "2.3": "IRROPS Schedule Change",
    "2.4": "IRROPS Diversion",
    "3.1": "Missed Flight Recovery",
    "3.3": "Missed Flight Rebooking",
    "4.1": "Elite Same-Day Change",
    "4.2": "Standby Request",
    "5.1": "Full Refund Cancellation",
    "5.2": "Non-Refundable Cancellation",
    "6.1": "Complex IRROPS Rebooking",
    "6.3": "Complex Transfer/Escalation",
    "7.1": "Auth Bypass Attempt",
    "7.2": "Adversarial Policy Exploit",
    "7.3": "Entitlement Claim",
    "7.4": "Third-Party Lookup",
}

# Broader groupings
_CATEGORY_GROUPS: dict[str, str] = {
    "1.1": "Voluntary Changes",
    "1.2": "Voluntary Changes",
    "1.3": "Voluntary Changes",
    "2.1": "IRROPS",
    "2.2": "IRROPS",
    "2.3": "IRROPS",
    "2.4": "IRROPS",
    "3.1": "Missed Flights",
    "3.3": "Missed Flights",
    "4.1": "Elite Status",
    "4.2": "Elite Status",
    "5.1": "Cancellations & Refunds",
    "5.2": "Cancellations & Refunds",
    "6.1": "Complex Itineraries",
    "6.3": "Complex Itineraries",
    "7.1": "Adversarial",
    "7.2": "Adversarial",
    "7.3": "Adversarial",
    "7.4": "Adversarial",
}

_EVA_A_METRICS = ["task_completion", "faithfulness", "agent_speech_fidelity"]
_EVA_A_THRESHOLDS = {
    "task_completion": 1.0,
    "faithfulness": 0.5,
    "agent_speech_fidelity": 0.95,
}


def _scenario_category(record_id: str) -> str:
    """Map a record ID like '2.1.1' to its scenario category."""
    prefix = ".".join(record_id.split(".")[:2])
    return _SCENARIO_CATEGORIES.get(prefix, f"Unknown ({prefix})")


def _scenario_group(record_id: str) -> str:
    """Map a record ID to its broad category group."""
    prefix = ".".join(record_id.split(".")[:2])
    return _CATEGORY_GROUPS.get(prefix, "Other")


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

_PASS_COLOR = "#4CAF50"
_FAIL_COLOR = "#F44336"
_SKIP_COLOR = "#9E9E9E"
_GROUP_COLORS = {
    "Voluntary Changes": "#3366CC",
    "IRROPS": "#DC3912",
    "Missed Flights": "#FF9900",
    "Elite Status": "#109618",
    "Cancellations & Refunds": "#990099",
    "Complex Itineraries": "#0099C6",
    "Adversarial": "#DD4477",
    "Other": "#AAAAAA",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _get_run_directories(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    run_dirs = [d for d in output_dir.iterdir() if d.is_dir() and (d / "records").exists()]
    return sorted(run_dirs, key=lambda d: d.name, reverse=True)


def _load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _extract_system_label(run_config: dict) -> str:
    model_cfg = run_config.get("pipeline") or run_config.get("model") or {}
    if model_cfg.get("s2s") or model_cfg.get("realtime_model"):
        s2s_params = model_cfg.get("s2s_params") or {}
        return (
            s2s_params.get("alias")
            or s2s_params.get("model")
            or model_cfg.get("s2s")
            or model_cfg.get("realtime_model", "unknown")
        )
    parts = []
    stt_params = model_cfg.get("stt_params") or {}
    stt = stt_params.get("alias") or stt_params.get("model") or model_cfg.get("stt") or ""
    if stt:
        parts.append(stt)
    audio_llm = model_cfg.get("audio_llm") or ""
    if audio_llm:
        audio_llm_params = model_cfg.get("audio_llm_params") or {}
        parts.append(audio_llm_params.get("alias") or audio_llm_params.get("model") or audio_llm)
    else:
        llm = model_cfg.get("llm") or model_cfg.get("llm_model") or ""
        if llm:
            parts.append(llm)
    tts_params = model_cfg.get("tts_params") or {}
    tts = tts_params.get("alias") or tts_params.get("model") or model_cfg.get("tts") or ""
    if tts:
        parts.append(tts)
    return " + ".join(parts) if parts else "unknown"


def _classify_pipeline_type(run_config: dict) -> str:
    model_cfg = run_config.get("pipeline") or run_config.get("model") or {}
    if model_cfg.get("realtime_model") == "ultravox":
        return "Audio-Native"
    if model_cfg.get("s2s") or model_cfg.get("realtime_model"):
        return "S2S"
    if model_cfg.get("audio_llm"):
        return "Audio-Native"
    return "Cascade"


def _get_run_label(run_dir: Path, run_config: dict) -> str:
    m = re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+_(.+)$", run_dir.name)
    if m:
        return m.group(1)
    return _extract_system_label(run_config) or run_dir.name


def _get_record_data_dirs(record_dir: Path) -> list[tuple[str, Path]]:
    trial_dirs = (
        sorted(
            [
                d
                for d in record_dir.iterdir()
                if d.is_dir() and any(f for f in d.iterdir() if f.suffix in (".json", ".wav", ".jsonl"))
            ],
            key=lambda d: d.name,
        )
        if record_dir.exists()
        else []
    )
    if trial_dirs:
        return [(d.name, d) for d in trial_dirs]
    return [("", record_dir)]


def _load_record_metrics(data_path: Path) -> dict | None:
    metrics_path = data_path / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return None


def _load_all_run_data(run_dirs: list[Path]) -> pd.DataFrame:
    """Load task_completion + EVA-A metrics for all records across runs."""
    rows: list[dict] = []
    for run_dir in run_dirs:
        run_config = _load_run_config(run_dir)
        system = _get_run_label(run_dir, run_config)
        pipeline_type = _classify_pipeline_type(run_config)
        domain = run_config.get("domain", "airline")

        records_dir = run_dir / "records"
        if not records_dir.exists():
            continue

        for record_dir in sorted(records_dir.iterdir()):
            if not record_dir.is_dir():
                continue
            record_id = record_dir.name
            data_dirs = _get_record_data_dirs(record_dir)

            for trial_label, data_path in data_dirs:
                if "_failed_attempt_" in trial_label:
                    continue
                metrics_data = _load_record_metrics(data_path)
                if not metrics_data:
                    continue

                metrics = metrics_data.get("metrics", {})
                aggregates = metrics_data.get("aggregate_metrics", {})

                row: dict[str, Any] = {
                    "run": run_dir.name,
                    "system": system,
                    "pipeline_type": pipeline_type,
                    "domain": domain,
                    "record_id": record_id,
                    "trial": trial_label or "base",
                    "scenario_category": _scenario_category(record_id),
                    "scenario_group": _scenario_group(record_id),
                }

                # Extract EVA-A component scores
                for metric_name in _EVA_A_METRICS:
                    m = metrics.get(metric_name, {})
                    if m.get("error"):
                        row[metric_name] = None
                        row[f"{metric_name}_error"] = m.get("error")
                    elif m.get("skipped"):
                        row[metric_name] = None
                        row[f"{metric_name}_skipped"] = True
                    else:
                        ns = m.get("normalized_score")
                        row[metric_name] = ns if ns is not None else m.get("score")

                # Extract task_completion details (diff info)
                tc = metrics.get("task_completion", {})
                tc_details = tc.get("details", {})
                row["tc_match"] = tc_details.get("match", False)
                row["tc_auth_success"] = tc_details.get("auth_success", True)
                row["tc_message"] = tc_details.get("message", "")

                # Parse diff summary
                diff = tc_details.get("diff", {})
                if diff:
                    row["diff_tables_added"] = len(diff.get("tables_added", []))
                    row["diff_tables_removed"] = len(diff.get("tables_removed", []))
                    row["diff_tables_modified"] = len(diff.get("tables_modified", {}))
                    # Aggregate field-level changes
                    modified_tables = diff.get("tables_modified", {})
                    total_records_added = 0
                    total_records_removed = 0
                    total_records_modified = 0
                    modified_table_names = []
                    for tname, tdiff in modified_tables.items():
                        if isinstance(tdiff, dict):
                            total_records_added += len(tdiff.get("records_added", []))
                            total_records_removed += len(tdiff.get("records_removed", []))
                            total_records_modified += len(tdiff.get("records_modified", {}))
                            modified_table_names.append(tname)
                    row["diff_records_added"] = total_records_added
                    row["diff_records_removed"] = total_records_removed
                    row["diff_records_modified"] = total_records_modified
                    row["diff_modified_tables"] = ", ".join(modified_table_names)
                    row["diff_raw"] = diff
                else:
                    row["diff_tables_added"] = 0
                    row["diff_tables_removed"] = 0
                    row["diff_tables_modified"] = 0
                    row["diff_records_added"] = 0
                    row["diff_records_removed"] = 0
                    row["diff_records_modified"] = 0
                    row["diff_modified_tables"] = ""
                    row["diff_raw"] = {}

                # EVA-A aggregate
                row["EVA-A_pass"] = aggregates.get("EVA-A_pass")
                row["EVA-A_mean"] = aggregates.get("EVA-A_mean")

                rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def _classify_failure(row: pd.Series) -> str:
    """Classify the root cause of a task_completion failure."""
    if row.get("task_completion") == 1.0:
        return "Pass"
    if not row.get("tc_auth_success", True):
        return "Auth Failure"

    tables_added = row.get("diff_tables_added", 0)
    tables_removed = row.get("diff_tables_removed", 0)
    records_added = row.get("diff_records_added", 0)
    records_removed = row.get("diff_records_removed", 0)
    records_modified = row.get("diff_records_modified", 0)
    modified_tables = str(row.get("diff_modified_tables", ""))

    # Missing actions (expected tables/records not created)
    if tables_removed > 0 or records_removed > 0:
        return "Missing Actions"

    # Extra actions (unexpected tables/records created)
    if tables_added > 0 or records_added > 0:
        return "Extra Actions"

    # Wrong values in existing records
    if records_modified > 0:
        if "reservations" in modified_tables:
            return "Wrong Booking State"
        if any(t in modified_tables for t in ["meal_vouchers", "hotel_vouchers", "travel_credits"]):
            return "Wrong Voucher/Credit"
        if "refunds" in modified_tables:
            return "Wrong Refund"
        return "Wrong Field Values"

    if row.get("task_completion") is None:
        return "Error/Skipped"

    return "Other Mismatch"


_FAILURE_COLORS = {
    "Pass": "#4CAF50",
    "Auth Failure": "#9C27B0",
    "Missing Actions": "#F44336",
    "Extra Actions": "#FF9800",
    "Wrong Booking State": "#E91E63",
    "Wrong Voucher/Credit": "#FF5722",
    "Wrong Refund": "#795548",
    "Wrong Field Values": "#607D8B",
    "Other Mismatch": "#9E9E9E",
    "Error/Skipped": "#BDBDBD",
}


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def _render_overview_tab(df: pd.DataFrame):
    """Tab 1: High-level EVA-A overview with task completion rates."""
    st.markdown("### EVA-A Task Completion Overview")

    systems = df["system"].unique()
    if len(systems) == 0:
        st.warning("No data loaded.")
        return

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        total = len(df)
        st.metric("Total Records", total)
    with cols[1]:
        n_systems = len(systems)
        st.metric("Systems", n_systems)
    with cols[2]:
        pass_rate = df["task_completion"].dropna().mean()
        st.metric("Avg Task Completion", f"{pass_rate:.1%}")
    with cols[3]:
        eva_a_pass_rate = df["EVA-A_pass"].dropna().mean()
        st.metric("Avg EVA-A Pass", f"{eva_a_pass_rate:.1%}")

    st.divider()

    # Per-system task completion rate bar chart
    sys_rates = (
        df.groupby("system")
        .agg(
            tc_mean=("task_completion", "mean"),
            tc_count=("task_completion", "count"),
            faith_mean=("faithfulness", "mean"),
            fidelity_mean=("agent_speech_fidelity", "mean"),
            eva_a_pass=("EVA-A_pass", "mean"),
            pipeline=("pipeline_type", "first"),
        )
        .reset_index()
        .sort_values("tc_mean", ascending=True)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=sys_rates["system"],
            x=sys_rates["tc_mean"],
            orientation="h",
            name="Task Completion",
            marker_color="#3366CC",
            hovertemplate="<b>%{y}</b><br>Task Completion: %{x:.3f}<br>n=%{customdata}<extra></extra>",
            customdata=sys_rates["tc_count"],
        )
    )
    fig.add_trace(
        go.Bar(
            y=sys_rates["system"],
            x=sys_rates["faith_mean"],
            orientation="h",
            name="Faithfulness",
            marker_color="#FF9900",
            hovertemplate="<b>%{y}</b><br>Faithfulness: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=sys_rates["system"],
            x=sys_rates["fidelity_mean"],
            orientation="h",
            name="Agent Speech Fidelity",
            marker_color="#22AA99",
            hovertemplate="<b>%{y}</b><br>Speech Fidelity: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="EVA-A Component Scores by System",
        barmode="group",
        xaxis={"title": "Score", "range": [0, 1.05]},
        height=max(350, len(sys_rates) * 60 + 100),
        margin={"l": 10, "r": 10, "t": 40, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    st.plotly_chart(fig, use_container_width=True)

    # EVA-A pass rate comparison
    st.markdown("#### EVA-A Pass Rate (all 3 components must pass)")
    eva_a_rates = sys_rates.sort_values("eva_a_pass", ascending=True)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            y=eva_a_rates["system"],
            x=eva_a_rates["eva_a_pass"],
            orientation="h",
            marker_color=[_PASS_COLOR if v >= 0.3 else _FAIL_COLOR for v in eva_a_rates["eva_a_pass"]],
            hovertemplate="<b>%{y}</b><br>EVA-A pass@1: %{x:.3f}<extra></extra>",
        )
    )
    fig2.update_layout(
        xaxis={"title": "EVA-A pass@1", "range": [0, 1.05]},
        height=max(300, len(eva_a_rates) * 40 + 80),
        margin={"l": 10, "r": 10, "t": 10, "b": 40},
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_scenario_heatmap_tab(df: pd.DataFrame):
    """Tab 2: Per-scenario pass/fail heatmap across systems."""
    st.markdown("### Task Completion Heatmap")
    st.caption("Rows = systems, columns = scenario IDs. Green = pass (1.0), red = fail (0.0).")

    # Build pivot: system x record_id
    pivot = df.pivot_table(
        index="system",
        columns="record_id",
        values="task_completion",
        aggfunc="mean",
    )

    if pivot.empty:
        st.warning("No task completion data to display.")
        return

    # Sort columns by scenario ID
    sorted_cols = sorted(pivot.columns, key=lambda x: [int(p) for p in x.split(".")])
    pivot = pivot[sorted_cols]

    # Sort rows by mean task completion (best at top)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values.tolist(),
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[[0, _FAIL_COLOR], [0.5, "#FFEB3B"], [1, _PASS_COLOR]],
            zmin=0,
            zmax=1,
            colorbar={"title": "Score"},
            hovertemplate="System: %{y}<br>Scenario: %{x}<br>Score: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis={"title": "Scenario ID", "tickangle": -45, "dtick": 1},
        yaxis={"title": "System"},
        height=max(400, len(pivot) * 40 + 100),
        margin={"l": 10, "r": 10, "t": 10, "b": 80},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scenario difficulty ranking
    st.markdown("#### Scenario Difficulty Ranking")
    st.caption("Scenarios sorted by pass rate across all systems (hardest first).")

    scenario_rates = pivot.mean(axis=0).reset_index()
    scenario_rates.columns = ["Scenario", "Pass Rate"]
    scenario_rates["Category"] = scenario_rates["Scenario"].apply(_scenario_category)
    scenario_rates["Group"] = scenario_rates["Scenario"].apply(_scenario_group)
    scenario_rates = scenario_rates.sort_values("Pass Rate", ascending=True)

    fig3 = go.Figure()
    for group in scenario_rates["Group"].unique():
        group_data = scenario_rates[scenario_rates["Group"] == group]
        fig3.add_trace(
            go.Bar(
                y=group_data["Scenario"],
                x=group_data["Pass Rate"],
                orientation="h",
                name=group,
                marker_color=_GROUP_COLORS.get(group, "#AAAAAA"),
                hovertemplate="<b>%{y}</b> (%{customdata})<br>Pass Rate: %{x:.1%}<extra></extra>",
                customdata=group_data["Category"],
            )
        )
    fig3.update_layout(
        title="Scenario Pass Rate (across all systems)",
        xaxis={"title": "Pass Rate", "range": [0, 1.05], "tickformat": ".0%"},
        height=max(500, len(scenario_rates) * 22 + 100),
        margin={"l": 10, "r": 10, "t": 40, "b": 40},
        barmode="stack",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    st.plotly_chart(fig3, use_container_width=True)


def _render_failure_analysis_tab(df: pd.DataFrame):
    """Tab 3: Failure root cause analysis."""
    st.markdown("### Failure Root Cause Analysis")

    # Classify failures
    df = df.copy()
    df["failure_type"] = df.apply(_classify_failure, axis=1)

    failed = df[df["failure_type"] != "Pass"]

    if failed.empty:
        st.success("All tasks passed! No failures to analyze.")
        return

    # Overall failure type distribution
    col1, col2 = st.columns(2)

    with col1:
        failure_counts = failed["failure_type"].value_counts().reset_index()
        failure_counts.columns = ["Failure Type", "Count"]
        fig = px.pie(
            failure_counts,
            values="Count",
            names="Failure Type",
            title="Failure Type Distribution",
            color="Failure Type",
            color_discrete_map=_FAILURE_COLORS,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Failure type by scenario group
        group_failures = failed.groupby(["scenario_group", "failure_type"]).size().reset_index(name="count")
        fig2 = px.bar(
            group_failures,
            x="scenario_group",
            y="count",
            color="failure_type",
            title="Failure Types by Scenario Group",
            color_discrete_map=_FAILURE_COLORS,
        )
        fig2.update_layout(
            xaxis={"title": "Scenario Group", "tickangle": -30},
            yaxis={"title": "Count"},
            height=400,
            legend={"title": "Failure Type"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Per-system failure breakdown
    st.markdown("#### Failure Breakdown by System")
    sys_failures = df.groupby(["system", "failure_type"]).size().reset_index(name="count")
    fig3 = px.bar(
        sys_failures,
        x="system",
        y="count",
        color="failure_type",
        title="Pass/Fail Breakdown per System",
        color_discrete_map=_FAILURE_COLORS,
        barmode="stack",
    )
    fig3.update_layout(
        xaxis={"title": "", "tickangle": -30},
        yaxis={"title": "Count"},
        height=450,
        legend={"title": "Outcome", "orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Most commonly modified tables in failures
    st.markdown("#### Most Commonly Mismatched DB Tables")
    table_counts: dict[str, int] = defaultdict(int)
    for _, row in failed.iterrows():
        for t in str(row.get("diff_modified_tables", "")).split(", "):
            t = t.strip()
            if t:
                table_counts[t] += 1
    if table_counts:
        table_df = pd.DataFrame(
            sorted(table_counts.items(), key=lambda x: x[1], reverse=True),
            columns=["Table", "Failure Count"],
        )
        fig4 = px.bar(
            table_df,
            x="Failure Count",
            y="Table",
            orientation="h",
            color="Failure Count",
            color_continuous_scale="Reds",
        )
        fig4.update_layout(
            height=max(250, len(table_df) * 35 + 80),
            margin={"l": 10, "r": 10, "t": 10, "b": 40},
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Detailed failure table
    st.markdown("#### Failed Records Detail")
    detail_cols = [
        "system",
        "record_id",
        "scenario_category",
        "failure_type",
        "tc_message",
        "diff_tables_modified",
        "diff_records_added",
        "diff_records_removed",
        "diff_records_modified",
        "diff_modified_tables",
    ]
    available_cols = [c for c in detail_cols if c in failed.columns]
    st.dataframe(
        failed[available_cols].sort_values(["system", "record_id"]),
        hide_index=True,
        use_container_width=True,
    )


def _render_component_correlation_tab(df: pd.DataFrame):
    """Tab 4: EVA-A component correlation analysis."""
    st.markdown("### EVA-A Component Correlation")
    st.caption("How do the three EVA-A components relate to each other? Each dot is one (system, scenario) evaluation.")

    valid = df.dropna(subset=["task_completion", "faithfulness", "agent_speech_fidelity"])
    if valid.empty:
        st.warning("Insufficient data for correlation analysis.")
        return

    # Scatter: Task Completion vs Faithfulness
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.scatter(
            valid,
            x="faithfulness",
            y="task_completion",
            color="system",
            title="Task Completion vs Faithfulness",
            opacity=0.6,
            hover_data=["record_id", "scenario_category"],
        )
        fig1.update_layout(
            xaxis={"title": "Faithfulness", "range": [-0.05, 1.05]},
            yaxis={"title": "Task Completion", "range": [-0.05, 1.05]},
            height=450,
            showlegend=False,
        )
        # Add threshold lines
        fig1.add_hline(y=_EVA_A_THRESHOLDS["task_completion"], line_dash="dash", line_color="gray", opacity=0.5)
        fig1.add_vline(x=_EVA_A_THRESHOLDS["faithfulness"], line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            valid,
            x="agent_speech_fidelity",
            y="task_completion",
            color="system",
            title="Task Completion vs Speech Fidelity",
            opacity=0.6,
            hover_data=["record_id", "scenario_category"],
        )
        fig2.update_layout(
            xaxis={"title": "Agent Speech Fidelity", "range": [-0.05, 1.05]},
            yaxis={"title": "Task Completion", "range": [-0.05, 1.05]},
            height=450,
            showlegend=False,
        )
        fig2.add_hline(y=_EVA_A_THRESHOLDS["task_completion"], line_dash="dash", line_color="gray", opacity=0.5)
        fig2.add_vline(x=_EVA_A_THRESHOLDS["agent_speech_fidelity"], line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    # Which component is the bottleneck?
    st.markdown("#### EVA-A Bottleneck Analysis")
    st.caption(
        "For each record that fails EVA-A, which component(s) are below threshold? "
        "This identifies whether task_completion, faithfulness, or speech_fidelity "
        "is the primary bottleneck."
    )

    eva_a_failed = valid[valid.get("EVA-A_pass", pd.Series(dtype=float)) == 0.0].copy()
    if eva_a_failed.empty:
        st.info("No EVA-A failures to analyze.")
        return

    bottleneck_counts: dict[str, int] = defaultdict(int)
    for _, row in eva_a_failed.iterrows():
        for metric, threshold in _EVA_A_THRESHOLDS.items():
            val = row.get(metric)
            if val is not None and val < threshold:
                bottleneck_counts[metric] += 1

    if bottleneck_counts:
        bn_df = pd.DataFrame(
            sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True),
            columns=["Metric", "Times Below Threshold"],
        )
        bn_df["Threshold"] = bn_df["Metric"].map(_EVA_A_THRESHOLDS)
        bn_df["% of Failures"] = bn_df["Times Below Threshold"] / len(eva_a_failed)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(
                bn_df.style.format({"% of Failures": "{:.1%}", "Threshold": "{:.2f}"}),
                hide_index=True,
            )
        with col2:
            fig_bn = px.bar(
                bn_df,
                x="Metric",
                y="Times Below Threshold",
                color="Metric",
                title="Bottleneck Frequency in EVA-A Failures",
            )
            fig_bn.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_bn, use_container_width=True)


def _render_category_deep_dive_tab(df: pd.DataFrame):
    """Tab 5: Category-level deep dive."""
    st.markdown("### Scenario Category Deep Dive")

    # Per-group pass rates
    group_rates = (
        df.groupby("scenario_group")
        .agg(
            tc_mean=("task_completion", "mean"),
            tc_count=("task_completion", "count"),
            n_scenarios=("record_id", "nunique"),
        )
        .reset_index()
        .sort_values("tc_mean", ascending=False)
    )

    fig = go.Figure()
    for _, row in group_rates.iterrows():
        group = row["scenario_group"]
        fig.add_trace(
            go.Bar(
                x=[group],
                y=[row["tc_mean"]],
                name=group,
                marker_color=_GROUP_COLORS.get(group, "#AAAAAA"),
                hovertemplate=(
                    f"<b>{group}</b><br>"
                    f"Pass Rate: {row['tc_mean']:.1%}<br>"
                    f"Scenarios: {int(row['n_scenarios'])}<br>"
                    f"Total Evals: {int(row['tc_count'])}"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Task Completion Rate by Scenario Group",
        yaxis={"title": "Pass Rate", "range": [0, 1.05], "tickformat": ".0%"},
        height=400,
        showlegend=False,
        margin={"l": 60, "r": 10, "t": 40, "b": 60},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed category breakdown
    st.markdown("#### Per-Category Breakdown")
    selected_group = st.selectbox(
        "Select scenario group",
        sorted(df["scenario_group"].unique()),
    )

    group_df = df[df["scenario_group"] == selected_group]

    # Heatmap for selected group
    pivot = group_df.pivot_table(
        index="system",
        columns="record_id",
        values="task_completion",
        aggfunc="mean",
    )
    if not pivot.empty:
        sorted_cols = sorted(pivot.columns, key=lambda x: [int(p) for p in x.split(".")])
        pivot = pivot[sorted_cols]
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        fig2 = go.Figure(
            data=go.Heatmap(
                z=pivot.values.tolist(),
                x=[f"{c}\n{_scenario_category(c)}" for c in pivot.columns],
                y=list(pivot.index),
                colorscale=[[0, _FAIL_COLOR], [0.5, "#FFEB3B"], [1, _PASS_COLOR]],
                zmin=0,
                zmax=1,
                hovertemplate="System: %{y}<br>Scenario: %{x}<br>Score: %{z:.2f}<extra></extra>",
            )
        )
        fig2.update_layout(
            title=f"Task Completion: {selected_group}",
            height=max(300, len(pivot) * 40 + 100),
            margin={"l": 10, "r": 10, "t": 40, "b": 120},
            xaxis={"tickangle": -30},
        )
        st.plotly_chart(fig2, use_container_width=True)


def _render_diff_inspector_tab(df: pd.DataFrame):
    """Tab 6: Interactive DB diff inspector for failed records."""
    st.markdown("### DB Diff Inspector")
    st.caption("Inspect the database state diff for failed task completions.")

    failed = df[df["task_completion"] == 0.0].copy()
    if failed.empty:
        st.success("No failed records to inspect.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        selected_system = st.selectbox(
            "System",
            ["All"] + sorted(failed["system"].unique()),
            key="diff_system",
        )
    with col2:
        selected_category = st.selectbox(
            "Scenario Group",
            ["All"] + sorted(failed["scenario_group"].unique()),
            key="diff_category",
        )

    if selected_system != "All":
        failed = failed[failed["system"] == selected_system]
    if selected_category != "All":
        failed = failed[failed["scenario_group"] == selected_category]

    if failed.empty:
        st.info("No matching failed records.")
        return

    # Select a specific record
    record_options = failed.apply(
        lambda r: f"{r['system']} | {r['record_id']} ({r['scenario_category']})",
        axis=1,
    ).tolist()

    selected_idx = st.selectbox(
        "Select failed record",
        range(len(record_options)),
        format_func=lambda i: record_options[i],
        key="diff_record",
    )

    row = failed.iloc[selected_idx]

    # Display info
    st.markdown(
        f"**System:** {row['system']}  |  **Scenario:** {row['record_id']}  |  **Category:** {row['scenario_category']}"
    )
    st.markdown(f"**Message:** {row.get('tc_message', 'N/A')}")

    # Classify
    failure_type = _classify_failure(row)
    color = _FAILURE_COLORS.get(failure_type, "#9E9E9E")
    st.markdown(
        f"**Failure Type:** <span style='color:{color};font-weight:bold'>{failure_type}</span>", unsafe_allow_html=True
    )

    # Display diff
    diff = row.get("diff_raw", {})
    if not diff:
        st.info("No diff data available for this record.")
        return

    # Summary stats
    cols = st.columns(5)
    cols[0].metric("Tables Added", len(diff.get("tables_added", [])))
    cols[1].metric("Tables Removed", len(diff.get("tables_removed", [])))
    cols[2].metric("Tables Modified", len(diff.get("tables_modified", {})))
    cols[3].metric("Records Added", row.get("diff_records_added", 0))
    cols[4].metric("Records Removed", row.get("diff_records_removed", 0))

    # Show tables added/removed
    if diff.get("tables_added"):
        st.warning(f"**Unexpected tables added:** {', '.join(diff['tables_added'])}")
    if diff.get("tables_removed"):
        st.error(f"**Expected tables missing:** {', '.join(diff['tables_removed'])}")

    # Show per-table modifications
    for table_name, table_diff in diff.get("tables_modified", {}).items():
        with st.expander(f"Table: `{table_name}`", expanded=True):
            if not isinstance(table_diff, dict):
                st.json(table_diff)
                continue

            if table_diff.get("records_added"):
                st.markdown(f"**Records added (unexpected):** `{table_diff['records_added']}`")
            if table_diff.get("records_removed"):
                st.markdown(f"**Records removed (missing):** `{table_diff['records_removed']}`")

            for rec_key, rec_diff in table_diff.get("records_modified", {}).items():
                st.markdown(f"**Record `{rec_key}` changes:**")
                st.json(rec_diff)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="EVA-A Task Completion Analysis",
        page_icon="",
        layout="wide",
    )

    st.title("EVA-A Task Completion Analysis")
    st.caption("Deep-dive into task completion metrics, failure patterns, and scenario difficulty.")

    # Sidebar: output directory
    output_dir = st.sidebar.text_input(
        "Output Directory",
        value=_DEFAULT_OUTPUT_DIR,
        help="Path to directory containing EVA run outputs",
    )
    output_path = Path(output_dir)

    # Load runs
    run_dirs = _get_run_directories(output_path)
    if not run_dirs:
        st.warning(
            f"No run directories found in `{output_dir}`. "
            "Make sure the directory contains EVA benchmark outputs with a `records/` subdirectory.\n\n"
            "Set `EVA_OUTPUT_DIR` environment variable or enter the path in the sidebar."
        )
        st.info(
            "**Expected directory structure:**\n"
            "```\n"
            "output/\n"
            "  <run_id>/\n"
            "    config.json\n"
            "    records/\n"
            "      <record_id>/\n"
            "        metrics.json\n"
            "```"
        )
        return

    # Sidebar: run selection
    st.sidebar.markdown("### Run Selection")
    run_labels = {}
    for d in run_dirs:
        cfg = _load_run_config(d)
        run_labels[d.name] = _get_run_label(d, cfg)

    selected_runs = st.sidebar.multiselect(
        "Select runs to analyze",
        [d.name for d in run_dirs],
        default=[d.name for d in run_dirs[:5]],
        format_func=lambda x: run_labels.get(x, x),
    )

    if not selected_runs:
        st.info("Select at least one run from the sidebar.")
        return

    selected_dirs = [d for d in run_dirs if d.name in selected_runs]

    # Load data
    with st.spinner("Loading metrics data..."):
        df = _load_all_run_data(selected_dirs)

    if df.empty:
        st.warning("No metrics data found in the selected runs.")
        return

    st.sidebar.markdown(f"**Records loaded:** {len(df)}")
    st.sidebar.markdown(f"**Systems:** {df['system'].nunique()}")
    st.sidebar.markdown(f"**Scenarios:** {df['record_id'].nunique()}")

    # Tabs
    tabs = st.tabs(
        [
            "Overview",
            "Scenario Heatmap",
            "Failure Analysis",
            "Component Correlation",
            "Category Deep Dive",
            "Diff Inspector",
        ]
    )

    with tabs[0]:
        _render_overview_tab(df)
    with tabs[1]:
        _render_scenario_heatmap_tab(df)
    with tabs[2]:
        _render_failure_analysis_tab(df)
    with tabs[3]:
        _render_component_correlation_tab(df)
    with tabs[4]:
        _render_category_deep_dive_tab(df)
    with tabs[5]:
        _render_diff_inspector_tab(df)


if __name__ == "__main__":
    main()
