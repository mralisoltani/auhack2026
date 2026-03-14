"""Supply & demand plots — Notebook 02."""
import streamlit as st
import matplotlib.pyplot as plt

from dashboard.utils import naive_index, format_date_axis, tight_layout
from src.data_loader import load_total_load, load_generation_pivot


def plot_supply_demand(zone: str, start: str, end: str) -> None:
    """Load vs Generation."""
    load = load_total_load(zone)
    gen = load_generation_pivot(zone)
    gen["total_gen"] = gen.sum(axis=1)
    l_sample = naive_index(load.loc[start:end])
    g_sample = naive_index(gen.loc[start:end])
    if l_sample.empty or g_sample.empty:
        st.warning("No data for selected zone/range.")
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    l_sample["load"].plot(ax=ax, label="Load", color="C0")
    g_sample["total_gen"].plot(ax=ax, label="Generation", color="C1", alpha=0.8)
    ax.set_ylabel("MW")
    ax.set_title(f"{zone}: Load vs Generation")
    ax.legend()
    format_date_axis(ax)
    tight_layout(fig)
    st.pyplot(fig)
    plt.close()
