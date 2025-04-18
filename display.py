import streamlit as st
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


COLOR_PALETTE = plt.cm.tab10.colors 
plt.style.use('seaborn-v0_8')

def load_strategy_data(dataset_name, strategy_name):
    file_path = f"results/{dataset_name}/{strategy_name}/results.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None

def get_available_datasets():
    if not os.path.exists("results"):
        return []
    return [d for d in os.listdir("results") if os.path.isdir(f"results/{d}")]

def get_available_strategies(dataset_name):
    path = f"results/{dataset_name}"
    if not os.path.exists(path):
        return []
    return [s for s in os.listdir(path) if os.path.isdir(f"{path}/{s}")]

def main():
    st.title("📊 Active Learning Results Dashboard")

    datasets = get_available_datasets()
    if not datasets:
        st.error("No datasets found in 'results/' folder.")
        return

    dataset_name = st.sidebar.selectbox("Select Dataset", datasets)

    strategies = get_available_strategies(dataset_name)
    selected_strategies = st.sidebar.multiselect("Select Strategies", strategies, default=strategies)

    if not selected_strategies:
        st.warning("Select at least one strategy to visualize.")
        return

    strategy_data = {}
    for strategy in selected_strategies:
        data = load_strategy_data(dataset_name, strategy)
        if data:
            strategy_data[strategy] = data

    if not strategy_data:
        st.error("No valid data found for selected strategies.")
        return

    max_cycle_len = max(len(data['cycles']) for data in strategy_data.values())
    cycle_range = st.sidebar.slider("Select Cycle Range", 1, max_cycle_len, (1, max_cycle_len))
    cycle_start, cycle_end = cycle_range
    selected_indices = list(range(cycle_start - 1, cycle_end))

    st.subheader("🧮 Plot Options")
    col1, col2 = st.columns(2)
    with col1:
        show_accuracy = st.checkbox("Show Accuracy", True)
    with col2:
        show_loss = st.checkbox("Show Loss", True)

    st.subheader("Performance Plot")
    fig, ax = plt.subplots(figsize=(10, 6))

    strategy_colors = {strategy: COLOR_PALETTE[i % len(COLOR_PALETTE)] 
                      for i, strategy in enumerate(selected_strategies)}

    for strategy, data in strategy_data.items():
        cycles = np.array(data["cycles"])[selected_indices]
        color = strategy_colors[strategy]
        
        if show_accuracy and "accuracies" in data:
            acc = np.array(data["accuracies"])[selected_indices]
            ax.plot(cycles, acc, 
                   label=f"{strategy} Accuracy", 
                   color=color,
                   marker='o',
                   linewidth=2)
        
        if show_loss and "losses" in data:
            loss = np.array(data["losses"])[selected_indices]
            ax.plot(cycles, loss, 
                   linestyle='--', 
                   label=f"{strategy} Loss", 
                   color=color,
                   marker='s',
                   linewidth=2)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Metric")
    ax.set_title(f"{dataset_name} - Accuracy & Loss Over Selected Cycles")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.tight_layout() 
    st.pyplot(fig)


    if show_accuracy:
        st.subheader("📈 Accuracy Table")
        acc_table = pd.DataFrame(index=range(cycle_start, cycle_end + 1))
        for strategy, data in strategy_data.items():
            acc_values = np.array(data["accuracies"])[selected_indices]
            acc_table[strategy] = acc_values
        acc_table.index.name = "Cycle"
        st.dataframe(acc_table.style.format("{:.4f}"))


    if show_loss:
        st.subheader("📉 Loss Table")
        loss_table = pd.DataFrame(index=range(cycle_start, cycle_end + 1))
        for strategy, data in strategy_data.items():
            loss_values = np.array(data["losses"])[selected_indices]
            loss_table[strategy] = loss_values
        loss_table.index.name = "Cycle"
        st.dataframe(loss_table.style.format("{:.4f}"))

if __name__ == "__main__":
    main()