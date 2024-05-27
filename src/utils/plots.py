import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List
from scipy.io import wavfile
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 1200


import matplotlib.pyplot as plt


def plot_with_separate_axis_for_pitch(
    table_data, title, y_label, pitch_y_label, log_scale=False
):
    # Set plot style
    sns.set(style="whitegrid")

    # Number of features
    num_features = len(table_data.keys())

    # Create a gridspec for subplots
    gs = gridspec.GridSpec(1, num_features)

    # To store legend elements
    legend_elements = []

    # Create first 5 subplots
    for i, (feature, data) in enumerate(list(table_data.items())[:-2]):
        ax = plt.subplot(gs[i])

        # Prepare the data
        models = list(data.keys())
        uncertainty_coeffs = list(data.values())

        num_models = len(models)
        colors = plt.cm.viridis(np.linspace(0, 1, num_models))

        x = np.arange(len(models))  # label locations
        width = 0.8  # width of the bars

        # For each model, plot the uncertainty coefficient
        for j, (model, uncertainty_coeff) in enumerate(zip(models, uncertainty_coeffs)):
            color = colors[j]  # consistent color for each model
            bars = ax.bar(x[j], uncertainty_coeff, width, color=color, label=model)

            # Add to legend elements (only for the first subplot to avoid duplicates)
            if i == 0:
                legend_elements.append(bars)

        # Add title for subfigure
        ax.set_title(feature, fontsize=14, fontweight="bold")

        # Remove x-ticks for all but the last plot
        ax.set_xticks([])

        # y-logscale
        if log_scale:
            ax.set_yscale("log")

    # Create a separate subplot for pitch with secondary y-axis
    ax_pitch = plt.subplot(gs[-1])
    pitch_data = table_data["Pitch"]
    models = list(pitch_data.keys())
    uncertainty_coeffs = list(pitch_data.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    x = np.arange(len(models))
    width = 0.8
    for j, (model, uncertainty_coeff) in enumerate(zip(models, uncertainty_coeffs)):
        ax_pitch.bar(x[j], uncertainty_coeff, width, color=colors[j])
    ax_pitch.set_title("Pitch", fontsize=14, fontweight="bold")
    ax_pitch.set_xticks([])
    ax_pitch2 = ax_pitch.twinx()
    ax_pitch2.set_ylabel(pitch_y_label, fontsize=18, fontweight="bold")
    if log_scale:
        ax_pitch2.set_yscale("log")

    # Add some text for labels and main title
    fig = plt.gcf()
    fig.suptitle(
        title,
        fontsize=20,
        fontweight="bold",
        x=0.5,
        y=1.05,  # Adjust title position
    )

    # Add legend outside of the subplots
    fig.legend(
        handles=[item for sublist in legend_elements for item in sublist],
        labels=models,
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=len(models),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])  # adjust layout for main title and legend

    plt.show()


def plot_multiple_uncertainty_coeff_sep_axis(
    table_data, title, y_label, log_scale=False
):
    # Set plot style
    sns.set(style="whitegrid")

    # Number of features
    num_features = len(table_data.keys())

    # Create subplots
    fig, axs = plt.subplots(1, num_features, figsize=(15, 6))

    # To store legend elements
    legend_elements = []

    # Iterate through each feature
    for ax, (feature, data) in zip(axs, table_data.items()):
        # Prepare the data
        models = list(data.keys())
        uncertainty_coeffs = list(data.values())

        num_models = len(models)
        colors = plt.cm.viridis(np.linspace(0, 1, num_models))  # same colors as before

        x = np.arange(len(models))  # label locations
        width = 0.8  # width of the bars

        # For each model, plot the uncertainty coefficient
        for i, (model, uncertainty_coeff) in enumerate(zip(models, uncertainty_coeffs)):
            color = colors[i]  # consistent color for each model
            bars = ax.bar(x[i], uncertainty_coeff, width, color=color, label=model)

            # Add to legend elements (only for the first subplot to avoid duplicates)
            if feature == list(table_data.keys())[0]:
                legend_elements.append(bars)

        # Add title for subfigure
        ax.set_title(feature, fontsize=14, fontweight="bold")

        # Remove x-ticks
        ax.set_xticks([])

        # y-logscale
        if log_scale:
            ax.set_yscale("log")

        # if feature == "Pitch": then y_range (0,8), else y_range (0,1)
        if feature == "Pitch":
            ax.set_ylim([0, 8])
            ax.y_label = "Mutual Information (bits)"
        else:
            ax.set_ylim([0, 1])

    # Add some text for labels and main title
    axs[0].set_ylabel(y_label, fontsize=18, fontweight="bold")
    fig.suptitle(
        title,
        fontsize=20,
        fontweight="bold",
        x=0.5,
        y=1.05,  # Adjust title position
    )

    # Add legend outside of the subplots
    fig.legend(
        handles=[item for sublist in legend_elements for item in sublist],
        labels=models,
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=num_models,
    )

    # plt.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])  # adjust layout for main title and legend

    plt.show()


def plot_multiple_uncertainty_coeff_same_axis(
    table_data, title, y_label, log_scale=False
):
    # Set font family to Times New Roman
    plt.rc("font", family="serif", serif="Times New Roman")

    # Set plot style
    sns.set(style="whitegrid")

    # Number of features
    num_features = len(table_data.keys())

    # Create subplots
    fig, axs = plt.subplots(1, num_features, figsize=(15, 6), sharey=True)

    # Iterate through each feature
    for ax, (feature, data) in zip(axs, table_data.items()):
        # Prepare the data
        models = list(data.keys())
        uncertainty_coeffs = list(data.values())

        num_models = len(models)
        colors = plt.cm.viridis(np.linspace(0, 1, num_models))  # same colors as before

        x = np.arange(len(models))  # label locations
        width = 0.8  # width of the bars

        # For each model, plot the uncertainty coefficient
        for i, (model, uncertainty_coeff) in enumerate(zip(models, uncertainty_coeffs)):
            color = colors[i]  # consistent color for each model
            ax.bar(x[i], uncertainty_coeff, width, color=color)

        # Add title for subfigure
        ax.set_title(feature, fontsize=14, fontweight="bold")

        # x-axis tick labels for subfigure
        ax.set_xticks(x)
        ax.set_xticklabels(
            models, fontsize=10, rotation=30, ha="right", fontweight="bold"
        )  # rotated and right-aligned

    # Add some text for labels and main title
    axs[0].set_ylabel(y_label, fontsize=18, fontweight="bold")
    fig.suptitle(
        title,
        fontsize=20,
        fontweight="bold",
    )

    # y-logscale
    if log_scale:
        axs[0].set_yscale("log")

    # plt.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # adjust layout for main title

    plt.show()


def plot_uncertainty_coeff(table_data, log_scale=False):
    # Set plot style
    sns.set(style="whitegrid")

    # Prepare the data
    features = list(table_data.keys())
    models = list(table_data[features[0]]["models"].keys())
    num_models = len(models)
    colors = plt.cm.viridis(np.linspace(0, 1, num_models))  # same colors as before

    x = np.arange(len(features))  # label locations
    width = 0.15  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, feature in enumerate(features):
        # Calculate mutual info values and sort models by these values

        mutual_info_values = [
            (
                model,
                (
                    table_data[feature]["entropy"]
                    - (
                        table_data[feature]["models"].get(model, 0)
                        if table_data[feature]["models"].get(model) is not None
                        else 0
                    )
                )
                / (table_data[feature]["entropy_lower_bound"]),
            )
            for model in models
        ]

        # mutual_info_values.sort(key=lambda x: x[1])

        # print(mutual_info_values)

        # For each model, plot the mutual info for the feature
        for j, (model, mutual_info) in enumerate(mutual_info_values):
            color = colors[models.index(model)]  # consistent color for each model
            ax.bar(x[i] - width / 2 + j * width, mutual_info, width, color=color)

    print(mutual_info_values)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_xlabel("Prosodic Features", fontsize=18, fontweight="bold")
    ax.set_ylabel("Uncertainty Coefficient", fontsize=18, fontweight="bold")
    ax.set_title(
        "Baselined Uncertainty Coefficient by Model and Feature",
        fontsize=20,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        features, fontsize=14, rotation=30, ha="right", fontweight="bold"
    )  # rotated and right-aligned
    ax.tick_params(axis="y", labelsize=14)  # y-axis labels fontsize

    # y-logscale
    if log_scale:
        ax.set_yscale("log")

    # Manually create legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=4, label=model)
        for model, color in zip(models, colors)
    ]
    ax.legend(handles=legend_elements, fontsize=14)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.figure(figsize=(10, 3))

    fig.tight_layout()

    plt.show()


def plot_mutual_info_and_r2(mutual_info_data, r2_data):
    # Prepare the data
    features = list(mutual_info_data.keys())
    models = list(mutual_info_data[features[0]]["models"].keys())
    # Ensure models are in the desired order
    models = ["GloVe", "fastText"] + [
        model for model in models if model not in ["GloVe", "fastText"]
    ]
    num_models = len(models)
    colors = plt.cm.viridis(
        np.linspace(1, 0, num_models)
    )  # reversed so dark color is the left bar

    x = np.arange(len(features))  # label locations
    width = 0.15  # width of the bars

    # Create a single figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Subplot for Mutual Information
    ax = axes[0]
    for i, feature in enumerate(features):
        # Calculate mutual info values and sort models by these values
        mutual_info_values = [
            (
                model,
                mutual_info_data[feature]["entropy"]
                - (
                    mutual_info_data[feature]["models"].get(model, 0)
                    if mutual_info_data[feature]["models"].get(model) is not None
                    else 0
                ),
            )
            for model in models
        ]
        mutual_info_values.sort(key=lambda x: x[1])

        # For each model, plot the mutual info for the feature
        for j, (model, mutual_info) in enumerate(mutual_info_values):
            color = colors[models.index(model)]
            ax.bar(x[i] - width / 2 + j * width, mutual_info, width, color=color)

    ax.get_xaxis().set_visible(False)
    ax.set_ylabel("Mutual Information", fontsize=16)
    ax.set_title(
        "Mutual Information and R2 by Model and Feature", fontsize=18, fontweight="bold"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot for R2
    ax = axes[1]
    for i, feature in enumerate(features):
        # Get R2 values and sort models by these values
        r2_values = [
            (
                model.replace(
                    " R2", ""
                ),  # remove " R2" from model name for consistency
                r2_data[feature].get(model, 0)
                if r2_data[feature].get(model) is not None
                else 0,
            )
            for model in r2_data[feature].keys()
        ]
        r2_values.sort(key=lambda x: x[1])

        # For each model, plot the R2 for the feature
        for j, (model, r2) in enumerate(r2_values):
            color = colors[models.index(model)]
            ax.bar(x[i] - width / 2 + j * width, r2, width, color=color)

    ax.set_xlabel("Prosodic Features", fontsize=16)
    ax.set_ylabel("R2", fontsize=16)
    # ax.set_title("R2 by Model and Feature", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Manually create legend (shared between subplots) in the upper left
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=4, label=model)
        for model, color in zip(models, colors)
    ]
    fig.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.1, 0.95),
        loc="upper left",
        fontsize=12,
    )

    fig.tight_layout()

    plt.show()


def plot_mutual_info(table_data, log_scale=False):
    # Set plot style
    sns.set(style="whitegrid")

    # Prepare the data
    features = list(table_data.keys())
    models = list(table_data[features[0]]["models"].keys())
    num_models = len(models)
    colors = plt.cm.viridis(np.linspace(0, 1, num_models))  # same colors as before

    x = np.arange(len(features))  # label locations
    width = 0.15  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, feature in enumerate(features):
        # Calculate mutual info values and sort models by these values
        mutual_info_values = [
            (
                model,
                table_data[feature]["entropy"]
                - (
                    table_data[feature]["models"].get(model, 0)
                    if table_data[feature]["models"].get(model) is not None
                    else 0
                ),
            )
            for model in models
        ]
        mutual_info_values.sort(key=lambda x: x[1])

        # For each model, plot the mutual info for the feature
        for j, (model, mutual_info) in enumerate(mutual_info_values):
            color = colors[models.index(model)]  # consistent color for each model
            ax.bar(x[i] - width / 2 + j * width, mutual_info, width, color=color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Prosodic Features", fontsize=18, fontweight="bold")
    ax.set_ylabel("Mutual Information", fontsize=18, fontweight="bold")
    ax.set_title(
        "Mutual Information by Model and Feature", fontsize=20, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        features, fontsize=14, rotation=45, ha="right"
    )  # rotated and right-aligned
    ax.tick_params(axis="y", labelsize=14)  # y-axis labels fontsize

    # y-logscale
    if log_scale:
        ax.set_yscale("log")

    # Manually create legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=4, label=model)
        for model, color in zip(models, colors)
    ]
    ax.legend(handles=legend_elements, fontsize=14)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()

    plt.show()


def plot_bars(entropy: float, surprisal: float, title: str, labels=None):
    if labels is None:
        labels = ["Entropy", "Surprisal"]

    # Set plot style
    plt.style.use("seaborn-whitegrid")

    # Set data
    data = [entropy, surprisal]

    # Create x coordinates for the bars
    x = np.arange(len(labels))

    # Specify figure size (width, height) in inches
    plt.figure(figsize=(5, 4))

    # Create the bar chart
    plt.bar(x, data, color="darkblue", width=0.4)

    # Set the x-axis tick labels
    plt.xticks(x, labels, fontsize=16)

    # Set y-axis label
    plt.ylabel("Pearson correlation", fontsize=16)

    # Set the title for the figure and make it large
    plt.title(title, fontsize=20)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_signals_with_mse_pearson(
    original, reconstructed, title=None, xlabel=None, ylabel=None
):
    mse = mean_squared_error(original, reconstructed)
    pearson_corr, _ = pearsonr(original, reconstructed)

    plt.figure(figsize=(10, 5))
    plt.plot(original, label=f"Original Signal)")
    plt.plot(
        reconstructed,
        label=f"Reconstructed Signal (MSE: {mse:.3f}, Pearson: {pearson_corr:.3f})",
    )
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def plot_correlations(feature_labels, spearman_values, pearson_values, title=None):
    # Bar plot settings
    x = np.arange(len(feature_labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(
        x - width / 2, spearman_values, width, label="Spearman's", color="tab:blue"
    )
    rects2 = ax.bar(
        x + width / 2, pearson_values, width, label="Pearson's", color="tab:red"
    )

    # Custoze plot
    ax.set_ylabel("Correlation")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Correlation between Prosodic Prominence and Various Features")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels)
    ax.legend()

    # Add a dotted line at y=0
    ax.axhline(0, color="black", linestyle="dotted")

    # Function to add labels above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{:.3f}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()


def plot_sentence_f0(WAV_PATH, word_alignments, f0_values):
    # Load the audio file
    sr, y = wavfile.read(WAV_PATH)

    # Create a figure with two subplots (one for the audio signal and one for the f0 values)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot the audio signal
    ax1.plot(y)
    ax1.set_title("Audio Signal")

    # Plot the word boundaries
    for alignment in word_alignments:
        start, end, word = alignment
        ax1.axvline(sr * start, linestyle="--", color="red", alpha=0.6)
        ax1.axvline(sr * end, linestyle="--", color="red", alpha=0.6)

    # Replace None values with -1 to avoid plotting them
    f0_values = [-1 if val is None else val for val in f0_values]

    # Calculate word midpoints for f0 values and x-axis labels
    word_midpoints = [(sr * start + sr * end) / 2 for start, end, _ in word_alignments]

    # Plot the f0 values as dots
    ax2.scatter(word_midpoints, f0_values, color="red")

    # Connect the dots with a dotted line
    ax2.plot(word_midpoints, f0_values, linestyle="--", color="red", alpha=0.6)

    # Set the x-axis ticks to be the word midpoints
    ax2.set_xticks(word_midpoints)

    # Set the x-axis tick labels to be the words
    ax2.set_xticklabels(
        [word for _, _, word in word_alignments], rotation=45, ha="right"
    )

    # Set the x-axis label to be "Words"
    ax2.set_xlabel("Words")

    # Set the y-axis label to be "F0 (Hz)"
    ax2.set_ylabel("F0 (Hz)")

    # Set the title of the plot to be "Mean F0 values over words"
    ax2.set_title("Mean F0 values over words")

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_labels_probabilities_entropies(words, labels, probabilities, entropies):
    assert len(words) == len(labels) == len(probabilities) == len(entropies)

    x = np.arange(len(words))

    fig, ax1 = plt.subplots()

    # Plot labels
    ax1.plot(
        x,
        labels,
        marker="o",
        linestyle="-",
        label="Labels",
        color="blue",
        alpha=1,
    )
    ax1.set_ylabel("Labels", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Plot probabilities
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        probabilities,
        marker="o",
        linestyle="--",  # Dashed lines for probabilities
        label="Probabilities",
        color="red",
        alpha=0.8,
    )
    ax2.set_ylabel("Probabilities", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    # Plot entropies
    ax3 = ax1.twinx()
    # Offset the third axis to the right
    ax3.spines["right"].set_position(("outward", 60))
    ax3.plot(
        x,
        entropies,
        marker="o",
        linestyle=":",  # Dotted lines for entropies
        label="Entropies",
        color="green",
        alpha=0.8,
    )
    ax3.set_ylabel("Entropies", color="g")
    ax3.tick_params(axis="y", labelcolor="g")

    ax1.set_xticks(x)
    ax1.set_xticklabels(words, rotation=75, ha="right")
    ax1.set_xlabel("Words")
    ax1.set_title("Labels, Probabilities and Entropies for Each Word")

    # Add a legend for all y-axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_labels_and_probabilities(words, labels, probabilities):
    assert len(words) == len(labels) == len(probabilities)

    x = np.arange(len(words))

    fig, ax1 = plt.subplots()

    # Plot labels
    ax1.plot(
        x,
        labels,
        marker="o",
        linestyle="-",
        label="Labels",
        color="blue",
        alpha=1,
    )
    ax1.set_ylabel("Labels", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Plot probabilities
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        probabilities,
        marker="o",
        linestyle="--",  # Dashed lines for probabilities
        label="Probabilities",
        color="red",
        alpha=0.8,
    )
    ax2.set_ylabel("Probabilities", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax1.set_xticks(x)
    ax1.set_xticklabels(words, rotation=75, ha="right")
    ax1.set_xlabel("Words")
    ax1.set_title("Labels and Probabilities for Each Word")

    # Add a legend for both y-axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_ground_truth_and_predictions(text, labels, preds_list, model_names):
    assert len(text) == len(labels)
    assert all(len(labels) == len(preds) for preds in preds_list)

    x = np.arange(len(text))

    fig, ax = plt.subplots()

    colors = ["blue", "green", "red", "purple", "orange", "cyan", "magenta", "yellow"]

    # Plot ground truth
    ax.plot(
        x,
        labels,
        marker="o",
        linestyle="-",
        label="Ground Truth",
        color="blue",
        alpha=1,  # 100% opacity for ground truth
    )

    # Plot predictions for each model
    for i, (preds, model_name) in enumerate(zip(preds_list, model_names)):
        ax.plot(
            x,
            preds,
            marker="o",
            linestyle="--",  # Dashed lines for the models
            label=f"{model_name} Predictions",
            color=colors[(i + 1) % len(colors)],
            alpha=0.8,  # 50% opacity for the models
        )

    ax.set_xticks(x)
    ax.set_xticklabels(text, rotation=75, ha="right")
    ax.set_xlabel("Words")
    ax.set_ylabel("Values")
    ax.set_title("Ground Truth and Predictions for Each Word")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_mean_mse(
    models,
    model_names,
    mode="sentence_length",
    error_bars=False,
    plot_title=None,
    x_label=None,
    y_label=None,
    min_samples=1,
):
    if not isinstance(models, list):
        models = [models]

    if not isinstance(model_names, list):
        model_names = [model_names]

    assert len(models) == len(
        model_names
    ), "The number of models and model names should be the same."

    width = 0.2
    offsets = np.arange(-width * len(models) / 2, width * len(models) / 2, width)
    colors = ["blue", "green", "red", "purple", "orange", "cyan", "magenta", "yellow"]

    for i, (model, model_name) in enumerate(zip(models, model_names)):
        if mode == "sentence_length":
            mean_std_mse = model.compute_stats_per_sentence_length()
            x_title = "Sentence Length in Words"
        elif mode == "word_position":
            mean_std_mse = model.compute_stats_per_word_position()
            x_title = "Word Position"

        keys = [
            key
            for key in mean_std_mse.keys()
            if mean_std_mse[key]["count"] >= min_samples
        ]
        means = [mean_std_mse[key]["mean"] for key in keys]
        stds = [mean_std_mse[key]["std"] for key in keys]

        if error_bars:
            lower_error = np.clip(stds, 0, means)
            upper_error = stds
            error = [lower_error, upper_error]
        else:
            error = None

        plt.errorbar(
            np.array(keys) + offsets[i],
            means,
            yerr=error,
            fmt="o",
            capsize=5,
            ecolor="red",
            markeredgecolor="black",
            markerfacecolor=colors[i % len(colors)],
            alpha=0.5,
            label=model_name,
        )

    plt.xlabel(x_label if x_label else x_title)
    plt.ylabel(y_label if y_label else "Mean MSE")
    plt.title(plot_title if plot_title else f"Mean MSE per {x_title}")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.show()


def plot_language_model_performance_all_metrics(data, show=True, save_path=None):
    model_names = list(data.keys())
    task_names = list(data[model_names[0]].keys())
    metric_names = list(data[model_names[0]][task_names[0]].keys())

    num_models = len(model_names)
    num_tasks = len(task_names)
    num_metrics = len(metric_names)

    colors = plt.cm.viridis(np.linspace(0, 1, num_models))

    bar_width = 0.1
    index = np.arange(num_models)

    for metric_index, metric_name in enumerate(metric_names):
        plt.figure(figsize=(10, 6))

        for task_index, task_name in enumerate(task_names):
            baseline_values = [
                data[model_names[1]][task_name][metric_name]
                if data[model_names[0]][task_name][metric_name] == "undef"
                else data[model_names[0]][task_name][metric_name]
                for model_name in model_names
            ]

            for model_index, model_name in enumerate(model_names):
                value = (
                    baseline_values[model_index]
                    if data[model_name][task_name][metric_name] == "undef"
                    else data[model_name][task_name][metric_name]
                )
                bars = plt.bar(
                    index[model_index] + (task_index * bar_width),
                    value,
                    width=bar_width,
                    color=colors[model_index],
                    label=model_name
                    if task_index == 0
                    else None,  # prevent duplicate labels
                )

                if model_index > 0:
                    for bar in bars:
                        improvement = (value / baseline_values[0] - 1) * 100
                        plt.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{improvement:.1f}%",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            rotation=45,
                        )

        plt.xlabel("Models")
        plt.ylabel(metric_name)
        plt.title(
            f"Language Model Performance on Token-based Regression Tasks - {metric_name}"
        )
        plt.xticks(
            index + bar_width * (num_tasks / 2) - bar_width / 2,
            model_names,
        )
        plt.legend(loc="upper right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        if show:
            plt.show()

        if save_path is not None:
            plt.savefig(f"{save_path}_{metric_name}", dpi=700)


def plot_language_model_performance(
    data,
    error_names=None,
    show=True,
    save_path=None,
    metric="MSE",
    title="",
    minmetric=False,
    y_lim=None,
):
    """
    ::param data: Dictionary of model names and their corresponding task errors. Format: {model_name: {task_name: error}}
    ::param error_names: List of error names for each task.
    ::param minmetric: True if the goal is to minimize the metric, otherwise False (default).
    """
    if error_names is None:
        error_names = [metric for _ in range(len(list(data.values())[0]))]

    model_names = list(data.keys())
    task_names = list(data[model_names[0]].keys())
    num_models = len(model_names)
    num_tasks = len(task_names)

    # Aesthetic colors
    colors = plt.cm.viridis(np.linspace(0, 1, num_models))

    bar_width = 0.1
    index = np.arange(num_tasks)

    plt.figure(figsize=(10, 6))

    baseline_errors = [
        float(data[model_names[0]][task_name])
        if data[model_names[0]][task_name] != "undef"
        else 0.0
        for task_name in task_names
    ]

    for idx, model_name in enumerate(model_names):
        task_errors = [
            float(data[model_name][task_name])
            if data[model_name][task_name] != "undef"
            else 0.0
            for task_name in task_names
        ]
        bars = plt.bar(
            index + (idx * bar_width),
            task_errors,
            width=bar_width,
            color=colors[idx],
            label=model_name,
        )

        if idx > 0:
            for task_name, bar, base_error, task_error in zip(
                task_names, bars, baseline_errors, task_errors
            ):
                if base_error == 0:
                    base_error = float(data[model_names[1]][task_name])

                if minmetric:
                    improvement = ((base_error - task_error) / base_error) * 100
                else:
                    improvement = ((task_error / base_error) - 1) * 100
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{improvement:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

    # plt.xlabel("Tasks")
    plt.ylabel(metric, fontsize=12)
    plt.title(f"{title}", fontsize=14)
    if y_lim:
        plt.ylim(y_lim)
    plt.xticks(
        index + bar_width * (num_models / 2) - bar_width / 2,
        [f"{task_name}" for idx, task_name in enumerate(task_names)],
    )
    plt.legend(loc="upper right")

    # Add grid
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    if show:
        plt.show


def plot_metrics_over_time(
    train_loss_list, validation_loss_list, loss_name="Loss", show=True, save_path=None
):
    """
    Plots the training and validation losses.
    """
    plt.plot(train_loss_list, label="Training loss")
    plt.plot(validation_loss_list, label="Validation loss")
    plt.title("Training and validation loss")
    plt.ylabel(loss_name)
    plt.xlabel("Epoch/Step")
    plt.legend()

    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path, dpi=700)

    plt.close()


def plot_class_distribution(class_labels, show_relative_freq=False, save_path=None):
    """
    Plots the class distribution of the provided class labels.
    """
    # Count the frequency of each class label
    counts = np.bincount(class_labels)

    if show_relative_freq:
        # Compute relative frequencies
        counts = counts / len(class_labels)
        ylabel = "Relative frequency"
    else:
        ylabel = "Absolute frequency"

    # Create a bar plot of the class distribution
    plt.bar(range(len(counts)), counts)

    # Set the x-ticks to only include class labels that appear in the data
    unique_labels = np.unique(class_labels)
    plt.xticks(unique_labels, unique_labels)

    # Add labels and title
    plt.xlabel("Class label")
    plt.ylabel(ylabel)
    plt.title("Class distribution")

    plt.show()

    # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path)

    plt.close()


def plot_features_over_sentence(
    words: List[str],
    durations: List[float],
    prominences: List[float],
    energies: List[float],
    save_path=None,
) -> None:
    # Calculate the x-coordinates of the words based on their durations
    x_coords = [sum(durations[:i]) for i in range(len(words) + 1)]

    # Set up the plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # create a second y-axis on the right-hand side of the plot

    for i, word in enumerate(words):
        ax1.plot(x_coords[i] + durations[i] / 2, prominences[i], ".", color="blue")
        ax2.plot(x_coords[i] + durations[i] / 2, energies[i], "o", color="red")
        if i > 0:
            ax1.plot(
                [
                    x_coords[i - 1] + durations[i - 1] / 2,
                    x_coords[i] + durations[i] / 2,
                ],
                [prominences[i - 1], prominences[i]],
                ":",
                color="blue",
            )
            ax2.plot(
                [
                    x_coords[i - 1] + durations[i - 1] / 2,
                    x_coords[i] + durations[i] / 2,
                ],
                [energies[i - 1], energies[i]],
                ":",
                color="red",
            )

    # Set the x-ticks and labels
    ax1.set_xticks([sum(durations[:i]) + durations[i] / 2 for i in range(len(words))])
    ax1.set_xticklabels(words, rotation=75, ha="right")

    # Set the y-labels
    ax1.set_ylabel("Prominence")
    ax2.set_ylabel("Energy")

    # Set the limits of the y-axes
    ax1.set_ylim(0, max(prominences) + 1)
    ax2.set_ylim(0, 1)

    # Add legend
    ax1.legend(["Prominence"], loc="upper left")
    ax2.legend(["Energy"], loc="upper right")

    # Show the plot
    plt.show()

    if save_path is not None:
        plt.savefig(save_path)

    plt.close()


def plot_continuous_histogram(
    labels,
    label_name="Value",
    title="Histogram",
    num_bins=10,
    relative_freq=True,
    save_path=None,
):
    # Convert the labels to an array and get the range of the data
    data = np.array(labels)
    data_range = data.max() - data.min()

    # Calculate the histogram and bin edges
    hist, bin_edges = np.histogram(data, bins=num_bins)

    # Calculate the relative frequency if requested
    if relative_freq:
        hist = hist / len(labels)

    # Plot the histogram as a bar chart
    plt.bar(bin_edges[:-1], hist, width=data_range / num_bins, align="edge")

    # Set the x-axis limits and labels
    plt.xlim(data.min(), data.max())
    plt.xlabel(label_name)

    # Set the y-axis limits and label
    if relative_freq:
        plt.ylim(0, 1)
        plt.ylabel("Relative Frequency")
    else:
        plt.ylabel("Frequency")

    plt.title(title)

    # Show the plot
    plt.show()

    if save_path is not None:
        plt.savefig(save_path)

    plt.close()


def plot_r2_ordered(table_data):
    # Prepare the data
    features = list(table_data.keys())
    models = list(table_data[features[0]].keys())
    num_models = len(models)
    colors = plt.cm.viridis(np.linspace(0, 1, num_models))

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(features))  # label locations
    width = 0.15  # width of the bars

    for i, feature in enumerate(features):
        # Get R2 values and sort models by these values
        r2_values = [
            (
                model,
                table_data[feature].get(model, 0)
                if table_data[feature].get(model) is not None
                else 0,
            )
            for model in models
        ]
        r2_values.sort(key=lambda x: x[1])

        # For each model, plot the R2 for the feature
        for j, (model, r2) in enumerate(r2_values):
            color = colors[models.index(model)]  # consistent color for each model
            ax.bar(x[i] - width / 2 + j * width, r2, width, color=color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Prosodic Features", fontsize=14)
    ax.set_ylabel("R2", fontsize=14)
    ax.set_title("R2 by Model and Feature", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45)

    # Manually create legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=4, label=model)
        for model, color in zip(models, colors)
    ]
    ax.legend(handles=legend_elements)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()

    plt.show()


def plot_vector_kde(
    data: np.array,
    bw_scalar=0.1,
    label_names=None,
    title="Kernel Density Estimation",
    save_path=None,
):
    # Set plot style
    sns.set(style="whitegrid")

    # Check if label_names is given
    if label_names is None:
        label_names = [f"Value {i+1}" for i in range(data.shape[1])]

    # Check if the number of label names match the number of coefficients
    assert (
        len(label_names) == data.shape[1]
    ), "Number of label names must match the number of coefficients"

    # Plot the KDE for each coefficient
    for i in range(data.shape[1]):
        data_i = data[:, i]

        # Plot the KDE using seaborn
        sns.kdeplot(data_i, fill=True, label=label_names[i], bw=bw_scalar)

        # Calculate mean and standard deviation
        mean = np.mean(data_i)
        std = np.std(data_i)

        # Add mean and standard deviation to the legend
        # plt.text(
        #     0.02,
        #     0.92 - 0.04 * i,
        #     f"Mean of {label_names[i]}: {mean:.2f}",
        #     transform=plt.gca().transAxes,
        # )
        # plt.text(
        #     0.02,
        #     0.88 - 0.04 * i,
        #     f"Std Dev of {label_names[i]}: {std:.2f}",
        #     transform=plt.gca().transAxes,
        # )

    # Set labels and title
    plt.xlabel("Values", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)

    if save_path is not None:
        plt.savefig(save_path, dpi=1000, bbox_inches="tight")

    # Show the plot
    plt.show()
    plt.close()


def plot_kde(
    data: np.array,
    bw_adjust,
    label_name="Value",
    title="Kernel Density Estimation",
    y_log=False,
    save_path=None,
    x_range=None,
    y_range=None,
):
    # Set plot style
    sns.set(style="whitegrid")

    # Plot the KDE using seaborn
    sns.kdeplot(data, fill=True, bw_adjust=bw_adjust)

    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)

    # Plot mean and standard deviation as dotted lines
    plt.axvline(
        x=mean, linestyle=":", color="pink", label=f"Mean: {mean:.2f}", linewidth=2
    )
    plt.axvline(
        x=mean - std,
        linestyle=":",
        color="plum",
        label=f"Std Dev: {std:.2f}",
        linewidth=2,
    )
    plt.axvline(x=mean + std, linestyle=":", color="plum", linewidth=2)

    # Set labels and title
    plt.xlabel(label_name, fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)

    if x_range is not None:
        plt.xlim(x_range[0], x_range[1])

    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])

    if save_path is not None:
        plt.savefig(save_path, dpi=1000, bbox_inches="tight")

    # Show the plot
    plt.show()
    plt.close()


def plot_continuous_kde(
    labels_dict,
    label_name="Value",
    title="Kernel Density Estimation",
    save_path=None,
):
    # Convert the labels to a list of arrays
    try:
        data = [np.array(labels_dict[label]) for label in labels_dict]
    except:
        data = labels_dict  # was already in format TODO: this is unsafe

    # Plot the KDE using seaborn
    for i in range(len(data)):
        sns.kdeplot(data[i], fill=True, label=list(labels_dict.keys())[i])

    # Set labels and title
    plt.xlabel(label_name)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()

    # Show the plot
    plt.show()

    if save_path is not None:
        plt.savefig(save_path)

    plt.close()
