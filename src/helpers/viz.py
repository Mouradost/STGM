# Build-in
import logging
from pathlib import Path

# ML
import torch
import wandb

# Plotting
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates


def init_plot(style: str = "paper"):
    if style.lower() != "default":
        plt.style.use(style.lower())


def display_metrics(metrics: dict, precision: int = 2, line_size: int = 80):
    spacing = len(metrics) // line_size
    str_display = ""
    for name, metric in metrics.items():
        str_display += f"{f'{name}: {metric:.{precision}f} ':^{spacing}}"
    logging.info(str_display)


def plot(data_timestamps, data, labels):
    fig = plt.figure(figsize=(8, 2), tight_layout=True)
    ax = fig.add_subplot()
    for item, label in zip(data, labels):
        ax.plot_date(data_timestamps, item, fmt="-", label=label)
    ax.xaxis.set_major_formatter(mpl_dates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_formatter(
        mpl_dates.ConciseDateFormatter(
            ax.xaxis.get_major_locator(),
            formats=["%Y", "%b", "%A %d", "%A %H:%M", "%A %H:%M", "%A %S.%f"],
        )
    )
    ax.tick_params(axis="x", which="major", rotation=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=3)
    return fig


def plot_history(history):
    fig = plt.figure(figsize=(14, 4), tight_layout=True)
    show_clip = len(history.get("train/clip", [])) > 0
    # Training overview
    ax = fig.add_subplot(1, 2 if show_clip else 1, 1)
    ax.plot(history["train/loss"], label="train/loss")
    if history["val/loss"][-1] != 0:
        ax.plot(history["val/loss"], label="val/loss")
    ax.legend()
    # Gradiant clipping overview
    if show_clip:
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(history["train/clip"], label="train/clip")
        ax.legend()
    return fig


def plot_predictions(
    predictions,
    targets,
    timestamps=None,
    index=0,
    node=0,
    return_fig=False,
    save_path=None,
    transparent=False,
):
    assert (
        index <= targets.shape[1] and index >= 0
    ), f"The index is out of bound [0, {target.shape[1] + 1}]"
    fig = plt.figure(figsize=(8, 2), tight_layout=True)
    ax = fig.add_subplot()
    ax.plot(
        torch.arange(targets.shape[0]) if timestamps is None else timestamps,
        targets[:, index, node, 0],
        label="Ground Truth",
    )
    ax.plot(
        torch.arange(targets.shape[0]) if timestamps is None else timestamps,
        predictions[:, index, node, 0],
        label="Prediction",
    )
    if timestamps is not None:
        ax.xaxis.set_major_formatter(
            mpl_dates.ConciseDateFormatter(
                ax.xaxis.get_major_locator(),
                formats=["%Y", "%b", "%A %d", "%A %H:%M", "%A %H:%M", "%A %S.%f"],
            )
        )

    ax.set_xlabel("Timestamps")
    ax.set_ylabel("Speed (Km/h)")
    # If the title is important shrinking the plot so that the legend appears above the title
    # ax.set_title("Ground Truth vs Prediction")
    # pos = ax.get_position()
    # ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=3,
    )
    if return_fig:
        return fig
    if save_path is not None:
        fig.savefig(save_path, transparent=transparent)
    plt.show()


def log_fig(
    logger_name: str,
    fig,
    fig_title: str = "",
    category: str = "figures/chart",
    transparent: bool = True,
    extention: str = "pdf",
    path: Path = Path.cwd(),
):
    if logger_name == "wandb":
        wandb.log({category: fig})
    elif logger_name == "tensorboard":
        pass
    else:
        fig.savefig(path / f"{fig_title}.{extention}", transparent=transparent)
    plt.close(fig)
