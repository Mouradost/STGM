import torch


def MSE(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: tuple = (0, 1, 2, 3),
    null_val: float = torch.nan,
) -> torch.Tensor:
    if null_val is torch.nan or null_val is None:
        return ((target - pred) ** 2).nanmean(dim=dim)
    return torch.where(
        target != null_val,
        (target - pred) ** 2,
        torch.zeros_like(pred),
    ).mean(dim=dim)


def RMSE(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: tuple = (0, 1, 2, 3),
    null_val: float = torch.nan,
) -> torch.Tensor:
    return MSE(pred=pred, target=target, dim=dim, null_val=null_val).sqrt()


def MAE(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: tuple = (0, 1, 2, 3),
    null_val: float = torch.nan,
) -> torch.Tensor:
    if null_val is torch.nan or null_val is None:
        return ((target - pred).abs()).nanmean(dim=dim)
    return torch.where(
        target != null_val,
        (target - pred).abs(),
        torch.zeros_like(pred),
    ).mean(dim=dim)


def MAPE(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: tuple = (0, 1, 2, 3),
    percents: bool = False,
    null_val: float = torch.nan,
) -> torch.Tensor:
    if null_val is torch.nan or null_val is None:
        return ((target - pred).abs() / target.abs()).nanmean(dim=dim)
    return torch.where(
        target != null_val,
        (target - pred).abs() / target.abs(),
        torch.zeros_like(pred),
    ).mean(dim=dim) * (100 if percents else 1)


def calculate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: tuple = (0, 1, 2),
    null_val: float = torch.nan,
    metrics_fn: tuple = (MSE, RMSE, MAE, MAPE),
    percents=True,
):
    metrics = {}
    for metric_fn in metrics_fn:
        if metric_fn.__name__ == "MAPE":
            metrics[metric_fn.__name__] = metric_fn(
                pred=pred,
                target=target,
                dim=dim,
                null_val=null_val,
                percents=percents,
            ).item()
        else:
            metrics[metric_fn.__name__] = metric_fn(
                pred=pred, target=target, dim=dim, null_val=null_val
            ).item()
    return metrics


def summary(
    pred: torch.Tensor,
    target: torch.Tensor,
    suffix: str = "",
    line_size: int = 80,
    precision: int = 2,
    dim: tuple = (0, 2, 3),
    null_val: float = torch.nan,
    fn_metrics: tuple = (MSE, RMSE, MAE, MAPE),
) -> tuple:
    """Computes the given metrics.
    Args:
        preds: Estimated labels. Second dimension is the number of ensemble members. [B, T, N, C]
        target: Ground truth labels. [B, T, N, C]
        suffix: Additional suffix title
        line_size: The maximum width of the displayed string
        precision: Number of floating points
        dim: All dimension that doesn't include the time-step dimension.
        fn_metric: All the desired metric function to use. Must follow a pattern
        `fn(pred: torch.Tensor, target: torch.Tensor, dim: tuple[int]) -> torch.Tensor`
    Return:
        Tuple[str, Tensor] The display string and values of the metrics
    """
    percents_idx = []
    # Calculating the metrics along the time dimension
    metrics = [
        fn_metric(pred=pred, target=target, dim=dim, null_val=null_val)
        for fn_metric in fn_metrics
    ]

    # Calculating the spacing
    spacing = line_size // len(metrics)
    step_spacing = line_size // (len(metrics) + 1)

    # Mean of all errors
    results = "\n"
    results += "=" * line_size + "\n"
    results += f"{f'Mean Error {suffix}':^{line_size}}" + "\n"
    results += "-" * line_size + "\n"
    for i, fn_metric in enumerate(fn_metrics):
        if fn_metric.__name__ == "MAPE":
            percents_idx.append(i)
        results += f"{fn_metric.__name__:^{spacing}}"
    results += "\n"
    results += "-" * line_size + "\n"
    for i, metric in enumerate(metrics):
        if i in percents_idx:
            results += f"{f'{metric.mean().item():.{precision}%}':^{spacing}}"
        else:
            results += f"{f'{metric.mean().item():.{precision}f}':^{spacing}}"
    results += "\n"
    results += "=" * line_size + "\n"

    # By time step
    results += "=" * line_size + "\n"
    results += f"{f'By Step Error {suffix}':^{line_size}}" + "\n"
    results += "-" * line_size + "\n"
    results += f"{'Steps':^{step_spacing}}"
    for fn_metric in fn_metrics:
        results += f"{fn_metric.__name__:^{step_spacing}}"
    results += "\n"
    results += "-" * line_size + "\n"
    for i in range(metrics[0].shape[0]):
        results += f"{i+1:^{step_spacing}}"
        for j, metric in enumerate(metrics):
            if j in percents_idx:
                results += f"{f'{metric[i].item():.{precision}%}':^{step_spacing}}"
            else:
                results += f"{f'{metric[i].item():.{precision}f}':^{step_spacing}}"
        results += "\n"
    results += "=" * line_size + "\n"
    return results, metrics


if __name__ == "__main__":
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser(
        prog="Traffic Metrics",
        description="Metrics for traffic forecasting",
        epilog="By Dr. Mourad Lablack",
    )
    parser.add_argument("path", type=str, help="Path to predictions data")
    args = parser.parse_args()
    path = Path(args.path)
    print(summary(*torch.load(path), dim=(0, 2, 3), suffix="Normalized")[0])
