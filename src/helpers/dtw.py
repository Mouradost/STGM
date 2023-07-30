#!/usr/bin/env python

# Build-in
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool

# Logging
from tqdm import tqdm

# ML
import numpy as np
import pandas as pd


def adjust_length(longer_array: np.ndarray, shorter_array: np.ndarray) -> np.ndarray:
    """Pad the smaller series in order to have the same length as the longer array
    :param longer_array: the longer of the two sequences
    :param shorter_array: the shorter of the two sequences
    :return: the padded shorter sequences, which now has the same length as the longer array
    """

    difference_in_length = len(longer_array) - len(shorter_array)
    padding_zeros = np.zeros(difference_in_length)
    adjusted_array = np.concatenate([shorter_array, padding_zeros])
    return adjusted_array


def dtwupd(a: np.ndarray, b: np.ndarray, r: int):
    """Compute the DTW distance between 2 time series with a warping band constraint
    :param a: the time series array 1
    :param b: the time series array 2
    :param r: the size of Sakoe-Chiba warping band
    :return: the DTW distance
    """

    if len(a) < len(b):
        a = adjust_length(longer_array=b, shorter_array=a)
    elif len(a) > len(b):
        b = adjust_length(longer_array=a, shorter_array=b)

    m = len(a)
    k = 0

    # Instead of using matrix of size O(m^2) or O(mr), we will reuse two arrays of size O(r)
    cost = [float("inf")] * (2 * r + 1)
    cost_prev = [float("inf")] * (2 * r + 1)

    for i in range(0, m):
        k = max(0, r - i)

        for j in range(max(0, i - r), min(m - 1, i + r) + 1):
            # Initialize all row and column
            if i == 0 and j == 0:
                c = a[0] - b[0]
                cost[k] = c * c

                k += 1
                continue

            y = float("inf") if j - 1 < 0 or k - 1 < 0 else cost[k - 1]
            x = float("inf") if i < 1 or k > 2 * r - 1 else cost_prev[k + 1]
            z = float("inf") if i < 1 or j < 1 else cost_prev[k]

            # Classic DTW calculation
            d = a[i] - b[j]
            cost[k] = min(x, y, z) + d * d

            k += 1

        # Move current array to previous array
        cost, cost_prev = cost_prev, cost

    # The DTW distance is in the last cell in the matrix of size O(m^2) or at the middle of our array
    k -= 1
    return cost_prev[k]


def cal_dtw(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = []
    for yi in y:
        result.append(dtwupd(x, yi, 4))
    return np.array(result)


def parallel_dtw(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dtw = partial(cal_dtw, y=y)

    with Pool() as pool:
        results = pool.imap(dtw, x)
        results = np.array(list(results))
    return results


def sequential_dtw(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    results = np.zeros([x.shape[0], x.shape[0]])
    loop = tqdm(
        enumerate(x),
        desc="DTW",
        leave=False,
        total=x.shape[0],
    )

    for i, xi in loop:
        for j, yj in enumerate(y):
            results[i, j] = dtwupd(xi, yj, 4)
    return results


def main(
    path,
    start: int = 0,
    end: int = None,
    node_start: int = 0,
    node_end: int = None,
    name: str = "dtw",
    use_parrallel: bool = True,
):
    if not isinstance(path, Path):
        path = Path(path)
    data = pd.read_hdf(path).T.values
    if use_parrallel:
        fn = parallel_dtw
    else:
        fn = sequential_dtw

    dtw = fn(data[node_start:node_end, start:end], data[node_start:node_end, start:end])
    np.save(Path.cwd() / f"{name}.npy", dtw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="DTW",
        description="Create the similarity matrix based on the DTW algo",
        epilog="By Dr. Mourad Lablack",
    )
    parser.add_argument("path", type=str, help="Path to signal data")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="dtw",
        help="The saved DTW numpy file name. [str] (default: dtw) ",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="The starting index. [int] (default: 0)",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=-1,
        help="The end index. [int] (default: -1)",
    )
    parser.add_argument(
        "-S",
        "--node_start",
        type=int,
        default=0,
        help="The starting node index. [int] (default: 0)",
    )
    parser.add_argument(
        "-E",
        "--node_end",
        type=int,
        default=-1,
        help="The end node index. [int] (default: -1)",
    )
    parser.add_argument(
        "-p",
        "--parrallel",
        action="store_true",
        help="Use parallel DTW algo.",
    )
    args = parser.parse_args()
    main(
        path=Path(args.path),
        start=args.start,
        end=args.end,
        node_start=args.node_start,
        node_end=args.node_end,
        name=args.name,
        use_parrallel=args.parrallel,
    )
