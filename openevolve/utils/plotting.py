"""
Utility module for generating evolution plots
"""

import glob
import logging
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def parse_log_file(log_file: str) -> Dict[str, List[float]]:
    """
    Parse the log file to extract iteration, best score, avg score, and diversity.
    """
    data = {
        "iterations": [],
        "best_scores": [],
        "avg_scores": [],
        "diversities": [],
    }

    current_iter = 0

    try:
        with open(log_file, "r") as f:
            for line in f:
                # Match iteration start/end to keep track
                # "Iteration 14: Program ..."
                iter_match = re.search(r"Iteration (\d+):", line)
                if iter_match:
                    current_iter = int(iter_match.group(1))

                # Match Island Status for aggregate metrics
                # "Island 0: 13 programs, best=0.9162, avg=0.9136, diversity=44.57"
                # We prioritize this for plotting aggregate progress
                island_match = re.search(
                    r"Island \d+: .* best=([\d\.]+), avg=([\d\.]+), diversity=([\d\.]+)",
                    line,
                )
                if island_match:
                    # We might have multiple islands, but let's just take the first one found per reporting cycle
                    # or strictly Island 0 for simplicity in this single-island run context
                    if "Island 0" in line:
                        data["iterations"].append(current_iter)
                        data["best_scores"].append(float(island_match.group(1)))
                        data["avg_scores"].append(float(island_match.group(2)))
                        data["diversities"].append(float(island_match.group(3)))
    except Exception as e:
        logger.error(f"Error parsing log file {log_file}: {e}")

    return data


def plot_evolution(log_dir: str, output_dir: str) -> None:
    """
    Generate evolution plots from the latest log file in log_dir and save to output_dir.
    """
    # Find the latest log file
    log_files = glob.glob(os.path.join(log_dir, "openevolve_*.log"))
    if not log_files:
        logger.warning(f"No log files found in {log_dir}, skipping plot generation.")
        return

    latest_log = max(log_files, key=os.path.getmtime)
    logger.info(f"Generating plots from log: {latest_log}")

    data = parse_log_file(latest_log)

    if not data["iterations"]:
        logger.warning("No adequate data found in log file for plotting.")
        return

    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        # Plot Best vs Avg Score
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(data["iterations"], data["best_scores"], "b-", label="Best Score")
        plt.plot(data["iterations"], data["avg_scores"], "g--", label="Avg Score")
        plt.title("Fitness Evolution")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)

        # Plot Diversity
        plt.subplot(1, 2, 2)
        plt.plot(data["iterations"], data["diversities"], "r-", label="Diversity")
        plt.title("Population Diversity")
        plt.xlabel("Iteration")
        plt.ylabel("Diversity Metric")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        output_file = os.path.join(plots_dir, "evolution_plots.png")
        plt.savefig(output_file)
        logger.info(f"Evolution plots saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
    finally:
        plt.close("all")
