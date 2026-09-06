import pandas as pd
import matplotlib.pyplot as plt

def load_markdown_table(md_path):
    """
    Load a GitHub-style markdown table into a Pandas DataFrame.
    """
    with open(md_path, "r") as f:
        lines = f.readlines()

    # Filter only table lines (those containing '|')
    table_lines = [line.strip() for line in lines if "|" in line]

    # Remove separator line (|---|)
    table_lines = [line for line in table_lines if "---" not in line]

    # Split each row into columns
    rows = [line.split("|")[1:-1] for line in table_lines]  # remove outer pipes

    # First row is header
    header = [h.strip() for h in rows[0]]
    data = [[cell.strip() for cell in row] for row in rows[1:]]

    df = pd.DataFrame(data, columns=header)

    # Convert numeric columns
    numeric_cols = [
        "n", "success", "within_tol", "exact_match", "fallback",
        "mean_hour_err", "mean_minute_err", "mean_second_err"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_mean_errors(df, output_path="results/stats_summary_plot.png"):
    """
    Create a bar chart of mean errors for each condition.
    """
    plt.figure(figsize=(12, 6))

    # Plot mean errors
    plt.plot(df["condition_id"], df["mean_hour_err"], marker="o", label="Hour Error")
    plt.plot(df["condition_id"], df["mean_minute_err"], marker="o", label="Minute Error")
    plt.plot(df["condition_id"], df["mean_second_err"], marker="o", label="Second Error")

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Error (degrees)")
    plt.title("Symbolic Clock Reading Errors Under Perturbation Conditions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved graph to {output_path}")


if __name__ == "__main__":
    md_file = "results/stats_summary.md"  # path to your MD file
    df = load_markdown_table(md_file)
    plot_mean_errors(df)
