# process_ellipses
Predicted vs Observed Vegetation Degradation Accuracy by Region
import math
import os
from pathlib import Path

# Ensure a writable Matplotlib config/cache directory before importing pyplot
mpl_dir = Path("./outputs/mpl-cache")
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy as transform_xy
from rasterio.warp import Resampling, reproject
from shapely.affinity import rotate, scale
from shapely.geometry import Point


def raster_to_points(raster_path: Path) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        band = src.read(1, masked=True)
        mask = np.ma.getmaskarray(band)
        rows, cols = np.where(~mask)
        if rows.size == 0:
            raise ValueError(f"No valid data points found in {raster_path}")
        values = band[rows, cols].astype(int)
        xs, ys = transform_xy(src.transform, rows, cols, offset="center")
        df = pd.DataFrame({"region": values, "x": xs, "y": ys})
        df["region"] = df["region"].astype(int)
        df["crs"] = str(src.crs) if src.crs else None
        return df


def compute_sde(points: np.ndarray):
    if points.shape[0] == 1:
        mean_center = points[0]
        covariance = np.zeros((2, 2))
    else:
        mean_center = points.mean(axis=0)
        centered = points - mean_center
        covariance = np.cov(centered, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    major_vector = eigenvectors[:, 0]
    orientation_rad = math.atan2(major_vector[1], major_vector[0])
    orientation_deg = math.degrees(orientation_rad)
    semimajor = math.sqrt(eigenvalues[0]) if eigenvalues[0] > 0 else 0.0
    semiminor = math.sqrt(eigenvalues[1]) if eigenvalues[1] > 0 else 0.0
    circle = Point(mean_center).buffer(1, resolution=256)
    ellipse = scale(circle, semimajor, semiminor)
    ellipse = rotate(
        ellipse,
        orientation_deg,
        origin=(float(mean_center[0]), float(mean_center[1])),
        use_radians=False,
    )
    return {
        "mean_x": float(mean_center[0]),
        "mean_y": float(mean_center[1]),
        "semimajor": float(semimajor),
        "semiminor": float(semiminor),
        "orientation_deg": float(orientation_deg),
        "area": float(ellipse.area),
        "geometry": ellipse,
    }


def process_raster(path: Path, label: str) -> gpd.GeoDataFrame:
    df = raster_to_points(path)
    results = []
    for region_id, group in df.groupby("region"):
        coords = group[["x", "y"]].to_numpy()
        sde = compute_sde(coords)
        sde.update({
            "dataset": label,
            "region": int(region_id),
            "n_points": int(len(group)),
        })
        results.append(sde)
    crs_value = df["crs"].dropna().iloc[0] if df["crs"].notnull().any() else None
    gdf = gpd.GeoDataFrame(results, geometry="geometry", crs=crs_value)
    return gdf


def compute_confusion_metrics(predicted_path: Path, observed_path: Path, boundary_path: Path) -> pd.DataFrame:
    """Compute per-region confusion statistics using all raster data, grouped by region."""

    with rasterio.open(predicted_path) as pred_src, rasterio.open(
        observed_path
    ) as obs_src:
        pred = pred_src.read(1, masked=True)
        obs = obs_src.read(1, masked=True)

        boundary_values = None
        if boundary_path is not None and Path(boundary_path).exists():
            with rasterio.open(boundary_path) as boundary_src:
                boundary_projected = np.zeros(pred.shape, dtype=np.float32)

                reproject(
                    source=rasterio.band(boundary_src, 1),
                    destination=boundary_projected,
                    src_transform=boundary_src.transform,
                    src_crs=boundary_src.crs,
                    dst_transform=pred_src.transform,
                    dst_crs=pred_src.crs,
                    resampling=Resampling.nearest,
                    src_nodata=boundary_src.nodata,
                    dst_nodata=0,
                )

            boundary_values = np.rint(boundary_projected).astype(int)

    consider_mask = (~pred.mask) | (~obs.mask)
    pred_values = pred.filled(0).astype(int)
    obs_values = obs.filled(0).astype(int)
    pred_positive = pred_values > 0
    obs_positive = obs_values > 0

    rows = []

    for region in range(1, 8):
        if boundary_values is not None:
            region_mask = consider_mask & (boundary_values == region)
        else:
            region_mask = consider_mask & (
                (pred_values == region) | (obs_values == region)
            )

        if not np.any(region_mask):
            tp = fp = fn = tn = 0
        else:
            tp = int(np.sum(pred_positive & obs_positive & region_mask))
            fp = int(np.sum(pred_positive & (~obs_positive) & region_mask))
            fn = int(np.sum((~pred_positive) & obs_positive & region_mask))
            tn = int(np.sum((~pred_positive) & (~obs_positive) & region_mask))

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        if np.isnan(precision) or np.isnan(recall):
            f1_score = np.nan
        elif (precision + recall) == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        fom = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan

        rows.append(
            {
                "Region ID": region,
                "Region": f"Region {region}",
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score,
                "FoM": fom,
            }
        )

    overall_mask = consider_mask
    tp = int(np.sum(pred_positive & obs_positive & overall_mask))
    fp = int(np.sum(pred_positive & (~obs_positive) & overall_mask))
    fn = int(np.sum((~pred_positive) & obs_positive & overall_mask))
    tn = int(np.sum((~pred_positive) & (~obs_positive) & overall_mask))
    if tp + fp + fn == 0:
        precision = recall = f1_score = fom = np.nan
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        if np.isnan(precision) or np.isnan(recall):
            f1_score = np.nan
        elif (precision + recall) == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        fom = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan

    rows.append(
        {
            "Region ID": None,
            "Region": "Overall",
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "FoM": fom,
        }
    )

    df = pd.DataFrame(rows)
    return df


def main():
    base_dir = Path("/Users/sherry/Downloads/Geo数据")
    output_dir = Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    raster_info = {
        "Predicted": base_dir / "vege_degradation_predict2.tif",
        "Observed": base_dir / "vege_degradation_real2.tif",
    }
    boundary_path = base_dir / "ditu.tif"

    geodataframes = [process_raster(path, label) for label, path in raster_info.items()]
    combined = pd.concat(geodataframes, ignore_index=True)

    parameters = combined.drop(columns="geometry").copy()
    parameters = parameters[
        [
            "dataset",
            "region",
            "n_points",
            "mean_x",
            "mean_y",
            "semimajor",
            "semiminor",
            "orientation_deg",
            "area",
        ]
    ]
    metrics_df = compute_confusion_metrics(
        raster_info["Predicted"], raster_info["Observed"], boundary_path
    )

    analysis_tag = raster_info["Predicted"].stem
    if "vege_degradation_" in analysis_tag:
        analysis_tag = analysis_tag.replace("vege_degradation_", "")

    excel_path = output_dir / "ellipse_parameters.xlsx"
    try:
        from openpyxl import load_workbook
    except ImportError:
        load_workbook = None

    existing_sheets = set()
    workbook = None
    if excel_path.exists() and load_workbook is not None:
        workbook = load_workbook(excel_path)
        existing_sheets = set(workbook.sheetnames)

    def unique_sheet_name(base: str) -> str:
        name = base
        counter = 2
        while name in existing_sheets:
            name = f"{base}_{counter}"
            counter += 1
        existing_sheets.add(name)
        return name

    ellipses_sheet = unique_sheet_name(f"Ellipses_{analysis_tag}")
    metrics_sheet = unique_sheet_name(f"Confusion_{analysis_tag}")

    if workbook is None:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            parameters.to_excel(writer, sheet_name=ellipses_sheet, index=False)
            metrics_df.to_excel(writer, sheet_name=metrics_sheet, index=False)
    else:
        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            parameters.to_excel(writer, sheet_name=ellipses_sheet, index=False)
            metrics_df.to_excel(writer, sheet_name=metrics_sheet, index=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    dataset_order = ["Observed", "Predicted"]
    dataset_styles = {
        "Observed": {
            "fill_alpha": 0.35,
            "line_style": "-",
            "marker": "o",
            "color": "#264653",
        },
        "Predicted": {
            "fill_alpha": 0.0,
            "line_style": "--",
            "marker": "x",
            "color": "#E76F51",
        },
    }

    for dataset in dataset_order:
        gdf = next(g for g in geodataframes if g["dataset"].iloc[0] == dataset)
        style = dataset_styles[dataset]
        for _, row in gdf.iterrows():
            x, y = row.geometry.exterior.xy
            color = style["color"]
            if style["fill_alpha"] > 0:
                ax.fill(x, y, alpha=style["fill_alpha"], color=color, zorder=2)
            ax.plot(
                x,
                y,
                color=color,
                linewidth=1.8,
                linestyle=style["line_style"],
                zorder=3,
            )
            ax.plot(
                row.mean_x,
                row.mean_y,
                marker=style["marker"],
                color=color,
                markersize=5,
                markeredgewidth=1.2,
                zorder=4,
                linestyle="None",
            )

    handles = [
        plt.Line2D([0], [0], color=dataset_styles["Observed"]["color"], linewidth=1.8, linestyle="-", label="Observed Ellipse"),
        plt.Line2D([0], [0], color=dataset_styles["Predicted"]["color"], linewidth=1.8, linestyle="--", label="Predicted Ellipse"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=dataset_styles["Observed"]["color"], label="Observed Mean Center", markersize=6),
        plt.Line2D([0], [0], marker="x", color=dataset_styles["Predicted"]["color"], linestyle="None", label="Predicted Mean Center", markersize=6),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10)

    png_path = output_dir / f"vegetation_degradation_ellipses_{analysis_tag}.png"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", transparent=True)
    plt.close(fig)

    metrics_regions = metrics_df[metrics_df["Region"] != "Overall"].copy()
    metric_columns = [
        "Precision",
        "Recall",
        "F1 Score",
        "FoM",
    ]
    metric_colors = [
        "#264653",
        "#2A9D8F",
        "#F4A261",
        "#E76F51",
    ]

    fig_metrics, (ax_scores, ax_counts) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    x = np.arange(len(metrics_regions))
    bar_width = 0.18
    offsets = (np.arange(len(metric_columns)) - (len(metric_columns) - 1) / 2) * bar_width

    for idx, metric in enumerate(metric_columns):
        ax_scores.bar(
            x + offsets[idx],
            metrics_regions[metric],
            bar_width,
            label=metric,
            color=metric_colors[idx],
            alpha=0.88,
        )

    ax_scores.set_ylabel("Score", fontweight="bold")
    ax_scores.set_ylim(0, 1.05)
    ax_scores.set_yticks(np.linspace(0, 1, 6))
    ax_scores.spines["top"].set_visible(False)
    ax_scores.spines["right"].set_visible(False)
    ax_scores.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_scores.legend(loc="upper right", frameon=False, ncol=3, fontsize=9)

    count_columns = ["TP", "FP", "FN"]
    count_colors = ["#2A9D8F", "#E76F51", "#F4A261"]
    bottom = np.zeros(len(metrics_regions), dtype=float)
    for idx, column in enumerate(count_columns):
        values = metrics_regions[column].to_numpy()
        ax_counts.bar(
            x,
            values,
            width=0.6,
            bottom=bottom,
            color=count_colors[idx],
            edgecolor="#ffffff",
            linewidth=0.6,
            label=column,
            alpha=0.9,
        )
        bottom += values

    ax_counts.set_ylabel("Cell Count", fontweight="bold")
    ax_counts.set_xlabel("Region", fontweight="bold")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(metrics_regions["Region"], rotation=0)
    ax_counts.spines["top"].set_visible(False)
    ax_counts.spines["right"].set_visible(False)
    ax_counts.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_counts.legend(loc="upper right", frameon=False, ncol=4, fontsize=9)

    fig_metrics.suptitle(
        "Predicted vs Observed Vegetation Degradation Accuracy by Region",
        fontsize=14,
        fontweight="bold",
    )
    fig_metrics.tight_layout()
    fig_metrics.subplots_adjust(top=0.92)

    metrics_png = output_dir / f"region_confusion_metrics_{analysis_tag}.png"
    fig_metrics.savefig(metrics_png, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig_metrics)

    confusion_regions = metrics_df[metrics_df["Region"] != "Overall"].copy()
    matrices = []
    labels = []
    for _, row in confusion_regions.iterrows():
        matrix = np.array(
            [[row["TP"], row["FN"]], [row["FP"], row["TN"]]], dtype=float
        )
        matrices.append(matrix)
        labels.append(row["Region"])

    combined_matrix = np.concatenate(matrices, axis=1)
    heatmap_fig, ax_hm = plt.subplots(figsize=(18, 6))
    im = ax_hm.imshow(combined_matrix, cmap="YlOrRd")

    num_regions = len(labels)
    ax_hm.set_xticks(np.arange(num_regions * 2))
    tick_labels = []
    for label in labels:
        tick_labels.extend([f"{label}\nTP", f"{label}\nFN"])
    ax_hm.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax_hm.set_yticks([0, 1])
    ax_hm.set_yticklabels(["Observed Degraded", "Observed Intact"], fontweight="bold")

    vmax = np.nanmax(combined_matrix)
    if vmax > 0:
        im.set_clim(0, vmax)

    for i in range(combined_matrix.shape[0]):
        for j in range(combined_matrix.shape[1]):
            value = int(combined_matrix[i, j])
            ax_hm.text(
                j,
                i,
                f"{value}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    ax_hm.set_title(
        "Confusion Matrix Heatmap by Region (Observed vs Predicted)",
        fontsize=16,
        fontweight="bold",
    )
    ax_hm.grid(False)
    heatmap_fig.tight_layout()
    heatmap_fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    heatmap_png = output_dir / f"confusion_heatmap_{analysis_tag}.png"
    heatmap_fig.savefig(heatmap_png, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(heatmap_fig)


if __name__ == "__main__":
    main()
