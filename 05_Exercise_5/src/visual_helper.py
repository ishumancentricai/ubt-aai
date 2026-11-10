import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import geopandas as gpd
from geodatasets import get_path

warnings.simplefilter(action="ignore", category=FutureWarning)
plt.style.use(os.path.join(os.getcwd(), "configs", "visualisations.mplstyle"))
defined_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _plot_histogram_bins(df, diff_column, hist_column, number_bins=30):

  df["Continent Category"] = df["Continent"].apply(
    lambda x: x if x in ["Africa", "Asia", "Americas", "Europe"] else "Other"
  )

  df_plot = df[
    df[hist_column].notna() & np.isfinite(df[hist_column]) & df[diff_column].notna()
  ]
  df_plot = df_plot[df_plot[hist_column] > 0]

  for category in [
    "Asia",
    "Americas",
    "Europe",
    "Africa",
  ]:
    # ---- YOUR CODE GOES HERE (START) ---
    subset = df_plot[df_plot[diff_column] == category]
    log_values = np.log10(subset[hist_column])
    plt.hist(log_values, label=category, bins=number_bins, fill=True, alpha=1)
    # --- YOUR CODE GOES HERE (END) ---

  plt.title(f"Histogram of {hist_column}, {number_bins} bins")
  plt.xlabel(hist_column)
  plt.xticks(
    ticks=np.log10([1e4, 1e6, 1e8, 1e10]),
    labels=["$10^4$", "$10^6$", "$10^8$", "$10^{10}$"],
  )
  plt.ylabel("Event Count")
  plt.legend(title=diff_column)
  plt.show()


def _plot_histogram_descriptives(
  df,
  hist_column="Total Damages (US$)",
  number_bins=50,
  five_percentile=False,
  median=False,
  mean=False,
  mode=False,
  ninety_five_percentile=False,
  std=False,
  iqr=False,
  range=False,
  line_width=3,
):
  defined_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

  df_plot = df[df[hist_column].notna() & np.isfinite(df[hist_column])]
  df_plot = df_plot[df_plot[hist_column] > 0]
  log_values = np.log10(df[hist_column])

  plt.hist(log_values, bins=number_bins, fill=True, alpha=1)
  plt.title(f"Histogram of {hist_column}, {number_bins} bins")
  plt.xlabel(hist_column)
  plt.xticks(
    ticks=np.log10([1e4, 1e6, 1e8, 1e10]),
    labels=["$10^4$", "$10^6$", "$10^8$", "$10^{10}$"],
  )

  # Location parameter
  if five_percentile:
    five_percentile_value = log_values.quantile(0.05)
    plt.axvline(
      five_percentile_value,
      color=defined_colors[3],
      linestyle="dashed",
      linewidth=line_width,
      label="5th Percentile",
    )
  if mean:
    mean_value = log_values.mean()
    plt.axvline(
      mean_value,
      color=defined_colors[0],
      linestyle="dashed",
      linewidth=line_width,
      label="Mean",
    )
  if median:
    # ---- YOUR CODE GOES HERE (START) ---
    median_value = log_values.median()
    plt.axvline(
      median_value,
      color=defined_colors[1],
      linestyle="dashed",
      linewidth=line_width,
      label="Median",
    )
    # --- YOUR CODE GOES HERE (END) ---
  if mode:
    mode_value = log_values.mode()[0]
    plt.axvline(
      mode_value,
      color=defined_colors[2],
      linestyle="dashed",
      linewidth=line_width,
      label="Mode",
    )
  if ninety_five_percentile:
    ninety_five_percentile_value = log_values.quantile(0.95)
    plt.axvline(
      ninety_five_percentile_value,
      color=defined_colors[4],
      linestyle="dashed",
      linewidth=line_width,
      label="95th Percentile",
    )

  # Dispersion parameter
  if std:
    # ---- YOUR CODE GOES HERE (START) ---
    mean_value = log_values.mean()
    std = log_values.std()
    plt.axvline(
      mean_value + std,
      color=defined_colors[1],
      linestyle="dashed",
      linewidth=line_width,
      label="Mean +/- Std.",
    )
    plt.axvline(
      mean_value - std,
      color=defined_colors[1],
      linestyle="dashed",
      linewidth=line_width,
      label="",
    )
    # --- YOUR CODE GOES HERE (END) ---
        
  if iqr:
    interquartile_range = log_values.quantile(0.75) - log_values.quantile(0.25)
    plt.axvline(
      log_values.quantile(0.25),
      color=defined_colors[3],
      linestyle="dashed",
      linewidth=line_width,
      label="1st/3rd Quartile",
    )
    plt.axvline(
      log_values.quantile(0.75),
      color=defined_colors[3],
      linestyle="dashed",
      linewidth=line_width,
      label="",
    )
  if range:
    range_values = (log_values.min(), log_values.max())
    plt.axvline(
      range_values[0],
      color=defined_colors[2],
      linestyle="dashed",
      linewidth=line_width,
      label="Min/Max",
    )
    plt.axvline(
      range_values[1],
      color=defined_colors[2],
      linestyle="dashed",
      linewidth=line_width,
      label="",
    )

  plt.legend()
  plt.ylim(0, 500)
  plt.ylabel("Event Count")
  plt.show()


def _plot_scatter_confidence(df, independant_variable, dependent_variable, n=1):
  mask = (df[independant_variable] > 0) & (df[dependent_variable] > 0)
  x = np.log10(df.loc[mask, independant_variable])
  y = np.log10(df.loc[mask, dependent_variable])

  plt.scatter(x, y, s=20, marker='o', alpha=1, edgecolors='w', linewidth=0.5)

  coefficients = np.polyfit(x, y, n)
  poly = np.poly1d(coefficients)
  x_fit = np.linspace(x.min(), x.max(), 100)
  y_fit = poly(x_fit)

  # Calculate confidence interval (approximate, using residual std)
  y_pred = poly(x)
  residuals = y - y_pred
  std_err = residuals.std()
  # 95% confidence interval (approximate, not accounting for leverage)
  ci = 1.96 * std_err

  plt.plot(x_fit, y_fit, color='red', linewidth=2, label=f'Polyfit ${n}^{{nd}}$ Degree')
  plt.fill_between(x_fit, y_fit - ci, y_fit + ci, color='red', alpha=0.2, label='95% CI')

  plt.xticks(ticks=np.log10([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]), labels=['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'])
  plt.yticks(ticks=np.log10([1e4, 1e6, 1e8, 1e10, 1e12]), labels=['$10^4$', '$10^6$', '$10^8$', '$10^{10}$', '$10^{12}$'])
  plt.legend()
  plt.xlabel(f'{independant_variable}')
  plt.ylabel(f'{dependent_variable}')
  plt.title(f'Scatterplot of {independant_variable} vs. {dependent_variable}')
  plt.show()

def _plot_disaster_map(df, disaster="Volcanic activity"):
  world = gpd.read_file(get_path("naturalearth.land"))
  ax = world.plot(color="white", edgecolor="black")

  df = df[df["Disaster Type"].isin([disaster])]

  gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
    crs="EPSG:4326",
  )
  gdf.plot(
    ax=ax,
    column="Disaster Type",
    categorical=True,
    legend=False,
    markersize=30,
    alpha=0.8,
  )
  plt.title(f"Map {disaster} events")
  plt.xlabel("Longitude")
  plt.ylabel("Latitude")
  plt.show()