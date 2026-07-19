#!/usr/bin/env python3
"""Phase 2 predictive modeling for Lagos UHI.

Implements:
1) MLR baseline: LST ~ time + population density + NDBI + NDVI
2) CNN-LSTM-Attention sequence model for annual LST grid forecasting
3) 2030 and 2040 projections with model comparison metrics
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 LST predictive modeling")
    parser.add_argument("--lagos-boundary", required=True)
    parser.add_argument("--modis-lst-pattern", required=True)
    parser.add_argument("--ndbi-pattern", required=True)
    parser.add_argument("--ndvi-pattern", required=True)
    parser.add_argument("--population-pattern", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-end-year", type=int, default=2018)
    parser.add_argument("--val-end-year", type=int, default=2021)
    parser.add_argument("--target-years", nargs="+", type=int, default=[2030, 2040])
    return parser.parse_args()


def find_files(pattern: str) -> list[Path]:
    files = sorted(Path().glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return files


def parse_timestamp(stem: str) -> pd.Timestamp | pd.NaT:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    if match:
        return pd.to_datetime(match.group(1), errors="coerce")
    return pd.to_datetime(stem, errors="coerce")


def read_raster_stack(pattern: str) -> xr.DataArray:
    files = find_files(pattern)
    arrs = []
    ts = []
    for file in files:
        da = xr.open_dataarray(file)
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.squeeze("band", drop=True)
        arrs.append(da)
        ts.append(parse_timestamp(file.stem))

    if any(pd.isna(t) for t in ts):
        ts = list(pd.date_range("2000-01-01", periods=len(files), freq="YS"))

    return xr.concat(arrs, dim="time").assign_coords(time=pd.DatetimeIndex(ts))


def load_polygon(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Boundary file is empty")
    return gdf


def clip_to_boundary(da: xr.DataArray, boundary: gpd.GeoDataFrame) -> xr.DataArray:
    target = boundary
    if da.rio.crs is None:
        raise ValueError("Raster CRS undefined")
    if boundary.crs is None:
        raise ValueError("Boundary CRS undefined")
    if boundary.crs != da.rio.crs:
        target = boundary.to_crs(da.rio.crs)
    return da.rio.clip(target.geometry, target.crs, drop=True)


def annualize(da: xr.DataArray) -> xr.DataArray:
    return da.resample(time="YS").mean(skipna=True)


def reproject_match_stack(source: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Reproject a time stack to target grid (CRS, resolution, extent)."""
    return source.rio.reproject_match(target)


def align_years(lst: xr.DataArray, ndbi: xr.DataArray, ndvi: xr.DataArray, pop: xr.DataArray):
    # Convert to year coordinate for robust alignment.
    lst_y = lst.assign_coords(year=lst.time.dt.year).swap_dims({"time": "year"}).drop_vars("time")
    ndbi_y = ndbi.assign_coords(year=ndbi.time.dt.year).swap_dims({"time": "year"}).drop_vars("time")
    ndvi_y = ndvi.assign_coords(year=ndvi.time.dt.year).swap_dims({"time": "year"}).drop_vars("time")
    pop_y = pop.assign_coords(year=pop.time.dt.year).swap_dims({"time": "year"}).drop_vars("time")

    # Fill missing years in population by linear interpolation then edge fill.
    full_years = np.arange(int(min(lst_y.year.values)), int(max(lst_y.year.values)) + 1)
    pop_y = pop_y.reindex(year=full_years)
    pop_y = pop_y.interpolate_na(dim="year", method="linear", fill_value="extrapolate")
    pop_y = pop_y.ffill(dim="year").bfill(dim="year")

    years = np.intersect1d(np.intersect1d(lst_y.year.values, ndbi_y.year.values), ndvi_y.year.values)
    years = np.intersect1d(years, pop_y.year.values)

    return (
        lst_y.sel(year=years),
        ndbi_y.sel(year=years),
        ndvi_y.sel(year=years),
        pop_y.sel(year=years),
    )


def build_pixel_dataframe(
    lst_y: xr.DataArray,
    ndbi_y: xr.DataArray,
    ndvi_y: xr.DataArray,
    pop_y: xr.DataArray,
) -> pd.DataFrame:
    rows = []
    for year in lst_y.year.values:
        lst_i = lst_y.sel(year=year)
        ndbi_i = ndbi_y.sel(year=year)
        ndvi_i = ndvi_y.sel(year=year)
        pop_i = pop_y.sel(year=year)

        # Align grids exactly.
        lst_i, ndbi_i, ndvi_i, pop_i = xr.align(lst_i, ndbi_i, ndvi_i, pop_i, join="inner")

        arr_lst = lst_i.values.ravel()
        arr_ndbi = ndbi_i.values.ravel()
        arr_ndvi = ndvi_i.values.ravel()
        arr_pop = pop_i.values.ravel()

        valid = np.isfinite(arr_lst) & np.isfinite(arr_ndbi) & np.isfinite(arr_ndvi) & np.isfinite(arr_pop)

        if not valid.any():
            continue

        frame = pd.DataFrame(
            {
                "year": int(year),
                "time_idx": int(year) - int(lst_y.year.values.min()),
                "lst": arr_lst[valid],
                "ndbi": arr_ndbi[valid],
                "ndvi": arr_ndvi[valid],
                "pop_density": arr_pop[valid],
            }
        )
        rows.append(frame)

    if not rows:
        raise ValueError("No valid data rows built for MLR.")
    return pd.concat(rows, ignore_index=True)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_mlr(df: pd.DataFrame, train_end: int, val_end: int):
    x_cols = ["time_idx", "pop_density", "ndbi", "ndvi"]
    train = df[df["year"] <= train_end]
    val = df[(df["year"] > train_end) & (df["year"] <= val_end)]
    test = df[df["year"] > val_end]

    model = LinearRegression()
    model.fit(train[x_cols], train["lst"])

    preds = {
        "train": model.predict(train[x_cols]),
        "val": model.predict(val[x_cols]),
        "test": model.predict(test[x_cols]),
    }

    metrics = pd.DataFrame(
        [
            {"split": "train", **evaluate(train["lst"].to_numpy(), preds["train"])},
            {"split": "val", **evaluate(val["lst"].to_numpy(), preds["val"])},
            {"split": "test", **evaluate(test["lst"].to_numpy(), preds["test"])},
        ]
    )

    coefs = pd.DataFrame(
        {
            "feature": x_cols,
            "coefficient": model.coef_,
        }
    )
    coefs = pd.concat(
        [pd.DataFrame([{"feature": "intercept", "coefficient": model.intercept_}]), coefs],
        ignore_index=True,
    )

    return model, metrics, coefs


def extrapolate_feature_stack(feature_y: xr.DataArray, target_years: list[int]) -> xr.DataArray:
    years_hist = feature_y.year.values.astype(float)
    values = feature_y.values  # year, y, x
    out = []
    for ty in target_years:
        # Linear trend per pixel based on all historical years.
        x = years_hist
        y = values

        x_mean = np.nanmean(x)
        y_mean = np.nanmean(y, axis=0)
        cov = np.nanmean((x[:, None, None] - x_mean) * (y - y_mean), axis=0)
        var = np.nanmean((x - x_mean) ** 2)
        slope = cov / var if var != 0 else np.zeros_like(cov)
        intercept = y_mean - slope * x_mean
        pred = intercept + slope * ty
        out.append(pred)

    data = np.stack(out, axis=0)
    return xr.DataArray(data, coords={"year": target_years, "y": feature_y.y, "x": feature_y.x}, dims=("year", "y", "x"))


def try_import_tf():
    try:
        import tensorflow as tf

        return tf
    except Exception:
        return None


def build_sequence_tensors(lst_y, ndbi_y, ndvi_y, pop_y, seq_len: int):
    years = lst_y.year.values.astype(int)
    xs = []
    ys = []
    y_years = []

    # Min-max scale each variable globally to [0, 1].
    def mm(a):
        amin = np.nanmin(a)
        amax = np.nanmax(a)
        if np.isclose(amax, amin):
            return np.zeros_like(a), (amin, amax)
        return (a - amin) / (amax - amin), (amin, amax)

    lst_scaled, lst_rng = mm(lst_y.values)
    ndbi_scaled, ndbi_rng = mm(ndbi_y.values)
    ndvi_scaled, ndvi_rng = mm(ndvi_y.values)
    pop_scaled, pop_rng = mm(pop_y.values)

    for i in range(seq_len, len(years)):
        x_seq = np.stack(
            [
                lst_scaled[i - seq_len : i],
                ndvi_scaled[i - seq_len : i],
                ndbi_scaled[i - seq_len : i],
                pop_scaled[i - seq_len : i],
            ],
            axis=-1,
        )  # seq, y, x, channels
        y_map = lst_scaled[i]

        # Fill remaining NaNs with zero after scaling.
        x_seq = np.nan_to_num(x_seq, nan=0.0)
        y_map = np.nan_to_num(y_map, nan=0.0)

        xs.append(x_seq)
        ys.append(y_map)
        y_years.append(years[i])

    x_arr = np.stack(xs, axis=0)
    y_arr = np.stack(ys, axis=0)

    scalers = {"lst": lst_rng, "ndbi": ndbi_rng, "ndvi": ndvi_rng, "pop": pop_rng}
    return x_arr, y_arr, np.asarray(y_years), scalers


def build_cnn_lstm_attention_model(tf, seq_len: int, h: int, w: int, c: int):
    inputs = tf.keras.Input(shape=(seq_len, h, w, c))

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"))(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)

    lstm_out = tf.keras.layers.LSTM(64, return_sequences=True)(x)

    # Temporal attention.
    attn_scores = tf.keras.layers.Dense(1, activation="tanh")(lstm_out)
    attn_scores = tf.keras.layers.Softmax(axis=1)(attn_scores)
    context = tf.keras.layers.Multiply()([lstm_out, attn_scores])
    context = tf.keras.layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(context)

    out = tf.keras.layers.Dense(h * w, activation="linear")(context)
    out = tf.keras.layers.Reshape((h, w))(out)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def run_cnn_lstm_attention(
    lst_y,
    ndbi_y,
    ndvi_y,
    pop_y,
    seq_len: int,
    train_end: int,
    val_end: int,
    epochs: int,
    batch_size: int,
):
    tf = try_import_tf()
    if tf is None:
        return None, None, "TensorFlow is not available in this environment; CNN-LSTM-Attention was skipped."

    x_arr, y_arr, y_years, scalers = build_sequence_tensors(lst_y, ndbi_y, ndvi_y, pop_y, seq_len)

    train_idx = y_years <= train_end
    val_idx = (y_years > train_end) & (y_years <= val_end)
    test_idx = y_years > val_end

    h, w = y_arr.shape[1], y_arr.shape[2]
    model = build_cnn_lstm_attention_model(tf, seq_len=seq_len, h=h, w=w, c=x_arr.shape[-1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ]

    model.fit(
        x_arr[train_idx],
        y_arr[train_idx],
        validation_data=(x_arr[val_idx], y_arr[val_idx]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    def unscale_lst(arr):
        amin, amax = scalers["lst"]
        return arr * (amax - amin) + amin

    metrics_rows = []
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        pred = model.predict(x_arr[idx], verbose=0)
        y_true = unscale_lst(y_arr[idx]).ravel()
        y_pred = unscale_lst(pred).ravel()
        metrics_rows.append({"split": split_name, **evaluate(y_true, y_pred)})

    metrics_df = pd.DataFrame(metrics_rows)

    context = {
        "model": model,
        "scalers": scalers,
        "seq_x": x_arr,
        "seq_y": y_arr,
        "seq_years": y_years,
    }
    return context, metrics_df, None


def forecast_mlr(
    mlr_model: LinearRegression,
    lst_y: xr.DataArray,
    ndbi_y: xr.DataArray,
    ndvi_y: xr.DataArray,
    pop_y: xr.DataArray,
    target_years: list[int],
):
    ndbi_f = extrapolate_feature_stack(ndbi_y, target_years)
    ndvi_f = extrapolate_feature_stack(ndvi_y, target_years)
    pop_f = extrapolate_feature_stack(pop_y, target_years)

    hist_means = {
        "pop_density": float(np.nanmean(pop_y.values)),
        "ndbi": float(np.nanmean(ndbi_y.values)),
        "ndvi": float(np.nanmean(ndvi_y.values)),
    }

    preds = []
    base_year = int(lst_y.year.values.min())
    for year in target_years:
        ndbi_i, ndvi_i, pop_i = xr.align(ndbi_f.sel(year=year), ndvi_f.sel(year=year), pop_f.sel(year=year), join="inner")
        X = pd.DataFrame(
            {
                "time_idx": np.full(ndbi_i.size, year - base_year),
                "pop_density": pop_i.values.ravel(),
                "ndbi": ndbi_i.values.ravel(),
                "ndvi": ndvi_i.values.ravel(),
            }
        )
        X = X.fillna(value=hist_means)
        y_pred = mlr_model.predict(X).reshape(ndbi_i.shape)
        da = xr.DataArray(y_pred, coords={"y": ndbi_i.y, "x": ndbi_i.x}, dims=("y", "x"))
        da = da.assign_coords(year=year).expand_dims("year")
        preds.append(da)

    return xr.concat(preds, dim="year")


def forecast_cnn_lstm(
    context,
    lst_y: xr.DataArray,
    ndbi_y: xr.DataArray,
    ndvi_y: xr.DataArray,
    pop_y: xr.DataArray,
    target_years: list[int],
    seq_len: int,
):
    model = context["model"]
    scalers = context["scalers"]

    years_hist = list(lst_y.year.values.astype(int))
    max_hist_year = max(years_hist)

    # Build extrapolated predictors through max target year.
    future_years = list(range(max_hist_year + 1, max(target_years) + 1))
    ndbi_f = extrapolate_feature_stack(ndbi_y, future_years)
    ndvi_f = extrapolate_feature_stack(ndvi_y, future_years)
    pop_f = extrapolate_feature_stack(pop_y, future_years)

    def scale(arr, rng):
        amin, amax = rng
        if np.isclose(amax, amin):
            return np.zeros_like(arr)
        return (arr - amin) / (amax - amin)

    def unscale(arr, rng):
        amin, amax = rng
        return arr * (amax - amin) + amin

    lst_series = {int(y): lst_y.sel(year=y).values for y in years_hist}
    ndbi_series = {int(y): ndbi_y.sel(year=y).values for y in years_hist}
    ndvi_series = {int(y): ndvi_y.sel(year=y).values for y in years_hist}
    pop_series = {int(y): pop_y.sel(year=y).values for y in years_hist}

    for y in future_years:
        ndbi_series[y] = ndbi_f.sel(year=y).values
        ndvi_series[y] = ndvi_f.sel(year=y).values
        pop_series[y] = pop_f.sel(year=y).values

    for year in future_years:
        seq_years = list(range(year - seq_len, year))

        x_lst = np.stack([scale(np.nan_to_num(lst_series[yy], nan=0.0), scalers["lst"]) for yy in seq_years], axis=0)
        x_ndvi = np.stack([scale(np.nan_to_num(ndvi_series[yy], nan=0.0), scalers["ndvi"]) for yy in seq_years], axis=0)
        x_ndbi = np.stack([scale(np.nan_to_num(ndbi_series[yy], nan=0.0), scalers["ndbi"]) for yy in seq_years], axis=0)
        x_pop = np.stack([scale(np.nan_to_num(pop_series[yy], nan=0.0), scalers["pop"]) for yy in seq_years], axis=0)

        x_seq = np.stack([x_lst, x_ndvi, x_ndbi, x_pop], axis=-1)
        x_seq = np.expand_dims(x_seq, axis=0)

        pred_scaled = model.predict(x_seq, verbose=0)[0]
        pred_lst = unscale(pred_scaled, scalers["lst"])
        lst_series[year] = pred_lst

    out_maps = []
    for ty in target_years:
        da = xr.DataArray(lst_series[ty], coords={"y": lst_y.y, "x": lst_y.x}, dims=("y", "x"))
        da = da.assign_coords(year=ty).expand_dims("year")
        out_maps.append(da)

    return xr.concat(out_maps, dim="year")


def summarize_forecasts(forecast_da: xr.DataArray, model_name: str) -> pd.DataFrame:
    rows = []
    for year in forecast_da.year.values:
        arr = forecast_da.sel(year=year).values
        rows.append(
            {
                "model": model_name,
                "year": int(year),
                "mean_lst": float(np.nanmean(arr)),
                "median_lst": float(np.nanmedian(arr)),
                "min_lst": float(np.nanmin(arr)),
                "max_lst": float(np.nanmax(arr)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boundary = load_polygon(args.lagos_boundary)

    lst = clip_to_boundary(read_raster_stack(args.modis_lst_pattern), boundary)
    ndbi = clip_to_boundary(read_raster_stack(args.ndbi_pattern), boundary)
    ndvi = clip_to_boundary(read_raster_stack(args.ndvi_pattern), boundary)
    pop = clip_to_boundary(read_raster_stack(args.population_pattern), boundary)

    # Force all predictors to the LST grid before annual joins.
    ndbi = reproject_match_stack(ndbi, lst)
    ndvi = reproject_match_stack(ndvi, lst)
    pop = reproject_match_stack(pop, lst)

    lst_y, ndbi_y, ndvi_y, pop_y = align_years(annualize(lst), annualize(ndbi), annualize(ndvi), annualize(pop))

    # MLR baseline
    df = build_pixel_dataframe(lst_y, ndbi_y, ndvi_y, pop_y)
    mlr_model, mlr_metrics, mlr_coefs = fit_mlr(df, train_end=args.train_end_year, val_end=args.val_end_year)
    mlr_metrics.to_csv(out_dir / "mlr_metrics.csv", index=False)
    mlr_coefs.to_csv(out_dir / "mlr_coefficients.csv", index=False)

    mlr_forecast = forecast_mlr(mlr_model, lst_y, ndbi_y, ndvi_y, pop_y, args.target_years)
    mlr_forecast.rio.write_crs(lst.rio.crs, inplace=True)
    for year in args.target_years:
        mlr_forecast.sel(year=year).rio.to_raster(out_dir / f"mlr_lst_forecast_{year}.tif")

    # CNN-LSTM-Attention
    cnn_context, cnn_metrics, cnn_error = run_cnn_lstm_attention(
        lst_y=lst_y,
        ndbi_y=ndbi_y,
        ndvi_y=ndvi_y,
        pop_y=pop_y,
        seq_len=args.sequence_length,
        train_end=args.train_end_year,
        val_end=args.val_end_year,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    model_comparison_rows = []
    mlr_test = mlr_metrics.loc[mlr_metrics["split"] == "test"].iloc[0].to_dict()
    model_comparison_rows.append({"model": "MLR", **mlr_test})

    forecast_tables = [summarize_forecasts(mlr_forecast, "MLR")]

    if cnn_error:
        (out_dir / "cnn_lstm_attention_status.json").write_text(
            json.dumps({"status": "skipped", "reason": cnn_error}, indent=2),
            encoding="utf-8",
        )
    else:
        cnn_metrics.to_csv(out_dir / "cnn_lstm_attention_metrics.csv", index=False)
        cnn_test = cnn_metrics.loc[cnn_metrics["split"] == "test"].iloc[0].to_dict()
        model_comparison_rows.append({"model": "CNN-LSTM-Attention", **cnn_test})

        cnn_forecast = forecast_cnn_lstm(
            context=cnn_context,
            lst_y=lst_y,
            ndbi_y=ndbi_y,
            ndvi_y=ndvi_y,
            pop_y=pop_y,
            target_years=args.target_years,
            seq_len=args.sequence_length,
        )
        cnn_forecast.rio.write_crs(lst.rio.crs, inplace=True)
        for year in args.target_years:
            cnn_forecast.sel(year=year).rio.to_raster(out_dir / f"cnn_lstm_attention_lst_forecast_{year}.tif")
        forecast_tables.append(summarize_forecasts(cnn_forecast, "CNN-LSTM-Attention"))

    pd.DataFrame(model_comparison_rows).to_csv(out_dir / "model_comparison.csv", index=False)
    pd.concat(forecast_tables, ignore_index=True).to_csv(out_dir / "forecast_summary_2030_2040.csv", index=False)

    print("Phase 2 modeling complete.")
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
