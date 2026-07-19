# Phase 1 ESDA Results

## Mann-Kendall + Sen's Slope (Lagos-wide summary)
|   n_valid_pixels |   significant_increasing_pixels |   significant_decreasing_pixels |   non_significant_pixels |   median_sen_slope_c_per_year |   mean_sen_slope_c_per_year |   median_tau |   alpha |
|-----------------:|--------------------------------:|--------------------------------:|-------------------------:|------------------------------:|----------------------------:|-------------:|--------:|
|             3308 |                            2883 |                               0 |                      425 |                     0.0668526 |                   0.0765173 |     0.486667 |    0.05 |

## Bivariate Spatial Correlations
| x_name   | y_name   |   pearson_r |   pearson_p |   spearman_rho |   spearman_p |    n |
|:---------|:---------|------------:|------------:|---------------:|-------------:|-----:|
| NDBI     | LST      |    0.912596 |           0 |       0.904534 |            0 | 3128 |
| NDVI     | LST      |   -0.766026 |           0 |      -0.664278 |            0 | 3128 |

## NDBI-LST Quadrant Overlay
|   quadrant_code | quadrant             |   pixel_count |   percentage |   ndbi_threshold |   lst_threshold |
|----------------:|:---------------------|--------------:|-------------:|-----------------:|----------------:|
|               1 | Low NDBI / Low LST   |          1420 |     42.9262  |        -0.132022 |         28.3032 |
|               2 | High NDBI / Low LST  |           234 |      7.07376 |        -0.132022 |         28.3032 |
|               3 | Low NDBI / High LST  |           234 |      7.07376 |        -0.132022 |         28.3032 |
|               4 | High NDBI / High LST |          1420 |     42.9262  |        -0.132022 |         28.3032 |

Critical UHI core (High NDBI / High LST): **42.93%** of valid Lagos pixels.
