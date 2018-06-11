# outlierml

`outlierml` is a small library to detect outliers in 2D arrays based on scikit-learn outlier detection functions: Local Outlier Factor (LOF), Robust Covariance (RC) and Isolation Forest (IF).

It includes a command-line tool that can be called through

```>> outlierml -i <initfile>```

where `<initfile>` is a text file including the following information

```
[outlierml]
file          : NetCDF file path
method        : <LOF> for Local Outlier Factor , <RC> for Robust Covariance , <IF> for Isolation Forest
outputdir     : Output directory
contamination : Contamination fraction from 0 to 1
decomposition : True or False time series to deseasonalization
```

The outlierml module can also be called via

```python
from outlierml import run_outlierml
```

where `run_outlierml` is a function which receives the `xarray.Dataset` object to be analised and returns 1) a mask containing outliers in a `xarray.DataArray` object and a `pd.DataFrame` object
