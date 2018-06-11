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

### Isolation forest
One efficient way of performing outlier detection in high-dimensional datasets
is to use random forests. The ensemble.IsolationForest ‘isolates’ observations
by randomly selecting a feature and then randomly selecting a split value between
the maximum and minimum values of the selected feature.

**References:**
Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. “Isolation forest.” Data Mining, 2008.
ICDM‘08. Eighth IEEE International Conference on.

### Local outlier factor
The neighbors.LocalOutlierFactor (LOF) algorithm computes a score (called local outlier factor)
reflecting the degree of abnormality of the observations. It measures the local density deviation
of a given data point with respect to its neighbors. The idea is to detect the samples that have
a substantially lower density than their neighbors.

**References:**
Breunig, Kriegel, Ng, and Sander (2000) LOF: identifying density-based local outliers.
Proc. ACM SIGMOD

### Roust covariance
The scikit-learn provides an object covariance.EllipticEnvelope that fits a
robust covariance estimate to the data, and thus fits an ellipse to the central
data points, ignoring points outside the central mode.

**References:**
Rousseeuw, P.J., Van Driessen, K. “A fast algorithm for the minimum covariance determinant estimator”.
Technometrics 41(3), 212 (1999)
