#!/usr/bin/env python

import os
import sys
import getopt
import warnings; warnings.simplefilter("ignore")
import configparser
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose

def main(argv):

    """
    Called by the command-line tool outlierml

    Usage
    -----
    >> outlierml -i initfile.cfg

    <initfile.cfg> contains
    -----------------------
    [outlierml]
    file          : NetCDF file path
    method        : Local Outlier Factor (LOF), Robust Covariance (RC), Isolation Forest (IF)
    outputdir     : Output directory
    contamination : Contamination fraction from 0 to 1
    """

    opts, args = getopt.getopt(argv,"i:")
    for o, a in opts:
        if o == "-i": inifile = a

    config = configparser.SafeConfigParser()
    config.read(inifile)

    file          = str(config.get('outlierml','file'))
    method        = str(config.get('outlierml','method'))
    decomposition = str(config.get('outlierml','decomposition'))
    outputdir     = str(config.get('outlierml','outputdir'))
    contamination = np.float64(config.get('outlierml','contamination'))

    file_name, file_extension = os.path.splitext(file)

    if file_extension == '.nc':
        nc = read_netcdf(file)
    elif file_extension == '.csv':
        df = read_csv(file)
    elif file_extension == '.txt':
        df = read_csv(file)
    else:
        sys.exit('ERROR: File extension not recognised')

    varname = list(nc.data_vars.keys())[0]

    if 't' in list(nc.coords.keys()):
        tinname = 't'
    elif 'time' in list(nc.coords.keys()):
        timname = 'time'
    else:
        sys.exit('ERROR: coordinate label should be <t> or <time>')

    if 'lat' in list(nc.coords.keys()):
        latname = 'lat'
    elif 'latitude' in list(nc.coords.keys()):
        latname = 'latitude'
    else:
        sys.exit('ERROR: coordinate label should be <lat> or <latitude>')

    if 'lon' in list(nc.coords.keys()):
        lonname = 'lon'
    elif 'longitude' in list(nc.coords.keys()):
        lonname = 'longitude'
    else:
        sys.exit('ERROR: coordinate label should be <lon> or <longitude>')

    csv,foo = run_outlierml(nc,method,contamination,varname,latname,lonname,timname,decomposition)

    # If command-line option is used two files are created log.csv and stats.nc
    csv.to_csv(outputdir+'log.csv')
    foo.to_netcdf(outputdir+'stats.nc',encoding={'myfreq': {'zlib': True},'mymean': {'zlib': True},'mystd': {'zlib': True}})

def run_outlierml(nc,method,contamination,varname,latname,lonname,timname,decomposition):

    """
    Program which detects outliers in xarray.dataset
    nc            : (xarray.Dataset)
    method        : Local Outlier Factor (LOF), Robust Covariance (RC), Isolation Forest (IF)
    contamination : Contamination fraction from 0 to 1
    varname       : (string) with varname label
    latname       : (string) with latitude label
    lonname       : (string) with longitude label
    timname       : (string) with time label

    Returns:
    foo : (xarray.DataArray) containing freq, mean, std
    csv : (pd.DataFrame) containing time, lat, lon, value, mean, std
    """

    var    = nc[varname]
    mymean = var.mean(dim=timname)
    mystd  = var.std(dim=timname)
    arr    = var.values

    csv = pd.DataFrame()
    res = np.zeros_like(arr)
    for j in range(res.shape[1]): # lat
        print(j)
        for i in range(res.shape[2]): # lon

            if decomposition:
                try:
                    thearr = seas_dec(arr[:,j,i],nc[timname].values)
                except ValueError:
                    print('Multiplicative seasonality is not appropriate for zero and negative values')
                    thearr = arr[:,j,i]
            else:
                thearr = arr[:,j,i]

            if method == 'LOF':
                myvec = localoutlierfactor(thearr,contamination)
            elif method == 'RC':
                myvec = robustcovariance(thearr,contamination)
            elif method == 'IF':
                myvec = isolationforest(thearr,contamination)
            else:
                sys.exit('ERROR: method not recognised should be <loc> or <rc>')

            res[:,j,i]     = myvec
            idx            = np.where(myvec>0)
            
            dates          = nc.indexes['time'][idx].strftime('%Y-%m-%d')
            thelat         = nc[latname].values[j]
            thelon         = nc[lonname].values[i]
            theval         = np.squeeze(arr[idx,j,i])
            themean        = np.array(mymean[j,i])
            thestd         = np.array(mystd[j,i])

            if len(idx)>0:
                thecsv         = pd.DataFrame({'time':dates})
                thecsv['lat']  = thelat
                thecsv['lon']  = thelon
                thecsv['ilat'] = j
                thecsv['ilon'] = i
                thecsv['val']  = theval
                thecsv['mean'] = themean
                thecsv['std']  = thestd
                csv = pd.concat((csv,thecsv))

    # Setting index and saving csv
    csv.set_index(csv['time'],inplace=True)
    del csv['time']

    # Saving netcdf
    foo                 = xr.Dataset()
    foo['myfreq']       = ((timname, latname, lonname), res)
    foo['mystd']        = mystd
    foo['mymean']       = mymean
    foo.coords[lonname] = ((lonname), nc[lonname].values)
    foo.coords[latname] = ((latname), nc[latname].values)
    foo.coords[timname] = nc[timname].values

    return csv, foo

def read_netcdf(file):

    """
    Returns a xarray.Dataset
    """
    nc = xr.open_dataset(file)
    return nc

def read_csv(file):

    """
    Returns a pd.DataFrame
    """
    df = pd.read_csv(file)
    return df

def isolationforest(nparray,contamination):

    """
    One efficient way of performing outlier detection in high-dimensional datasets
    is to use random forests. The ensemble.IsolationForest ‘isolates’ observations
    by randomly selecting a feature and then randomly selecting a split value between
    the maximum and minimum values of the selected feature.

    References:
    Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. “Isolation forest.” Data Mining, 2008.
    ICDM‘08. Eighth IEEE International Conference on.
    """

    df = pd.DataFrame(nparray)

    rng = np.random.RandomState(42)

    # Split data in training and testing
    train, test = train_test_split(df, test_size=0.66, random_state=rng)

    # Fit the model
    clf = IsolationForest(contamination=contamination, random_state=rng)
    clf.fit(train)
    y_pred_train = clf.predict(train)
    y_pred_test  = clf.predict(test)

    train['IF'] = y_pred_train
    test['IF']  = y_pred_test
    tmp_df = pd.concat([train,test]).sort_index()
    y_pred = tmp_df['IF'].values

    # df['IF'] = y_pred
    # ax = df[df['IF']==1]['value'].plot(style='.')
    # df[df['IF']==-1]['value'].plot(style='.',ax=ax)

    return y_pred

def localoutlierfactor(nparray,contamination):

    """
    The neighbors.LocalOutlierFactor (LOF) algorithm computes a score (called local outlier factor)
    reflecting the degree of abnormality of the observations. It measures the local density deviation
    of a given data point with respect to its neighbors. The idea is to detect the samples that have
    a substantially lower density than their neighbors.

    References:
    Breunig, Kriegel, Ng, and Sander (2000) LOF: identifying density-based local outliers.
    Proc. ACM SIGMOD
    """

    df = pd.DataFrame(nparray)

    clf = LocalOutlierFactor(contamination=contamination)
    y_pred = clf.fit_predict(df)

    y_pred[y_pred==1]  = 0
    y_pred[y_pred==-1] = 1

    # df['LOF'] = y_pred
    # ax = df[df['LOF']==1][0].plot(style='.')
    # df[df['LOF']==-1][0].plot(style='.',ax=ax)

    return y_pred

def robustcovariance(nparray,contamination):

    """
    The scikit-learn provides an object covariance.EllipticEnvelope that fits a
    robust covariance estimate to the data, and thus fits an ellipse to the central
    data points, ignoring points outside the central mode.

    References:
    Rousseeuw, P.J., Van Driessen, K. “A fast algorithm for the minimum covariance determinant estimator”.
    Technometrics 41(3), 212 (1999)
    """

    df = pd.DataFrame(nparray)

    # Fit the model
    clf = EllipticEnvelope(contamination=contamination)
    clf.fit(df)
    y_pred = clf.predict(df)

    y_pred[y_pred==1]  = 0
    y_pred[y_pred==-1] = 1

    # df['RC'] = y_pred
    # ax = df[df['RC']==1][0].plot(style='.')
    # df[df['RC']==-1][0].plot(style='.',ax=ax)

    return y_pred

def seas_dec(ts_array,idx):

    """
    Decomposed time series
    """

    df = pd.DataFrame(ts_array,index=idx)
    result = seasonal_decompose(df, model='multiplicative')

    return np.squeeze(result.resid.fillna(method='ffill').fillna(method='bfill').values)

if __name__ == '__main__':
    main(sys.argv[1:])
