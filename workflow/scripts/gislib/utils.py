# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from os.path import join, dirname, isfile, basename
import os
import glob
from shapely.geometry import Point
# import gdal
from osgeo import gdal
import numpy as np
import xarray as xr
import rasterio
import dask
from rasterio.transform import from_bounds
from rasterio.merge import merge as merge_tool


# local libraries
from pyflwdir.core import _mv
from .rio import open_raster

def xy_to_subidx(ds_lr, ds):
    """translate global x,y coordinates to local linear subidx"""
    xmin, ymin, xmax, ymax = ds.rio.bounds
    if xmin < -180:
        xs = ds_lr['outlon'].fillna(-np.inf).values
        xs = np.where(xs>0, xs-360, xs)
    elif xmax > 180:
        xs = ds_lr['outlon'].fillna(np.inf).values
        xs = np.where(xs<0, xs+360, xs)
    else:
        xs = ds_lr['outlon'].fillna(-np.inf).values
    ys = ds_lr['outlat'].fillna(-np.inf).values
    outidx = ds.rio.xy_to_idx(xs=xs, ys=ys, nodata=_mv, mask_outside=True)
    dims = ds_lr.rio.dims
    attrs = dict(_FillValue=_mv)
    da_outidx = xr.DataArray(data=outidx, dims=dims, coords=ds_lr.coords, attrs=attrs)
    return da_outidx

def subidx_to_xy(ds_lr, ds):
    """translate local linear subidx to global x,y coordinates"""
    outidx = ds_lr['outidx'].values 
    outlon, outlat = ds.rio.idx_to_xy(outidx, mask_outside=True, nodata=np.nan)
    xmin, _, xmax, _ = ds.rio.bounds
    if xmin < -180:
        outlon = np.where(outlon>0, outlon-360, outlon)
    elif xmax > 180:
        outlon = np.where(outlon<0, outlon+360, outlon)
    dims = ds_lr.rio.dims
    attrs = dict(_FillValue=np.nan)
    da_x = xr.DataArray(data=outlon, dims=dims, coords=ds_lr.coords, attrs=attrs)
    da_y = xr.DataArray(data=outlat, dims=dims, coords=ds_lr.coords, attrs=attrs)
    return da_x, da_y

def clip_bbox_global(ds, bbox, buffer, align=None):
    """clip from global maps with extents passed -180W / 180E"""
    bbox_glob = np.asarray(ds.rio.bounds).round()
    w, s, e, n = bbox

    # no min/max for east/west, we re-arrange the axis at e/w180 bounds
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        wbuf, ebuf = w - buffer, e + buffer
        sbuf = np.maximum(bbox_glob[1], s - buffer)
        nbuf = np.minimum(bbox_glob[3], n + buffer)
        bbox_buf = wbuf, sbuf, ebuf, nbuf
        if ebuf > 180 or wbuf < -180:
            lon_org = np.copy(ds["x"].values)
            if ebuf > 180:
                lon_new = np.where(lon_org < 0, lon_org + 360, lon_org)
            else:
                lon_new = np.where(lon_org > 0, lon_org - 360, lon_org)
            ds["x"] = xr.Variable("x", lon_new)
            ds = ds.sortby("x")
        ds_clip = ds.rio.clip_bbox(bbox_buf, align=align)
    return ds_clip

def gdal_buildvrt(fn_vrt, fns, **kwargs):
    """gdal vrt wrapper"""
    vrt_options = gdal.BuildVRTOptions(**kwargs)
    if isfile(fn_vrt):
        os.remove(fn_vrt)
    vrt = gdal.BuildVRT(fn_vrt, fns, options=vrt_options)
    if vrt is None:
        raise Exception('Creating vrt not successfull, check input files.')
    vrt = None # to write to disk

def vrt_props(fn_vrt):
    with rasterio.open(fn_vrt, 'r') as src:
        bbox = np.asarray(src.bounds).round()
    fns = []
    with open(fn_vrt) as f:
        for line in f.readlines():
            if "<SourceFilename" in line:
                fns.append(basename(line.split('>')[1].split('<')[0]))
    return fns, bbox
    
    

def build_vrt(root, name, nodata=None, bbox=None, res=None, fn_vrt=None):
    """Create vrt for all raster tiles in folder. 
    Force bounding box and resolution"""
    if fn_vrt is None:
        fn_vrt = join(root, f'{name}.vrt')
    fns = glob.glob(join(root, name, f'*.tif'))
    if nodata is None:
        with rasterio.open(fns[0], 'r') as src:
            nodata = src.nodata
    gdal_buildvrt(fn_vrt, fns, srcNodata=nodata, VRTNodata=nodata)
    # force bounding box and resolution
    if bbox is not None and res is not None:
        w, _, _, n = bbox
        new_line = f'  <GeoTransform> {w:+.16e},  {res:.16e},  {0.:.16e},  {n:+.16e},  {0:.16e}, {-res:+.16e}</GeoTransform>\n'
        vrt = []
        with open(fn_vrt, 'r') as f:
            for line in f.readlines():
                if line.strip().startswith('<GeoTransform>'):
                    line = new_line
                vrt.append(line)
        with open(fn_vrt, 'w') as f:
            f.writelines(vrt)

def pandas2geopandas(df, x_col='lon', y_col='lat', crs='EPSG:4326', drop_xy=False):
    geoms = [Point(x, y) for x, y in zip(df[x_col], df[y_col])]
    if drop_xy:
        df = df.drop([x_col, y_col], axis=1)
    return gp.GeoDataFrame(df, geometry=geoms, crs=crs)


def merge(fns, fn_out, bounds, res, method='first', **kwargs):
    """rasterio merge overlapping tiles"""
    
    datasets = [rasterio.open(f) for f in fns]
    nodata = datasets[0].nodata
    dtype = datasets[0].profile['dtype']

    dest, output_transform = merge_tool(datasets, bounds=bounds, res=res,
                                        nodata=nodata, precision=10,
                                        method=method)
    
    profile = dict()
    profile['dtype'] = dtype
    profile['driver'] = kwargs.get('driver', 'GTIFF')
    profile['compress'] = kwargs.get('compress', 'lzw')
    profile['transform'] = output_transform
    profile['height'] = dest.shape[1]
    profile['width'] = dest.shape[2]
    profile['count'] = dest.shape[0]
    profile['nodata'] = nodata
    profile.update(**kwargs)

    with rasterio.open(fn_out, 'w', **profile) as dst:
        dst.write(dest)

    [src.close() for src in datasets]

def merge1(fns, fn_out, bounds, res, method='first', **kwargs):
    """rasterio merge overlapping tiles"""


    if method == 'first':
        def copyto(old_data, new_data, old_nodata, new_nodata):
            mask = np.logical_and(old_nodata, ~new_nodata)
            old_data[mask] = new_data[mask]

    elif method == 'last':
        def copyto(old_data, new_data, old_nodata, new_nodata):
            mask = ~new_nodata
            old_data[mask] = new_data[mask]

    elif method == 'min':
        def copyto(old_data, new_data, old_nodata, new_nodata):
            mask = np.logical_and(~old_nodata, ~new_nodata)
            old_data[mask] = np.minimum(old_data[mask], new_data[mask])

            mask = np.logical_and(old_nodata, ~new_nodata)
            old_data[mask] = new_data[mask]

    elif method == 'max':
        def copyto(old_data, new_data, old_nodata, new_nodata):
            mask = np.logical_and(~old_nodata, ~new_nodata)
            old_data[mask] = np.maximum(old_data[mask], new_data[mask])

            mask = np.logical_and(old_nodata, ~new_nodata)
            old_data[mask] = new_data[mask]

    elif method == 'new':
        def copyto(old_data, new_data, old_nodata, new_nodata):
            data_overlap = np.logical_and(~old_nodata, ~new_nodata)
            assert not np.any(data_overlap)
            mask = ~new_nodata
            old_data[mask] = new_data[mask]

    elif callable(method):
        copyto = method

    first = True
    for _, fn in enumerate(fns):
        ds = open_raster(fn).rio.clip_bbox(bounds)

        if np.any([ds[dim].size==0 for dim in ds.rio.dims]):
            continue

        if first:
            nodata = ds.rio.nodata
            isnan = np.isnan(nodata)
            dtype = ds.dtype
            # print(dtype, nodata)
            x_dim = ds.rio.x_dim
            y_dim = ds.rio.y_dim
            # create global index
            w, s, e, n = bounds
            ys = np.arange(n-res/2., s, -res)
            xs = np.arange(w+res/2., e, res)
            index = {x_dim: xs, y_dim: ys}
        else:
            # try:
            w0, s0, e0, n0 = ds.rio.bounds
            top = np.where(ys<=n0)[0][0] if n0 < ys[0] else 0
            bottom = np.where(ys<s0)[0][0] if s0 > ys[-1] else None
            left = np.where(xs>=w0)[0][0] if w0 > xs[0] else 0  
            right = np.where(xs>e0)[0][0] if e0 < xs[-1] else None
            y_slice = slice(top, bottom)
            x_slice = slice(left, right)
            index = {x_dim: xs[x_slice], y_dim: ys[y_slice]}
            assert (index[y_dim].size, index[x_dim].size) == ds.rio.shape
            # except:
            #     import pdb; pdb.set_trace()
        try:
            ds = ds.reindex(**index, method='nearest', tolerance=res/4)
            if not isnan:
                ds = ds.fillna(nodata)
        except IndexError:
            continue
        
        if first:
            # create output dataset
            dims=ds.dims
            coords=ds.coords
            dest=ds.values.astype(dtype)
            first = False
        else:
            region = dest[y_slice, x_slice]
            temp = np.ma.masked_equal(ds.values.astype(dtype), nodata)
            if isnan:
                region_nodata = np.isnan(region)
                temp_nodata = np.isnan(temp)
            else:
                region_nodata = region == nodata
                temp_nodata = temp.mask
            copyto(region, temp, region_nodata, temp_nodata)
        ds.close()

    ds_out = xr.DataArray(
        dims=dims, 
        coords=coords, 
        data=dest, 
        attrs=dict(_FillValue=nodata)
    )
    if 'dtype' in kwargs:
        ds_out = ds_out.astype(kwargs['dtype'])
        ds_out.attrs.update(_FillValue=nodata)
    else:
        kwargs.update(dtype=dtype)
    ds_out.rio.to_raster(fn_out, compress='lzw', **kwargs)
    ds_out.close()
