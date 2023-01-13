# -- coding: utf-8 --
# Credits: This script is based on the rioxarray package (Apache License, Version 2.0)
# source file: https://github.com/corteva/rioxarray/blob/master/rioxarray/rioxarray.py#6452571
# license file: https://github.com/corteva/rioxarray/blob/master/LICENSE#8de6d32
"""
This module is an extension for xarray to provide rasterio capabilities
to xarray datasets/dataarrays.
"""

import copy
import os
import glob
from os.path import join, basename, dirname, isdir
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
from shapely.geometry import box
import geopandas as gp
import xarray as xr
import dask
from affine import Affine
from rasterio.crs import CRS
import rasterio.warp
from rasterio import features
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from scipy.interpolate import griddata
from scipy import ndimage
import tempfile
import pyproj
import logging

logger = logging.getLogger(__name__)

from pyflwdir import gis_utils, regions

# global variables
GDAL_DRIVER_CODE_MAP = {
    "asc": "AAIGrid",
    "blx": "BLX",
    "bmp": "BMP",
    "bt": "BT",
    "dat": "ZMap",
    "dem": "USGSDEM",
    "gen": "ADRG",
    "gif": "GIF",
    "gpkg": "GPKG",
    "grd": "NWT_GRD",
    "gsb": "NTv2",
    "gtx": "GTX",
    "hdr": "MFF",
    "hf2": "HF2",
    "hgt": "SRTMHGT",
    "img": "HFA",
    "jpg": "JPEG",
    "kro": "KRO",
    "lcp": "LCP",
    "map": "PCRaster",
    "mbtiles": "MBTiles",
    "mpr/mpl": "ILWIS",
    "ntf": "NITF",
    "pix": "PCIDSK",
    "png": "PNG",
    "pnm": "PNM",
    "rda": "R",
    "rgb": "SGI",
    "rst": "RST",
    "rsw": "RMF",
    "sdat": "SAGA",
    "sqlite": "Rasterlite",
    "ter": "Terragen",
    "tif": "GTiff",
    "vrt": "VRT",
    "xpm": "XPM",
    "xyz": "XYZ",
}
GDAL_EXT_CODE_MAP = {v: k for k, v in GDAL_DRIVER_CODE_MAP.items()}

XDIMS = ("x", "longitude", "lon", "long")
YDIMS = ("y", "latitude", "lat")
FILL_VALUE_NAMES = ("_FillValue", "missing_value", "fill_value", "nodata", "nodatavals")
UNWANTED_RIO_ATTRS = (
    "nodatavals",
    "crs",
    "is_tiled",
    "res",
    "transform",
    "scales",
    "offsets",
)
DEFAULT_GRID_MAP = "spatial_ref"

__all__ = [
    "RasterArray",
    "RasterDataset",
    "open_raster",
    "open_mfraster",
]


def open_raster(
    filename, mask_nodata=False, chunks=None, cache=None, lock=None, logger=logger
):
    """Open a file with rasterio based on xarray.open_rasterio, but parse to 
    hydromt.rio format.
    
    Parameters
    ----------
    filename : str, rasterio.DatasetReader, or rasterio.WarpedVRT
        Path to the file to open. Or already open rasterio dataset.
    mask_nodata : bool, optional
        set nodata values to np.nan (xarray default nodata value)
    chunks : int, tuple or dict, optional
        Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
        ``{'x': 5, 'y': 5}``. If chunks is provided, it used to load the new
        DataArray into a dask array.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False.
    lock : False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a global lock is
        used to avoid issues with concurrent access to the same file when using
        dask's multithreaded backend.
    
    Returns
    -------
    data : DataArray
        The newly created DataArray.
    """
    da = xr.open_rasterio(filename, chunks=chunks, cache=cache, lock=lock)
    da = da.squeeze().reset_coords(drop=True)  # drop band dimension if single layer
    # additional hydromt.rio parsing
    da.rio.set_nodata()  # parse nodatavals attr to default _FillValue
    if da.rio.nodata is None:
        logger.warning(f"nodata value missing for {filename}")
    da.rio.set_crs()  # parse crs information
    for k in UNWANTED_RIO_ATTRS:
        da.attrs.pop(k, None)
    if mask_nodata:
        da = da.rio.mask_nodata()
    return da


def open_mfraster(
    paths, chunks=None, concat=False, concat_dim="time", logger=logger, **kwargs
):
    """
    Returns a xarray DataSet based on multiple rasterio/gdal files.
    Names are infered from the filename.

    Additonal key-word arguments are passed to the open_raster function.

    Parameters
    ----------
    paths: str, list
        Paths to the rasterio/gdal files.
    chunks : int, tuple or dict, optional
        Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
        ``{'x': 5, 'y': 5}``. If chunks is provided, it used to load the new
        DataArray into a dask array.

    Returns
    -------
    data : DataSet
        The newly created DataSet.
    """
    prefix, postfix, _name = "", "", ""
    if isinstance(paths, str):
        if "*" in paths:
            prefix, postfix = basename(paths).split(".")[0].split("*")
        paths = [fn for fn in glob.glob(paths) if not fn.endswith(".xml")]
    else:
        paths = [str(p) if isinstance(p, Path) else p for p in paths]

    if len(paths) == 0:
        raise OSError("no files to open")

    da_lst, index_lst, fn_attrs = [], [], []
    for i, fn in enumerate(paths):
        da = open_raster(fn, chunks=chunks, **kwargs)
        bname = basename(fn)
        _name = bname.split(".")[0].strip(postfix).split("_")[0]
        if concat:
            da.name = prefix if prefix != "" else _name
            # index based on postfix behind "_"
            if "_" in bname and bname.split(".")[0].split("_")[1].isdigit():
                index = int(bname.split(".")[0].split("_")[1])
            # index based on file extension (PCRaster style)
            elif "." in bname and bname.split(".")[1].isdigit():
                index = int(bname.split(".")[1])
            # index based on postfix directly after prefix
            elif prefix != "" and bname.split(".")[0].strip(prefix).isdigit():
                index = int(bname.split(".")[0].strip(prefix))
            else:
                raise ValueError(f"No index for {concat_dim} infered from {bname}")
            index_lst.append(index)
            fn_attrs.append(bname)
        else:
            da.name = bname.split(".")[0].replace(prefix, "").replace(postfix, "")
            da.attrs.update(source_file=bname)
        # align coordinates if rounded(!) transform and shape are equal
        if i > 0:
            if not (
                np.all(np.array(da.rio.transform).round(6) == _transform)
                and np.all(da.shape == _shape)
                and np.logical_and(_xdim == da.rio.x_dim, _ydim == da.rio.y_dim)
            ):
                raise xr.MergeError(f"Geotransform and/or shape do not match")
        else:
            da.name = _name if da.name == "" else da.name
            _transform = np.array(da.rio.transform).round(6)
            _shape = da.shape
            _xdim = da.rio.x_dim
            _ydim = da.rio.y_dim
            _xcoords = da[_xdim]
            _ycoords = da[_ydim]
        da[_xdim] = _xcoords
        da[_ydim] = _ycoords
        da_lst.append(da)
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        if concat:
            da = xr.concat(da_lst, dim=concat_dim)
            da.coords[concat_dim] = xr.IndexVariable(concat_dim, index_lst)
            da = da.sortby(concat_dim)
            da.attrs.update(da_lst[0].attrs)
            ds = da.to_dataset()  # dataset for consistency
            ds.attrs.update(source_files="; ".join(fn_attrs))
        else:
            ds = xr.merge(da_lst)
    return ds


class XRasterBase(object):
    """This is the base class for the GIS extensions for xarray"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        # create new coordinate with attributes in which to save x_dim, y_dim and crs.
        # other spatial properties are always calculated on the fly to ensure consistency with data
        if DEFAULT_GRID_MAP not in self._obj.coords:
            self._obj.coords[DEFAULT_GRID_MAP] = xr.Variable((), 1)
        self._attrs = self._obj.coords[DEFAULT_GRID_MAP].attrs

    @property
    def attrs(self):
        """Dictionary with spatial attributes
        """
        return self._attrs

    @property
    def dims(self):
        if "y_dim" not in self._attrs:
            self.set_spatial_dims()
        return self._attrs["y_dim"], self._attrs["x_dim"]

    @property
    def coords(self):
        return {d: self._obj.coords[d] for d in list(self.dims) + [DEFAULT_GRID_MAP]}

    @property
    def crs(self):
        """Retrieve projection in wkt format
        """
        if self._attrs.get("crs_wkt", None) is None:
            self.set_crs()
        crs = self._attrs.get("crs_wkt", None)
        return CRS.from_wkt(crs) if crs is not None else crs

    @property
    def x_dim(self):
        if "x_dim" not in self._attrs:
            self.set_spatial_dims()
        return self._attrs["x_dim"]

    @property
    def y_dim(self):
        if "y_dim" not in self._attrs:
            self.set_spatial_dims()
        return self._attrs["y_dim"]

    @property
    def xcoords(self):
        """Returns the x coordinates"""
        return self._obj[self.x_dim]

    @property
    def ycoords(self):
        """Returns the y coordinates"""
        return self._obj[self.y_dim]

    @property
    def width(self):
        """Returns the width of the dataset (x dimension size)"""
        return self.xcoords.size

    @property
    def height(self):
        """Returns the height of the dataset (y dimension size)"""
        return self.ycoords.size

    @property
    def transform(self):
        """Return the affine transform of the raster"""
        transform = gis_utils.transform_from_bounds(
            *self.internal_bounds, self.width, self.height
        )
        return transform

    @property
    def internal_bounds(self):
        """Determine the internal bounds of the `xarray.DataArray`"""
        res_x, res_y = self.res
        left = float(self.xcoords[0]) - res_x / 2.0
        right = float(self.xcoords[-1]) + res_x / 2.0
        top = float(self.ycoords[0]) - res_y / 2.0
        bottom = float(self.ycoords[-1]) + res_y / 2.0
        return left, bottom, right, top

    @property
    def bounds(self):
        """Determine the bounds (xmin, ymin, xmax, ymax) of the `xarray.DataArray`"""
        left, bottom, right, top = self.internal_bounds
        xmin, xmax = min(left, right), max(left, right)
        ymin, ymax = min(top, bottom), max(top, bottom)
        return xmin, ymin, xmax, ymax

    @property
    def res(self):
        """return (x_resolution, y_resolution) Tuple."""
        left, right = float(self.xcoords[0]), float(self.xcoords[-1])
        top, bottom = float(self.ycoords[0]), float(self.ycoords[-1])
        res_x = (right - left) / max(self.width - 1, 1)
        res_y = (bottom - top) / max(self.height - 1, 1)
        return res_x, res_y

    @property
    def shape(self):
        """Returns shape spatial dimension (height, width)."""
        self._check_dimensions()
        return self.ycoords.size, self.xcoords.size

    @property
    def size(self):
        """Returns size."""
        return np.multiply(*self.shape)

    def _check_dimensions(self):
        """
        This function validates that the dimensions 2D/3D and
        they are are in the proper order.

        Returns
        -------
        str or None: Name extra dimension.
        """
        extra_dims = list(set(list(self._obj.dims)) - set([self.x_dim, self.y_dim]))
        if len(extra_dims) > 1:
            raise ValueError("Only 2D and 3D data arrays supported.")
        elif extra_dims and self._obj.dims != (extra_dims[0], self.y_dim, self.x_dim):
            raise ValueError(
                "Invalid dimension order. Expected order: {0}. "
                "You can use `DataArray.transpose{0}`"
                " to reorder your dimensions.".format(
                    (extra_dims[0], self.y_dim, self.x_dim)
                )
            )
        elif not extra_dims and self._obj.dims != (self.y_dim, self.x_dim):
            raise ValueError(
                "Invalid dimension order. Expected order: {0}"
                "You can use `DataArray.transpose{0}` "
                "to reorder your dimensions.".format((self.y_dim, self.x_dim))
            )
        return extra_dims[0] if extra_dims else None

    def set_spatial_dims(self, x_dim=None, y_dim=None):
        """
        This sets the spatial dimensions of the dataset.

        Arguments
        ----------
        x_dim: str, optional
            The name of the x dimension.
        y_dim: str, optional
            The name of the y dimension.
        """
        if x_dim is None:
            for dim in XDIMS:
                if dim in self._obj.dims:
                    x_dim = dim
                    break
        if x_dim and x_dim in self._obj.dims:
            self._attrs.update(x_dim=x_dim)
        else:
            raise ValueError(
                "x dimension not found. Use 'set_spatial_dims'"
                + " functions with correct x_dim argument provided."
            )

        if y_dim is None:
            for dim in YDIMS:
                if dim in self._obj.dims:
                    y_dim = dim
                    break
        if y_dim and y_dim in self._obj.dims:
            self._attrs.update(y_dim=dim)
        else:
            raise ValueError(
                "y dimension not found. Use 'set_spatial_dims'"
                + " functions with correct y_dim argument provided."
            )
        check_x = np.all(np.isclose(np.diff(np.diff(self._obj[x_dim])), 0, atol=1e-4))
        check_y = np.all(np.isclose(np.diff(np.diff(self._obj[y_dim])), 0, atol=1e-4))
        if check_x == False or check_y == False:
            raise ValueError("Rio only applies to regular grids")

    def transform_bounds(self, dst_crs, densify_pts=21):
        """Transform bounds from src_crs to dst_crs.

        Optionally densifying the edges (to account for nonlinear transformations
        along these edges) and extracting the outermost bounds.

        Note: this does not account for the antimeridian.

        Arguments
        ----------
        dst_crs: str, :obj:`rasterio.crs.CRS`, or dict
            Target coordinate reference system.
        densify_pts: uint, optional
            Number of points to add to each edge to account for nonlinear
            edges produced by the transform process.  Large numbers will produce
            worse performance.  Default: 21 (gdal default).

        Returns
        -------
        left, bottom, right, top: float
            Outermost coordinates in target coordinate reference system.
        """
        return rasterio.warp.transform_bounds(
            self.crs, dst_crs, *self.internal_bounds, densify_pts=densify_pts
        )

    def set_crs(self, input_crs=None):
        """
        Set the CRS value.

        Arguments
        ----------
        input_crs: object
            Anything accepted by `rasterio.crs.CRS.from_user_input`.
            If no input_crs is given it will tried to be found in its 
            'crs' attribute or the default grid_map variable
        grid_mapping_name: str
            Name of grid mapping variable

        Returns
        -------
        xarray.Dataset or xarray.DataArray:
        Dataset with crs attribute.

        """
        crs_names = ["crs_wkt", "crs", "epsg"]
        # user defined
        if input_crs is not None:
            input_crs = CRS.from_user_input(input_crs).wkt
        # look in grid_mapping and data variable attributes
        else:
            for name in crs_names:
                crs = self._obj.coords[DEFAULT_GRID_MAP].attrs.get(name, None)
                if crs is None:
                    crs = self._obj.attrs.pop(name, None)
                if crs is None and hasattr(self, "vars"):  # dataset
                    crs = self._obj[self.vars[0]].attrs.pop(name, None)
                if crs is not None:
                    # avoid Warning 1: +init=epsg:XXXX syntax is deprecated
                    crs = crs.strip("+init=") if isinstance(crs, str) else crs
                    try:
                        input_crs = CRS.from_user_input(crs).wkt
                    except:
                        pass
        if input_crs is not None:
            self._attrs.update(crs_wkt=input_crs)

    def idx_to_xy(self, idx, mask=None, mask_outside=False, nodata=np.nan):
        """Return x,y coordinates at flat index
        
        Arguments
        ----------
        idx : ndarray of int
            flat index
        mask : ndarray of bool, optional
            data mask of valid values, by default None
        mask_outside : boolean, optional
            mask xy points outside domain (i.e. set nodata), by default False
        nodata : int, optional
            nodata value, used for output array, by default np.nan
        
        Returns
        -------
        Tuple of ndarray of float 
            x, y coordinates
        """
        idx = np.atleast_1d(idx)
        nrow, ncol = self.shape
        r, c = idx // ncol, idx % ncol
        points_inside = np.logical_and.reduce((r >= 0, r < nrow, c >= 0, c < ncol))
        if mask is None:
            mask = np.ones(idx.shape, dtype=np.bool)  # all valid
        if mask_outside:
            mask[points_inside == False] = False
        elif np.any(points_inside[mask] == False):
            raise ValueError("Linear indices outside domain.")
        y = np.full(idx.shape, nodata, dtype=np.float64)
        x = np.full(idx.shape, nodata, dtype=np.float64)
        y[mask] = self.ycoords.values[r[mask]]
        x[mask] = self.xcoords.values[c[mask]]
        return x, y

    def xy_to_idx(self, xs, ys, mask=None, mask_outside=False, nodata=-1):
        """Return flat index of x, y coordinates
        
        Arguments
        ----------
        xs, ys: ndarray of float
            x, y coordinates
        mask : ndarray of bool, optional
            data mask of valid values, by default None
        mask_outside : boolean, optional
            mask xy points outside domain (i.e. set nodata), by default False
        nodata : int, optional
            nodata value, used for output array, by default -1

        Returns
        -------
        ndarray of int
            flat indices
        """
        nrow, ncol = self.shape
        xs = np.atleast_1d(xs)
        ys = np.atleast_1d(ys)
        if mask is None:
            mask = np.logical_and(np.isfinite(xs), np.isfinite(ys))
        r = np.full(mask.shape, -1, dtype=np.intp)
        c = np.full(mask.shape, -1, dtype=np.intp)
        r[mask], c[mask] = rasterio.transform.rowcol(self.transform, xs[mask], ys[mask])
        points_inside = np.logical_and.reduce((r >= 0, r < nrow, c >= 0, c < ncol))
        if mask_outside:
            mask[points_inside == False] = False
        elif np.any(points_inside[mask] == False):
            raise ValueError("Coordinates outside domain.")
        idx = np.full(xs.shape, nodata, dtype=np.intp)
        idx[mask] = r[mask] * ncol + c[mask]
        return idx

    def clip_bbox(self, bbox, align=None, buffer=0):
        """Clip the array based on a bounding box.

        Arguments
        ----------
        bbox : tuple of floats
            (xmin, ymin, xmax, ymax) bounding box
        align : boolean, optional
            aligns target to
        buffer : int, optional
            Buffer around the bounding box expressed in resolution multiplicity, 
            by default 0
        
        Returns
        -------
        xarray.DataSet or DataArray
            Data clipped to bbox
        """
        w, s, e, n = bbox
        left, bottom, right, top = self.internal_bounds
        xres, yres = [np.abs(res) for res in self.res]
        if align is not None:
            align = abs(align)
            # align to grid
            w = (w // align) * align
            s = (s // align) * align
            e = (e // align + 1) * align
            n = (n // align + 1) * align
        if top > bottom:
            n = min(top, n + buffer * yres)
            s = max(bottom, s - buffer * yres)
            y_slice = slice(n, s)
        else:
            n = min(bottom, n + buffer * yres)
            s = max(top, s - buffer * yres)
            y_slice = slice(s, n)
        if left > right:
            e = min(left, e + buffer * xres)
            w = max(right, w - buffer * xres)
            x_slice = slice(e, w)
        else:
            e = min(right, e + buffer * xres)
            w = max(left, w - buffer * xres)
            x_slice = slice(w, e)
        return self._obj.sel({self.x_dim: x_slice, self.y_dim: y_slice})

    def clip_mask(self, mask):
        """Clip dataset to region with mask values greater than zero.
        
        Parameters
        ----------
        mask : xarray.DataArray
            Mask array.
        
        Returns
        -------
        xarray.DataSet or DataArray
            Data clipped to mask
        """
        if not isinstance(mask, xr.DataArray):
            raise ValueError("Mask should be xarray.DataArray type.")
        if not mask.rio.shape == self.shape:
            raise ValueError("Mask shape invalid.")
        mask_bin = (mask.values != 0).astype(np.uint8)
        if not np.any(mask_bin):
            raise ValueError("Invalid mask.")
        row_slice, col_slice = ndimage.find_objects(mask_bin)[0]
        self._obj.coords["mask"] = xr.Variable(self.dims, mask_bin)
        return self._obj.isel({self.x_dim: col_slice, self.y_dim: row_slice})

    def clip_geom(self, geom, align=None, buffer=0):
        """Returns clipped an array clipped to the bounding box of the geometry. 
        A rasterized version of the geometry is set to the 'mask' coordinate.
        
        Parameters
        ----------
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        align : float, optional
            Resolution to align the bounding box, by default None
        buffer : int, optional
            Buffer around the bounding box expressed in resolution multiplicity, 
            by default 0

        Returns
        -------
        xarray.DataSet or DataArray
            Data clipped to geometry
        """
        if hasattr(geom, "total_bounds"):
            bbox = geom.total_bounds
        else:
            raise ValueError("geom should be geopandas GeoDataFrame object.")
        ds_clip = self.clip_bbox(bbox, align=align, buffer=buffer)
        ds_clip.coords["mask"] = ds_clip.rio.geometry_mask(geom)
        return ds_clip

    def rasterize(
        self,
        gdf,
        col_name="index",
        nodata=0,
        all_touched=False,
        dtype=None,
        sindex=False,
        **kwargs,
    ):
        """Return an image array with input geometries burned in.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame of shapes and values to burn.
        col_name : str, optional
            GeoDataFrame column name to use for burning, by default 'index'
        nodata : int or float, optional
            Used as fill value for all areas not covered by input geometries, by default 0.
        all_touched : bool, optional
            If True, all pixels touched by geometries will be burned in. If false, only 
            pixels whose center is within the polygon or that are selected by 
            Bresenham’s line algorithm will be burned in.
        dtype : numpy dtype, optional
            Used as data type for results, by default it is derived from values.
        sindex : bool, optional
            Create a spatial index to select overlapping geometries before rasterizing,
            by default False.
        
        Returns
        -------
        xarray.DataArray
            DataArray with burned geometries
        
        Raises
        ------
        ValueError
            If no geometries are found inside the bounding box.
        """
        if sindex:
            idx = list(gdf.sindex.intersection(self.bounds))
            gdf = gdf.iloc[idx, :]

        if len(gdf.index) > 0:
            geoms = gdf.geometry.values
            values = gdf.reset_index()[col_name].values
            dtype = values.dtype if dtype is None else dtype
            if dtype == np.int64:
                dtype = np.int32  # max integer accuracy accepted
            shapes = list(zip(geoms, values))
            raster = np.full(self.shape, nodata, dtype=dtype)
            features.rasterize(
                shapes,
                out_shape=self.shape,
                fill=nodata,
                transform=self.transform,
                out=raster,
                all_touched=all_touched,
                **kwargs,
            )
        else:
            raise ValueError("No shapes found within raster bounding box")

        attrs = self._obj.attrs.copy()
        attrs.update(_FillValue=nodata)
        da_out = xr.DataArray(
            name=col_name, dims=self.dims, coords=self.coords, data=raster, attrs=attrs
        )
        return da_out

    def geometry_mask(self, gdf, all_touched=False, invert=False, **kwargs):
        """Create a mask from shapes. This adds a "mask" variable to coordinates.
        By default the mask is False where pixels overlap shapes. 
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame of shapes and values to burn.
        all_touched : bool, optional
            If True, all pixels touched by geometries will masked. If false, only 
            pixels whose center is within the polygon or that are selected by 
            Bresenham’s line algorithm will be burned in. By default False.
        invert : bool, optional
            If True, the mask will be False for pixels that overlap shapes, 
            by default False
        """
        gdf1 = gdf.copy()
        gdf1["mask"] = not invert
        gdf1["mask"] = gdf1["mask"].astype(np.uint8)
        return self.rasterize(
            gdf1, col_name="mask", all_touched=all_touched, nodata=invert, **kwargs
        ).astype(np.bool)

    def vector_grid(self):
        """Create vector of grid cells. Returns geopandas GeoDataFrame.
        """
        w, _, _, n = self.bounds
        dx, dy = self.res
        nrow, ncol = self.shape
        cells = []
        for i in range(nrow):
            top = n + i * dy
            bottom = n + (i + 1) * dy
            for j in range(ncol):
                left = w + j * dx
                right = w + (j + 1) * dx
                cells.append(box(left, bottom, right, top))
        return gp.GeoDataFrame(geometry=cells, crs=self.crs)


@xr.register_dataarray_accessor("rio")
class RasterArray(XRasterBase):
    """This is the GIS extension for xarray.DataArray"""

    def __init__(self, xarray_obj):
        super(RasterArray, self).__init__(xarray_obj)

    @staticmethod
    def from_numpy(data, transform, nodata=None, attrs=None, crs=None):
        """
        Transforms a 2D/3D numpy array into a RasterArray. 
        The data dimensions should have the y and x on the second last and last dimensions.

        Parameters
        ----------
        data : numpy.array, 2-dimensional
            values to parse into DataArray
        transform : affine transform
            Two dimensional affine transform for 2D linear mapping
        nodata : float or int, optional
            nodata value
        attrs : dict, optional
            additional attributes
        crs : str,
            coordinate reference system in WKT

        Returns
        -------
        da : RasterArray
            xarray.DataArray with geospatial information
        """
        if len(data.shape) == 2:
            nrow, ncol = data.shape
            dims = ("y", "x")
        elif len(data.shape) == 3:
            _, nrow, ncol = data.shape
            dims = ("dim0", "y", "x")
        else:
            raise ValueError("Only 2D and 3D arrays supported")
        _xcoords, _ycoords = gis_utils.affine_to_coords(transform, (nrow, ncol))
        da = xr.DataArray(data, dims=dims, coords={"y": _ycoords, "x": _xcoords},)
        da.rio.set_nodata(nodata=nodata)  # parse nodatavals attr to default _FillValue
        if attrs is not None:
            da.attrs.update(attrs)
        if crs is not None:
            da.rio.set_crs(input_crs=crs)  # parse crs information
        return da

    @property
    def nodata(self):
        """Get the nodata value for the dataset."""
        if self._obj.attrs.get("_FillValue", None) is None:
            self.set_nodata()
        return self._obj.attrs.get("_FillValue", None)

    def set_nodata(self, nodata=None):
        """Sets the nodata value in the attributes of to the DataArray in a CF 
        compliant manner.

        Arguments
        ----------
        nodata: float, integer
            Nodata value for the DataArray.
            If the nodata property and argument are both None, the _FillValue 
            attribute will be removed.
        """
        dtype = str(self._obj.dtype)
        if nodata is None and "_FillValue" in self._obj.encoding:
            nodata = self._obj.encoding["_FillValue"]
        elif nodata is None:
            for name in FILL_VALUE_NAMES:
                if name in self._obj.attrs:
                    nodata = self._obj.attrs.pop(
                        name
                    )  # remove from attrs and set property
                    if isinstance(nodata, tuple):
                        try:
                            nodata = getattr(np, dtype)(nodata[0])
                            break
                        except ValueError:
                            warnings.warn(f"Nodata value {nodata} beyond data range.")
        if nodata is not None:  # keep attribute property
            self._obj.attrs.update({"_FillValue": nodata})

    def mask_nodata(self):
        """Maks nodata values with np.nan. 
        Note this will change integer dtypes to float.
        """
        if self.nodata is not None and self.nodata != np.nan:
            self._obj = self._obj.where(self._obj != self.nodata)
            self.set_nodata(np.nan)
        return self._obj

    def reproject(
        self,
        dst_crs=None,
        dst_res=None,
        dst_transform=None,
        dst_width=None,
        dst_height=None,
        dst_nodata=None,
        method="nearest",
        align=False,
    ):
        """
        Reproject a DataArray. Powered by rasterio.warp.reproject.

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.

        Arguments
        ----------
        dst_crs: CRS or dict, optional
            Target coordinate reference system. Required if source and
            destination are ndarrays. Will be derived from target if it
            is a rasterio Band.
        dst_res: tuple (x resolution, y resolution) or float, optional
            Target resolution, in units of target coordinate reference
            system.
        dst_transform: affine.Affine(), optional
            Target affine transformation. Required if source and
            destination are ndarrays. Will be derived from target if it is
            a rasterio Band.
        dst_width, dst_height: int, optional
            Output file size in pixels and lines. Cannot be used together
            with resolution (dst_res).
        dst_nodata: int or float, optional
            The nodata value used to initialize the destination; it will
            remain in all areas not covered by the reprojected source.
            Defaults to the nodata value of the destination image (if set),
            the value of src_nodata, or 0 (GDAL default).
        method: str, optional
            See rasterio.warp.reproject for existing methods, by default nearest.
        align: boolean, optional
            If True, align target transform to resolution

        Returns
        -------
        da_reproject : xarray.DataArray
            A reprojected DataArray.
        """
        try:
            resampling = getattr(Resampling, method.lower())
        except AttributeError:
            raise ValueError(f"Resampling method unknown: {method}.")
        if self.crs is None:
            raise ValueError("CRS is missing. Use set_crs function to resolve.")
        # set crs if missing
        dst_crs = dst_crs if dst_crs is not None else self.crs
        # calculate missing transform (and shape if dst_res is given)
        if dst_transform is None:
            (
                dst_transform,
                dst_width,
                dst_height,
            ) = rasterio.warp.calculate_default_transform(
                self.crs,
                dst_crs,
                self.width,
                self.height,
                *self.internal_bounds,
                resolution=dst_res,
                dst_width=dst_width,
                dst_height=dst_height,
            )
        if align:
            dst_transform, dst_width, dst_height = rasterio.warp.aligned_target(
                dst_transform, dst_width, dst_height, dst_res
            )
        # allocate output array and deal with non-spatial dimensions
        extra_dim = self._check_dimensions()
        xs, ys = gis_utils.affine_to_coords(dst_transform, (dst_height, dst_width))
        if extra_dim:
            extra_coords = self._obj[extra_dim]
            dst_dims = (extra_dim, self.y_dim, self.x_dim)
            dst_coords = {extra_dim: extra_coords, self.y_dim: ys, self.x_dim: xs}
            dst_shape = (extra_coords.size, dst_height, dst_width)
        else:
            dst_dims = (self.y_dim, self.x_dim)
            dst_coords = {self.y_dim: ys, self.x_dim: xs}
            dst_shape = (dst_height, dst_width)
        # create new DataArray for output
        da_reproject = xr.DataArray(
            name=self._obj.name,
            data=np.zeros(dst_shape, dtype=self._obj.dtype),
            coords=dst_coords,
            dims=dst_dims,
            attrs=self._obj.attrs,
        )
        # apply rasterio warp reproject
        rasterio.warp.reproject(
            source=self._obj.load().data,
            destination=da_reproject.data,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=self.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=self.nodata if dst_nodata is None else dst_nodata,
            resampling=resampling,
        )
        # make sure old spatial attributes are removed and add new spatial properties
        da_reproject.rio.set_crs(dst_crs)
        return da_reproject

    def reproject_like(self, da_like, method="nearest"):
        """
        Reproject a DataArray object to match the grid of another DataArray.

        Arguments
        ----------
        da_like : xarray.DataArray or Dataset
            DataArray of the target resolution and projection.
        method : str, optional
            See rasterio.warp.reproject for existing methods, by default nearest.

        Returns
        --------
        da_dst : xarray.DataArray
            Contains the data from the src_data_array, reprojected to match da_like.

        """
        if self.x_dim not in da_like.coords or self.y_dim not in da_like.coords:
            raise ValueError("x or y dimensions not found in da_like")
        da_dst = self.reproject(
            dst_crs=da_like.rio.crs,
            dst_transform=da_like.rio.transform,
            dst_width=da_like.rio.width,
            dst_height=da_like.rio.height,
            method=method,
        )
        # overwrite dimension to make sure these are identical!
        da_dst[self.x_dim] = da_like.rio.xcoords
        da_dst[self.y_dim] = da_like.rio.ycoords
        return da_dst

    def _interpolate_na(self, src_data, method="nearest"):
        """
        This method uses scipy.interpolate.griddata to interpolate missing data.

        Arguments
        ----------
        method: {‘linear’, ‘nearest’, ‘cubic’}, optional
            The method to use for interpolation in `scipy.interpolate.griddata`.

        Returns
        -------
        :class:`numpy.ndarray`: An interpolated :class:`numpy.ndarray`.

        """
        src_data_flat = np.copy(src_data).ravel()
        data_isnan = True if self.nodata is None else np.isnan(self.nodata)
        if not data_isnan:
            data_bool = src_data_flat != self.nodata
        else:
            data_bool = ~np.isnan(src_data_flat)
        if not data_bool.any() or data_bool.all():
            return src_data
        x_coords, y_coords = np.meshgrid(
            self._obj.coords[self.x_dim].values, self._obj.coords[self.y_dim].values
        )
        return griddata(
            points=(x_coords.ravel()[data_bool], y_coords.ravel()[data_bool]),
            values=src_data_flat[data_bool],
            xi=(x_coords, y_coords),
            method=method,
            fill_value=self.nodata,
        )

    def interpolate_na(self, method="nearest"):
        """
        This method uses scipy.interpolate.griddata to interpolate missing data.

        Arguments
        ----------
        method: {‘linear’, ‘nearest’, ‘cubic’}, optional
            The method to use for interpolation in `scipy.interpolate.griddata`.

        Returns
        -------
        :class:`xarray.DataArray`: An interpolated :class:`xarray.DataArray` object.

        """
        extra_dim = self._check_dimensions()
        if extra_dim:
            interp_data = np.empty(self._obj.shape, dtype=self._obj.dtype)
            i = 0
            for _, sub_xds in self._obj.groupby(extra_dim):
                src_data = sub_xds.load().data
                interp_data[i, ...] = self._interpolate_na(src_data, method=method)
                i += 1
        else:
            interp_data = self._interpolate_na(self._obj.load().data, method=method)
        interp_array = xr.DataArray(
            name=self._obj.name,
            dims=self._obj.dims,
            coords=self._obj.coords,
            data=interp_data,
            attrs=self._obj.attrs,
        )
        return interp_array

    def to_raster(
        self,
        raster_path,
        driver="GTiff",
        dtype=None,
        tags=None,
        windowed=False,
        mask=False,
        logger=logger,
        **profile_kwargs,
    ):
        """
        Export the DataArray to a raster file.

        Arguments
        ----------
        raster_path: str
            The path to output the raster to.
        driver: str, optional
            The name of the GDAL/rasterio driver to use to export the raster.
            Default is "GTiff".
        dtype: str, optional
            The data type to write the raster to. Default is the datasets dtype.
        tags: dict, optional
            A dictionary of tags to write to the raster.
        windowed: bool, optional
            If True, it will write using the windows of the output raster.
            Default is False.
        **profile_kwargs
            Additional keyword arguments to pass into writing the raster. The
            nodata, transform, crs, count, width, and height attributes
            are ignored.

        """
        for k in ["height", "width", "count", "transform"]:
            if k in profile_kwargs:
                msg = f"{k} will be set based on the DataArray, remove the argument"
                raise ValueError(msg)
        da_out = self._obj
        # set nodata, mask, crs and dtype
        if "nodata" in profile_kwargs:
            da_out.rio.set_nodata(profile_kwargs.pop("nodata"))
        nodata = da_out.rio.nodata
        if nodata is not None and not np.isnan(nodata):
            da_out = da_out.fillna(nodata)
        elif nodata is None:
            logger.warning(f"nodata value missing for {raster_path}")
        if mask and "mask" in da_out.coords and nodata is not None:
            da_out = da_out.where(da_out.coords["mask"] != 0, nodata)
        if dtype is not None:
            da_out = da_out.astype(dtype)
        if "crs" in profile_kwargs:
            da_out.rio.set_crs(profile_kwargs.pop("crs"))
        # check dimensionality
        extra_dim = da_out.rio._check_dimensions()
        count = 1
        if extra_dim is not None:
            count = da_out[extra_dim].size
            da_out = da_out.sortby(extra_dim)
        # write
        profile = dict(
            driver=driver,
            height=da_out.rio.height,
            width=da_out.rio.width,
            count=count,
            dtype=str(da_out.dtype),
            crs=da_out.rio.crs,
            transform=da_out.rio.transform,
            nodata=nodata,
            **profile_kwargs,
        )
        with rasterio.open(raster_path, "w", **profile) as dst:
            if windowed:
                window_iter = dst.block_windows(1)
            else:
                window_iter = [(None, None)]
            for _, window in window_iter:
                if window is not None:
                    row_slice, col_slice = window.toslices()
                    sel = {self.x_dim: col_slice, self.y_dim: row_slice}
                    data = da_out.isel(sel).load().values
                else:
                    data = da_out.load().values
                if data.ndim == 2:
                    dst.write(data, 1, window=window)
                else:
                    dst.write(data, window=window)
            if tags is not None:
                dst.update_tags(**tags)

    def vectorize(self, connectivity=8):
        """Returns shapes and values of connected regionis in a DataArray
        
        Parameters
        ----------
        connectivity : int, optional
            Use 4 or 8 pixel connectivity for grouping pixels into features, by default 8
        
        Returns
        -------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame of shapes and values.
        """
        feats_gen = features.shapes(
            self._obj.values,
            mask=(self._obj != self.nodata).values,
            transform=self.transform,
            connectivity=connectivity,
        )
        feats = [
            {"geometry": geom, "properties": {"value": idx}}
            for geom, idx in list(feats_gen)
        ]
        crs = None if self.crs is None else self.crs.to_epsg()
        gdf = gp.GeoDataFrame.from_features(feats, crs=crs)
        gdf.index = gdf.index.astype(self._obj.dtype)

        return gdf

    # TODO port https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html#rasterio.features.sieve
    # def sieve(self):
    #     pass


@xr.register_dataset_accessor("rio")
class RasterDataset(XRasterBase):
    """This is the GIS extension for :class:`xarray.Dataset`"""

    @property
    def vars(self):
        """list: Returns non-coordinate varibles"""
        return list(self._obj.data_vars.keys())

    @staticmethod
    def from_numpy(data_vars, transform, attrs=None, crs=None):
        """
        Transforms multiple numpy 2D/3D arrays to a RasterDataset. 
        The data should have identical y and x dimension.

        Parameters
        ----------
        data_vars: - dict-like
            A mapping from variable names to numpy arrays. The following notations
            are accepted:
            {var_name: array-like}
            {var_name: (array-like, nodata)}
            {var_name: (array-like, nodata, attrs)}
        transform : affine transform
            Two dimensional affine transform for 2D linear mapping
        attrs : dict, optional
            additional global attributes
        crs : str,
            coordinate reference system in WKT

        Returns
        -------
        ds : xr.Dataset

        """
        da_lst = list()
        for i, (name, data) in enumerate(data_vars.items()):
            args = ()
            if isinstance(data, tuple):
                data, args = data[0], data[1:]
            da = RasterArray.from_numpy(data, transform, *args)
            da.name = name
            if i > 0:
                if da.shape[-2:] != _shape:
                    raise xr.MergeError(f"Data shapes do not match.")
            else:
                _shape = da.shape[-2:]
            da_lst.append(da)
        ds = xr.merge(da_lst)
        if attrs is not None:
            ds.attrs.update(attrs)
        if crs is not None:
            ds.rio.set_crs(input_crs=crs)  # parse crs information
        return ds

    def _check_dimensions(self):
        for dvar in self.vars:
            self._obj[dvar].rio._check_dimensions()

    def reproject(
        self,
        dst_crs=None,
        dst_res=None,
        dst_transform=None,
        dst_width=None,
        dst_height=None,
        dst_nodata=None,
        method="nearest",
        align=False,
    ):
        """
        Reproject a DataArray. Powered by rasterio.warp.reproject.

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.

        Arguments
        ----------
        dst_crs: CRS or dict, optional
            Target coordinate reference system. Required if source and
            destination are ndarrays. Will be derived from target if it
            is a rasterio Band.
        dst_res: tuple (x resolution, y resolution) or float, optional
            Target resolution, in units of target coordinate reference
            system.
        dst_transform: affine.Affine(), optional
            Target affine transformation. Required if source and
            destination are ndarrays. Will be derived from target if it is
            a rasterio Band.
        dst_width, dst_height: int, optional
            Output file size in pixels and lines. Cannot be used together
            with resolution (dst_res).
        dst_nodata: int or float, optional
            The nodata value used to initialize the destination; it will
            remain in all areas not covered by the reprojected source.
            Defaults to the nodata value of the destination image (if set),
            the value of src_nodata, or 0 (GDAL default).
        method: str, optional
            See rasterio.warp.reproject for existing methods, by default nearest.
        align: boolean, optional
            If True, align target transform to resolution

        Returns
        --------
        resampled_dataset : xarray.Dataset
            A reprojected Dataset.
        """
        resampled_dataset = xr.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            resampled_dataset[var] = self._obj[var].rio.reproject(
                dst_crs=dst_crs,
                dst_res=dst_res,
                dst_transform=dst_transform,
                dst_width=dst_width,
                dst_height=dst_height,
                dst_nodata=dst_nodata,
                method=method,
                align=align,
            )
        return resampled_dataset

    def reproject_like(self, da_like, method="nearest"):
        """
        Reproject a Dataset object to match the resolution, projection,
        and region of another DataArray.

        .. note:: Only 2D/3D arrays with dimensions 'x'/'y' are currently supported.
            Requires either a grid mapping variable with 'spatial_ref' or
            a 'crs' attribute to be set containing a valid CRS.
            If using a WKT (e.g. from spatiareference.org), make sure it is an OGC WKT.


        Arguments
        ----------
        da_like: :obj:`xarray.DataArray`
            DataArray of the target resolution and projection.
        method: str, optional
            See rasterio.warp.reproject for existing methods, by default nearest.

        Returns
        --------
        resampled_dataset : xarray.Dataset
            Contains the data from the src_data_array, reprojected to match da_like.
        """
        resampled_dataset = xr.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            resampled_dataset[var] = self._obj[var].rio.reproject_like(
                da_like, method=method
            )
        return resampled_dataset

    def to_mapstack(
        self,
        root,
        driver="GTiff",
        dtype=None,
        tags=None,
        windowed=False,
        mask=False,
        prefix="",
        postfix="",
        logger=logger,
        **profile_kwargs,
    ):
        """
        Export the DataArray to a raster file.
        Files are written to <root>/<prefix><name><postfix>.<ext> are read.

        Arguments
        ----------
        root : str
            The path to output the raster to. It is created if it does not yet exist.
        driver : str, optional
            The name of the GDAL/rasterio driver to use to export the raster.
            Default is "GTiff".
        dtype : str, optional
            The data type to write the raster to. Default is the datasets dtype.
        tags : dict, optional
            A dictionary of tags to write to the raster.
        windowed : bool, optional
            If True, it will write using the windows of the output raster.
            Default is False.
        prefix : str, optional
            Prefix to filenames in mapstack
        postfix : str, optional
            Postfix to filenames in mapstack
        **profile_kwargs
            Additional keyword arguments to pass into writing the raster. The
            nodata, transform, crs, count, width, and height attributes
            are ignored.

        """
        if driver not in GDAL_EXT_CODE_MAP:
            raise ValueError(f"Extension unknown for driver: {driver}")
        ext = GDAL_EXT_CODE_MAP.get(driver)
        if not isdir(root):
            os.makedirs(root)
        for var in self.vars:
            if "/" in var:
                # variables with in subfolders
                folders = "/".join(var.split("/")[:-1])
                if not isdir(join(root, folders)):
                    os.makedirs(join(root, folders))
                var0 = var.split("/")[-1]
                raster_path = join(root, folders, f"{prefix}{var0}{postfix}.{ext}")
            else:
                raster_path = join(root, f"{prefix}{var}{postfix}.{ext}")
            self._obj[var].rio.to_raster(
                raster_path,
                driver=driver,
                dtype=dtype,
                tags=tags,
                windowed=windowed,
                mask=mask,
                logger=logger,
                **profile_kwargs,
            )
