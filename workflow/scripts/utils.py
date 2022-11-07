#!/usr/bin/env python3

import os
import numpy as np
import xarray


def aggregate_to_month(x):
    # Get the number of days in each month for the current dataset
    days_in_month = xarray.DataArray(
        np.ones(x.time.shape[0]),
        {'time' : x.time.values},
        dims=['time']
    ).groupby('time.month').sum(dim='time')
    x_month = (
        x.groupby('time.month').mean(dim='time')
        * 60 * 60 * 24 * days_in_month
    )
    return x_month

def convert_to_2d(x,
                  variables,
                  lat,
                  lon,
                  mask,
                  soil_dim_name,
                  tile_dim_name,
                  pft_dim_name):

    nlat = len(lat)
    nlon = len(lon)
    # Find out whether latitudes are N-S or S-N
    ns_lat = lat[0] > lat[1]

    # Obtain length of dimensions
    if 'time' in x.dims:
        ntime = x.dims['time']
    if 'month' in x.dims:
        ntime = x.dims['month']
    if 'tile' in x.dims:
        ntile = x.dims['tile']
    if 'soil' in x.dims:
        nsoil = x.dims['soil']
    if 'pft' in x.dims:
        npft = x.dims['pft']
    if 'soilt' in x.dims:
        nsoilt = x.dims['soilt']

    # Update mask to handle subgrid
    # This method hinges on np.isclose(...) being able to
    # reliably match coordinates
    x_lat = x.latitude.values.squeeze()
    x_lon = x.longitude.values.squeeze()
    x_mask = np.zeros_like(mask)
    for i in range(len(x_lat)):
        latv = x_lat[i]
        lonv = x_lon[i]
        try:
            latv_ix = np.where(np.isclose(lat, latv))[0]
            lonv_ix = np.where(np.isclose(lon, lonv))[0]
        except:
            raise ValueError("1")
        if (len(latv_ix) != 1) | (len(lonv_ix) != 1):
            raise ValueError("2")
        x_mask[latv_ix, lonv_ix] = 1

    # Get IDs of cells in the 1D JULES output
    arr = np.arange(nlat * nlon).reshape((nlat, nlon))
    if ns_lat:
        arr = np.flipud(arr)
    y_index = arr[x_mask > 0]

    # Get coordinate pairs for grid
    lat_index = np.array([np.ones((nlon)) * i for i in range(nlat)], dtype=int).flatten()
    lon_index = np.tile(np.arange(nlon), nlat)
    indices = np.array([[lat_index[i], lon_index[i]] for i in y_index], dtype=int)
    def get_coords(soil=False, soilt=False, tile=False, pft=False):
        coords={}
        dims=[]
        if 'time' in x.dims:
            coords['time'] = x.time.values[:]
            dims.append('time')
        elif 'month' in x.dims:
            coords['month'] = x.month.values[:]
            dims.append('month')
        if soil:
            coords[soil_dim_name] = np.arange(1, nsoil + 1)
            dims.append(soil_dim_name)
        if soilt:
            coords['soilt'] = x.soilt.values[:]
            dims.append('soilt')
        if tile:
            coords[tile_dim_name] = np.arange(1, ntile + 1)
            dims.append(tile_dim_name)
        if pft:
            coords[pft_dim_name] = np.arange(1, npft + 1)
            dims.append(pft_dim_name)
        coords['lat'] = lat
        coords['lon'] = lon
        dims = dims + ['lat', 'lon']
        return coords, dims

    # Define functions to handle outputs with different dimensions
    def convert_gridbox_soilt_var(output_var):
        # (time, soilt, y, x)
        arr = np.zeros((ntime, nsoilt) + (nlat, nlon)) * np.nan
        for i in range(ntime):
            for j in range(nsoilt):
                output_arr = output_var.values[:][i, j, ...].squeeze()
                arr_ij = arr[i, j, ...]
                arr_ij[tuple(indices.T)] = output_arr
                arr[i, j, ...] = arr_ij
        coords, dims = get_coords(soilt=True)
        xarr = xarray.DataArray(arr, coords, dims, attrs=output_var.attrs)
        return xarr

    def convert_gridbox_var(output_var):
        # (time, y, x)
        arr = np.zeros((ntime,) + (nlat, nlon)) * np.nan
        for i in range(ntime):
            output_arr = output_var.values[:][i, ...].squeeze()
            arr_ij = arr[i, ...]
            arr_ij[tuple(indices.T)] = output_arr
            arr[i, ...] = arr_ij
        # TODO: add mask?
        coords, dims = get_coords()
        xarr = xarray.DataArray(arr, coords, dims, attrs=output_var.attrs)
        return xarr

    def convert_tile_var(output_var):
        # (time, tile, y, x)
        arr = np.zeros((ntime, ntile) + (nlat, nlon)) * np.nan
        for i in range(ntime):
            for j in range(ntile):
                output_arr = output_var.values[:][i, j, ...].squeeze()
                arr_ij = arr[i, j, ...]
                arr_ij[tuple(indices.T)] = output_arr
                arr[i, j, ...] = arr_ij
        coords, dims = get_coords(tile=True)
        xarr = xarray.DataArray(arr, coords, dims, attrs=output_var.attrs)
        return xarr

    def convert_soil_soilt_var(output_var):
        # (time, soil, soilt, y, x)
        arr = np.zeros((ntime, nsoil, nsoilt) + (nlat, nlon)) * np.nan
        for i in range(ntime):
            for j in range(nsoil):
                for k in range(nsoilt):
                    output_arr = output_var.values[:][i, j, k, ...].squeeze()
                    arr_ijk = arr[i, j, k, ...]
                    arr_ijk[tuple(indices.T)] = output_arr
                    arr[i, j, k, ...] = arr_ijk
        coords, dims = get_coords(soil=True, soilt=True)
        xarr = xarray.DataArray(arr, coords, dims, attrs=output_var.attrs)
        return xarr

    def convert_soil_var(output_var):
        # (time, soil, y, x)
        arr = np.zeros((ntime, nsoil) + (nlat, nlon)) * np.nan
        for i in range(ntime):
            for j in range(nsoil):
                output_arr = output_var.values[:][i, j, ...].squeeze()
                arr_ij = arr[i, j, ...]
                arr_ij[tuple(indices.T)] = output_arr
                arr[i, j, ...] = arr_ij
        coords, dims = get_coords(soil=True)
        xarr = xarray.DataArray(arr, coords, dims, attrs=output_var.attrs)
        return xarr

    def convert_pft_var(output_var):
        # (time, pft, y, x)
        arr = np.zeros((ntime, npft) + (nlat, nlon)) * np.nan
        for i in range(ntime):
            for j in range(npft):
                output_arr = output_var.values[:][i, j, ...].squeeze()
                arr_ij = arr[i, j, ...]
                arr_ij[tuple(indices.T)] = output_arr
                arr[i, j, ...] = arr_ij
        coords, dims = get_coords(pft=True)
        xarr = xarray.DataArray(arr, coords, dims, attrs=output_var.attrs)
        return xarr

    # Loop through variables and create a list of DataArrays
    xarr_list = []
    for var in variables:
        if var in x.variables:
            if 'tile' in x[var].dims:
                xarr = convert_tile_var(x[var])
            elif 'soil' in x[var].dims:
                if 'soilt' in x[var].dims:
                    xarr = convert_soil_soilt_var(x[var])
                else:
                    xarr = convert_soil_var(x[var])
            elif 'pft' in x[var].dims:
                xarr = convert_pft_var(x[var])
            else:
                if 'soilt' in x[var].dims:
                    xarr = convert_gridbox_soilt_var(x[var])
                else:
                    xarr = convert_gridbox_var(x[var])
            xarr.name = var
            xarr_list.append(xarr)

    # Merge DataArray objects into single dataset
    ds = xarray.merge(xarr_list)
    return ds
