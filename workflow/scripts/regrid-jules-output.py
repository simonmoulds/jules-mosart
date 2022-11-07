#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import xarray
import click
import yaml
from constants import OUTPUT_VARS
from utils import convert_to_2d


@click.command()
@click.option('-o', '--outputfile', default='.', help='Name of output file')
@click.option('--config', default='config.yml', help='YAML configuration file')
def main(outputfile, config):
    # Open configuration file
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Open 2D land fraction file to obtain land mask
    y = xarray.open_dataset(GRIDFILE)
    MASK = y[MASK_VAR_NAME].values
    LAT = y[Y_DIM_NAME].values[:]
    LON = y[X_DIM_NAME].values[:]
    y.close()

    # Loop through years
    # TODO - is it a requirement that JULES output is written annually? If so then we ought to also include option for monthly
    filelist = open(outputfile, 'w')
    for i in tqdm(range(len(YEARS))):
        yr = YEARS[i]
        job_name = JOB_NAME.format(year=yr)
        FN = ID_STEM + '.' + job_name + '.' + PROFILE_NAME + '.' + str(yr) + '.nc'
        x = xarray.open_dataset(os.path.join(DATADIR, FN))
        ds = convert_to_2d(
            x, OUTPUT_VARS[PROFILE_NAME], LAT, LON, MASK,
            config['jules']['soil_dim_name'],
            config['jules']['tile_dim_name'],
            config['jules']['pft_dim_name']
        )
        ds['lat'].attrs['standard_name'] = 'latitude'
        ds['lat'].attrs['units'] = 'degrees_north'
        ds['lon'].attrs['standard_name'] = 'longitude'
        ds['lon'].attrs['units'] = 'degrees_east'
        nc_outputfile = os.path.join(
            OUTDIR, os.path.splitext(FN)[0] + '.' + FILE_SUFFIX + '.nc'
        ),
        ds.to_netcdf(nc_outputfile, format="NETCDF4")
        x.close()
        filelist.write(("%s" + os.linesep) % nc_outputfile)
    filelist.close()

if __name__ == '__main__':
    main()

