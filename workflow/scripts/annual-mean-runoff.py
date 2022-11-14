#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr
import yaml
import click

@click.command()
@click.option('-i', '--inputfile', default='.', help='Name of input file')
@click.option('-o', '--outputfile', default='.', help='Name of output file')
@click.option('--config', default='config.yml', help='YAML configuration file')
def main(inputfile, outputfile, config):
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    start_year = str(int(config['jules']['start_year']))
    end_year = str(int(config['jules']['end_year']))
    with open(inputfile, 'r') as f:
        regrid_filelist = [ln.strip() for ln in f.readlines()]

    x = xr.open_mfdataset(regrid_filelist)
    jules_vars = [
        'surf_roff', 'sub_surf_roff', 'runoff', # 'precip',
        'ecan_gb', 'elake', 'esoil_gb', 'fao_et0'
    ]        
    for var in jules_vars:
        x[var] = x[var] * 60 * 60 * 24 / 1000  # kg m-2 s-1 -> m d-1
    # m d-1 -> m y-1
    x_year = x.groupby("time.year").sum(dim="time")
    x_annual_mean = x_year.mean(dim="year")

    path = os.path.split(regrid_filelist[0])[0]
    nc_outputfile = os.path.join(path, 'jules_ann_mean_' + start_year + '_' + end_year + '_regrid.nc')
    x_annual_mean.to_netcdf(nc_outputfile)
    x.close()

    # Write output filename to text file
    with open(outputfile, 'w') as f:
        f.write(("%s" + os.linesep) % nc_outputfile)

if __name__ == '__main__':
    main()
