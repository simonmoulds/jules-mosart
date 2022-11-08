import os
import numpy as np
import pandas as pd
import xarray
import click
import yaml
from tqdm import tqdm
from constants import OUTPUT_VARS
from utils import convert_to_2d


@click.command()
@click.option('-o', '--outputfile', default='.', help='Name of output file')
@click.option('--config', default='config.yml', help='YAML configuration file')
def main(outputfile, config):
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    start_year = int(config['jules']['start_year'])
    end_year = int(config['jules']['end_year'])
    years = np.arange(start_year, end_year+1)
    id_stem = config['jules']['id_stem']
    job_name = config['jules']['job_name']
    profile_name = config['jules']['profile_name']
    jules_output_directory = config['jules']['jules_output_directory']
    gridfile = config['jules']['gridfile']
    y_dim_name = config['jules']['y_dim_name']
    x_dim_name = config['jules']['x_dim_name']
    mask_var_name = config['jules']['mask_var_name']
    soil_dim_name = config['jules']['soil_dim_name']
    tile_dim_name = config['jules']['tile_dim_name']
    pft_dim_name = config['jules']['pft_dim_name']

    # Open 2D land fraction file to obtain land mask
    y = xarray.open_dataset(gridfile)
    mask = y[mask_var_name].values
    lat = y[y_dim_name].values[:]
    lon = y[x_dim_name].values[:]
    y.close()

    outputdir = config['regrid_jules_output']['output_directory']
    file_suffix = config['regrid_jules_output']['file_suffix']
    os.makedirs(outputdir, exist_ok=True)

    # Loop through years
    # TODO - is it a requirement that JULES output is written annually? If so then we ought to also include option for monthly
    # filelist = open(outputfile, 'w')
    for i in tqdm(range(len(years))):
        yr = years[i]
        job_name = job_name.format(year=yr)
        filename = (
            id_stem + '.' + job_name + '.' + profile_name + '.' + str(yr) + '.nc'
        )
        print(os.path.join(jules_output_directory, filename))
        x = xarray.open_dataset(os.path.join(jules_output_directory, filename))
        # print(x)
        # ds = convert_to_2d(
        #     x, OUTPUT_VARS[profile_name], lat, lon, mask,
        #     soil_dim_name, tile_dim_name, pft_dim_name
        # )
        # ds['lat'].attrs['standard_name'] = 'latitude'
        # ds['lat'].attrs['units'] = 'degrees_north'
        # ds['lon'].attrs['standard_name'] = 'longitude'
        # ds['lon'].attrs['units'] = 'degrees_east'
        # nc_outputfile = os.path.join(
        #     outputdir, os.path.splitext(filename)[0] + '.' + file_suffix + '.nc'
        # )
        # print(ds)
        # ds.to_netcdf(nc_outputfile) #, format="NETCDF4", engine="netcdf4")
        # x.close()
        # filelist.write(("%s" + os.linesep) % nc_outputfile)
    # filelist.close()

if __name__ == '__main__':
    main()

