#!/usr/bin/env python

import os
import sys
import tempfile
import subprocess
import numpy as np
import xarray
import click
import yaml
from tqdm import tqdm

TMPDIR = os.environ['TMPDIR']
if TMPDIR == '':
    TMPDIR = '/tmp'

@click.command()
@click.option('-i', '--inputfile', default='.', help='Name of input file')
@click.option('-o', '--outputfile', default='.', help='Name of output file')
@click.option('--config', default='config.yml', help='YAML configuration file')
def main(inputfile, outputfile, config):
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    res = config['resample_jules_output']['resolution']
    if res not in ['30sec', '5min', '15min']:
        raise ValueError('Invalid resolution: must be one of `30sec`, `5min` or `15min`')

    if res == '30sec':
        dres = 0.008333333333
    elif res == '5min':
        dres = 0.083333333333
    elif res == '15min':
        dres = 0.25

    xmin = config['resample_jules_output']['xmin']
    xmax = config['resample_jules_output']['xmax']
    ymin = config['resample_jules_output']['ymin']
    ymax = config['resample_jules_output']['ymax']

    with open(inputfile, 'r') as f:
        regrid_filelist = [ln.strip() for ln in f.readlines()]

    # Write CDO gridfile - use first input file as a template
    with xarray.open_dataset(regrid_filelist[0]) as x:
        lat = x['lat'].values
        ns_lat = lat[0] > lat[1]

    xsize = abs((xmax - xmin) / dres)
    ysize = abs((ymax - ymin) / dres)
    xfirst = xmin + dres / 2.
    xinc = abs(dres)
    if ns_lat:
        yfirst = ymax - dres / 2.
        yinc = abs(dres) * -1.
    else:
        yfirst = ymin + dres / 2.
        yinc = abs(dres)
    GRIDFILE = os.path.join(TMPDIR, 'gridfile.txt')
    with open(GRIDFILE, 'w') as f:
        f.write('gridtype=lonlat\n')
        f.write('xsize=' + str(int(xsize)) + '\n')
        f.write('ysize=' + str(int(ysize)) + '\n')
        f.write('xfirst=' + str(xfirst) + '\n')
        f.write('xinc=' + str(xinc) + '\n')
        f.write('yfirst=' + str(yfirst) + '\n')
        f.write('yinc=' + str(yinc) + '\n')
        f.close()

    output_filelist = open(outputfile, 'w')
    for filepath in tqdm(regrid_filelist):
        path = os.path.split(filepath)[0]
        filename = os.path.split(filepath)[1]
        basename = os.path.splitext(filename)[0]
        tmp_file = tempfile.NamedTemporaryFile(suffix='.nc')
        nc_outputfile = os.path.join(path, basename + '.regrid.nc')
        # Use bilinear interpolation for all continuous variables
        tmp_file = tempfile.NamedTemporaryFile(suffix='.nc')
        subprocess.run([
            'cdo',
            'remapbil,' + GRIDFILE,
            filepath,
            # os.path.join(TMPDIR, 'tmp.nc')
            tmp_file.name
        ])
        # Ensure that variables have datatype 'double', not 'short',
        # which seems to cause problems in JULES (not exactly sure why...)
        subprocess.run([
            'cdo',
            '-b', 'F64', 'copy',
            # os.path.join(TMPDIR, 'tmp.nc'),
            tmp_file.name,
            nc_outputfile
        ])
        output_filelist.write(("%s" + os.linesep) % nc_outputfile)
        # Tidy up
        os.remove(os.path.join(TMPDIR, 'tmp.nc'))

    output_filelist.close()
        
if __name__ == '__main__':
    main()
