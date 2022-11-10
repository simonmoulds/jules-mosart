#!/usr/bin/env python

import os
import sys
import tempfile
import subprocess
import numpy as np
import xarray

# export JULES_START_YEAR=$(yaml $CONFIG_FILE "['jules']['start_year']")
# export JULES_END_YEAR=$(yaml $CONFIG_FILE "['jules']['end_year']")
# export JULES_SUITE=$(yaml $CONFIG_FILE "['jules']['suite']")
# export JULES_ID_STEM=$(yaml $CONFIG_FILE "['jules']['id_stem']")
# export JULES_JOB_NAME=$(yaml $CONFIG_FILE "['jules']['job_name']")
# export JULES_PROFILE_NAME=$(yaml $CONFIG_FILE "['jules']['profile_name']")
# # export JULES_GRIDFILE=$(yaml $CONFIG_FILE "['jules']['gridfile']")
# # export JULES_OUTPUT_DIRECTORY=$(yaml $CONFIG_FILE "['jules']['raw_output_directory']")
# # export GRIDTYPE=$(yaml $CONFIG_FILE "['resample_jules_output']['gridtype']")
# # export XSIZE=$(yaml $CONFIG_FILE "['resample_jules_output']['xsize']")
# # export YSIZE=$(yaml $CONFIG_FILE "['resample_jules_output']['ysize']")
# # export XFIRST=$(yaml $CONFIG_FILE "['resample_jules_output']['xfirst']")
# # export YFIRST=$(yaml $CONFIG_FILE "['resample_jules_output']['yfirst']")
# # export XINC=$(yaml $CONFIG_FILE "['resample_jules_output']['xinc']")
# # export YINC=$(yaml $CONFIG_FILE "['resample_jules_output']['yinc']")
# NORTH=$(yaml $CONFIG_FILE "['resample_jules_output']['ymax']")
# SOUTH=$(yaml $CONFIG_FILE "['resample_jules_output']['ymin']")
# EAST=$(yaml $CONFIG_FILE "['resample_jules_output']['xmax']")
# WEST=$(yaml $CONFIG_FILE "['resample_jules_output']['xmin']")
# export N=$(ceil $NORTH)
# export S=$(floor $SOUTH)
# export E=$(ceil $EAST)
# export W=$(floor $WEST)
# # Must be one of 05min 15min 30sec [TODO: check]
# export RES=$(yaml $CONFIG_FILE "['resample_jules_output']['resolution']")
# export INPUT_FILE_SUFFIX=$(yaml $CONFIG_FILE "['resample_jules_output']['input_file_suffix']")
# export OUTPUT_FILE_SUFFIX=$(yaml $CONFIG_FILE "['resample_jules_output']['output_file_suffix']")
# export INPUT_DIRECTORY=$(yaml $CONFIG_FILE "['resample_jules_output']['input_directory']")
# export OUTPUT_DIRECTORY=$(yaml $CONFIG_FILE "['resample_jules_output']['output_directory']")
# export SRC_DIR=$(pwd)/../src

START = int(os.environ['JULES_START_YEAR'])
END = int(os.environ['JULES_END_YEAR'])
YEARS = np.arange(START, END + 1)
SUITE = str(os.environ['JULES_SUITE'])
ID_STEM = str(os.environ['JULES_ID_STEM'])
JOB_NAME = str(os.environ['JULES_JOB_NAME'])
PROFILE_NAME = str(os.environ['JULES_PROFILE_NAME'])
YMAX = float(os.environ['N'])
YMIN = float(os.environ['S'])
XMAX = float(os.environ['E'])
XMIN = float(os.environ['W'])
RES = str(os.environ['RES'])
INPUT_FILE_SUFFIX = str(os.environ['INPUT_FILE_SUFFIX'])
OUTPUT_FILE_SUFFIX = str(os.environ['OUTPUT_FILE_SUFFIX'])
INPUT_DIRECTORY = str(os.environ['INPUT_DIRECTORY'])
OUTPUT_DIRECTORY = str(os.environ['OUTPUT_DIRECTORY'])

if RES == '30sec':
    DRES = 0.008333333333
elif RES == '05min':
    DRES = 0.083333333333
elif RES == '15min':
    DRES = 0.25
else:
    stop()
    
def main():
    job_name = JOB_NAME.format(year=START)
    FN = os.path.join(
        INPUT_DIRECTORY, ID_STEM + '.' + job_name + '.' + PROFILE_NAME + '.' + str(START) + '.' + INPUT_FILE_SUFFIX + '.nc'
    )    
    with xarray.open_dataset(FN) as x:
        lat = x['lat'].values
        ns_lat = lat[0] > lat[1]
    # write CDO gridfile
    xsize = abs((XMAX - XMIN) / DRES)
    ysize = abs((YMAX - YMIN) / DRES)
    xfirst = XMIN + DRES / 2.
    xinc = abs(DRES)
    if ns_lat:
        yfirst = YMAX - DRES / 2.
        yinc = abs(DRES) * -1.
    else:
        yfirst = YMIN + DRES / 2.
        yinc = abs(DRES)
    with open('/tmp/gridfile.txt', 'w') as f:
        f.write('gridtype=lonlat\n')
        f.write('xsize=' + str(int(xsize)) + '\n')
        f.write('ysize=' + str(int(ysize)) + '\n')
        f.write('xfirst=' + str(xfirst) + '\n')
        f.write('xinc=' + str(xinc) + '\n')
        f.write('yfirst=' + str(yfirst) + '\n')
        f.write('yinc=' + str(yinc) + '\n')
        f.close()

    for yr in YEARS:
        job_name = JOB_NAME.format(year=yr)
        IN_FN = os.path.join(
            INPUT_DIRECTORY, ID_STEM + '.' + job_name + '.' + PROFILE_NAME + '.' + str(yr) + '.' + INPUT_FILE_SUFFIX + '.nc'            
        )
        TMP_FN = tempfile.NamedTemporaryFile(suffix='.nc')
        OUT_FN = os.path.join(
            OUTPUT_DIRECTORY, ID_STEM + '.' + job_name + '.' + PROFILE_NAME + '.' + str(yr) + '.' + OUTPUT_FILE_SUFFIX + '.nc'
        )
        # use bilinear interpolation for all continuous variables
        subprocess.run([
            'cdo',
            'remapbil,/tmp/gridfile.txt',
            IN_FN,
            TMP_FN.name
        ])
        # ensure that variables have datatype 'double', not 'short',
        # which seems to cause problems in JULES (not exactly sure why...)
        subprocess.run([
            'cdo',
            '-b', 'F64', 'copy',
            TMP_FN.name,
            OUT_FN
        ])
        
        
if __name__ == '__main__':
    main()
