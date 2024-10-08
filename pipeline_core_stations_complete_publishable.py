"""
Author: Cristina Cordun

pipeline_core_stations_complete_publishable.py

This is a short and simple pipeline that can calibrate LOFAR LBA data below 40 MHz when the dataset only contains core stations. 
It is important that the delay between stations is less than 10 ns. The observations from 2022 and later have two stations 
with delays around 250 ns. Their origin is unknown. This code does not deal with that unfortunately. A new code will come soon
 which can deal with this issues and can calibrate remote stations as well. Stay tuned. To run this, you need a basis container
   with all the lofar software. I can recomment you take a look here https://tikk3r.github.io/flocs/."
    
 
Usage:

run
    python3 pipeline_core_stations_complete_publishable.py --help 
to see all the options and how to use them


standard run (keep all options to default values)
    python3 pipeline_core_stations_complete_publishable.py --path=/path/to/datasets/ --model_cal=/path/to/datasets/calibrator.skymodel --sap_target=0 --model_target=/path/to/datasets/target.skymodel



Options:
    -h, --help                      Show this help message and exit
    --path                          The path to the folder with MS files. The folder should contain both the MS files with the name structure 
                                    of L{id}_SAP00{source_id}_SB_{subband_numer}_uv.MS. The subband number should alawys have 4 digits. 
                                    For example SB0001 for subband 1 or SB0233 for subband 233, etc. The source id is 0 or 1 depending if it is 
                                    the calibrator or targer
    --model_cal                     The path and name of the calibrator model file in .skymodel format. 
                                    You can use https://lcs165.lofar.eu/ to generate it and copy/paste the output.
    --model_target                  The path and name of the calibrator model file in .skymodel format. 
                                    You can use https://lcs165.lofar.eu/  to generate it and copy/paste the output.')
    --sap_target                    The SAP number of the target. Can either be 0 or 1.
    --demix_A_team                  Says if the A-team sources will be removed. If not, the data will only be averaged. Default: True
    --channels_avg                  The number of channels that you want for demixing/averaging. Default: 128
    --time_avg                      The number of time slots that you want for demixing/averaging. Default: 30
    --channels_image                The number of channels that you want when imaging in shorter frequency bands. Default: 20
    --channels_selfcal_target       The number of channels that you want constant for selfcalibration. Default: 5. 
                                    Important: This is after the data is averaged, so it is the number of averaged channels.
    --time_selfcal_target           The number of time slots that you want constant for selfcalibration. Default: 30. 
                                    Important!: This is after the data is averaged, so it is the number of averaged time slots.
    --start"                        The steps at which you want to start. If none, it starts at the most recent step that did not finish. 
                                    The steps are: Flagging, Demixing/Averaging, Merging, SecondFlagging, PolAlign, Bandpass, ClockTEC3, 
                                    FirstImaging, ThirdFlagging, TEC3, SecondImaging, PolLeakage, ThirdImaging, Dynspec. 
                                    Dynspec works only if dynspec == True
    --dynspec                       Says if a dynamic spectra of the phase center should be made. 
                                    Useful for transient studies if the target is in the phase center. Default: True
    --time_slots_avg_dynspec        The number of time slots that should be averaged in the dynamic spectrum. Default: 24
    --freq_slots_avg_dynspec        The number of frequency bands that should be averaged in the dynamic spectrum. Default: 10
    

"""


import os
import sys, getopt
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import signal
from textwrap import dedent
from scipy.optimize import curve_fit
from losoto.h5parm import h5parm
from losoto.operations import polalign
import multiprocessing
from functools import partial
import bdsf
from astropy.io import fits
from casacore.tables import *
import argparse
import logging
import datetime
import subprocess
import tempfile


def run_process(command):

    '''Funtion that runs a command in the shell. subprocess has to be used instead of os.system because os.system does not work with log files.
    The subprocesses library cannot work with long commands (which are required for wsclean) so i create a temporary fexecutable file and run it to trick it.

    ***Parameters***
    command: string
        The command that needs to be run

    ***Returns***
    Nothing, just runs the command.
    '''

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.bat') as script_file:
        script_file.write(command)
        script_path = script_file.name

    subprocess.run(['chmod', '+x', script_path])
    subprocess.run(command, shell = True)
    os.remove(script_path)



def calculatePolAlign(infile, folder):
    '''Function that calculates the polarization missalignment 
    using losoto (https://revoltek.github.io/losoto/losoto.html). 
    This function only plots the first direction (if there are more 
    than one in ddecal (https://dp3.readthedocs.io/en/latest/steps/DDECal.html)).

    **Parameters**
    infile: string
        The name of the gain solutions h5 file. 
        The solutions have to be calculated in rotation+diagonal mode or the plotting does not work.
    folder: string
        The folder name where the solutions will be saved, usually called "PolAlignPlots"

    **Returns**
    It does not return any values but it saves a folder called {folder} 
    where the phase, amplitude, and differential phase solutions and its 
    fitting can be looked at.'''

    logging.info(f"Calculating polarization misalignment with Losoto")
    H = h5parm(infile, readonly=False)
    soltab = H.getSolset('sol000').getSoltab('phase000')
    polalign.run(soltab,  refAnt = 'CS001LBA')
    H.close()

    logging.info(f"Reading and plotting the solutions. The plots are in the folder {folder}")
    f = h5py.File(infile)
    freq = np.array(((f['sol000/phase000/freq'])))
    phases = np.array(((f['sol000/phase000/val'])))
    amplitudes = np.array(((f['sol000/amplitude000/val'])))
    antennas = np.array(((f['sol000/phase000/ant'])))
    phasediff = np.array(((f['sol000/phasediff/val'])))
    time = f['sol000/phase000/time']
    

    freq = np.array(freq)
    phases = np.array(phases)
    amplitudes = np.array(amplitudes)
    time = np.array(time)

    phases = phases[:,:,0,:,:]
    amplitudes = amplitudes[:,:,0,:,:]
    phases = np.swapaxes(phases, 1,2)
    amplitudes = np.swapaxes(amplitudes,1,2)

    if not os.path.exists(folder):
        os.mkdir(folder)

    rows = 3
    subplots_num = int(phases.shape[2]/rows)
    phases_combined_ref = np.empty_like(phases)
    for ant in range(phases.shape[2]):   
        phases_combined_ref[:,:,ant,:] = np.angle(np.exp(1j*phases[:,:,ant,:])*np.exp(-1j*phases[:,:,0,:]))


    fig = plt.figure(layout='constrained', figsize=(20, 10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((phases_combined_ref[:,:,antenna,0]).T-(phases_combined_ref[:,:,antenna,1]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'hsv', origin = 'lower', vmin = -np.pi, vmax = np.pi)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Phases xx-yy', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/phases_xx-yy.png', format = 'png', bbox_inches = 'tight')

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((phases_combined_ref[:,:,antenna,0]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'hsv', origin = 'lower', vmin = -np.pi, vmax = np.pi)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Phases xx', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/phases_xx.png', format = 'png', bbox_inches = 'tight')

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((phases_combined_ref[:,:,antenna,1]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'hsv', origin = 'lower', vmin = -np.pi, vmax = np.pi)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Phases yy', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/phases_yy.png', format = 'png', bbox_inches = 'tight')

    med = np.nanmedian(amplitudes, axis = (0,1,2))
    mad = np.nanmedian (np.absolute(amplitudes - med), axis = (0,1,2))
    vmin = med - 3*mad
    vmax = med + 3*mad

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((amplitudes[:,:,antenna,0]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'plasma', origin = 'lower', vmin = vmin[0], vmax = vmax[0])
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Amplitudes xx', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/amplitudes_xx.png', format = 'png', bbox_inches = 'tight')

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((amplitudes[:,:,antenna,1]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'plasma', origin = 'lower', vmin = vmin[1], vmax = vmax[1])
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Amplitudes yy', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/amplitudes_yy.png', format = 'png', bbox_inches = 'tight')

    pol_sol = phases_combined_ref[:,:,:,0]-phases_combined_ref[:,:,:,1]
    for f in range(pol_sol.shape[0]):
        for ant in range(pol_sol.shape[2]):
            pol_sol[f,:,ant] = signal.medfilt(pol_sol[f,:,ant], kernel_size=3)
    pol_sol = np.nanmean(pol_sol, axis = 0)


    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)

    for antenna in range(0,phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].plot(freq,pol_sol[:,antenna])
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].plot(freq,-phasediff[0,antenna,0,:,1])
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylim(-0.2,0.2)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Phase [rad]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Fit xx-yy using LoSoTo', fontsize=12)
    plt.savefig(f'{folder}/Phase_fit_polAlign.png', format = 'png', bbox_inches = 'tight')


def calculate_bandpass(infile, outfile, folder):

    '''
    Function that calculates the bandpass of all the antennas. 
    This function only calculates solutions for the first direction (if there are more 
    than one in ddecal (https://dp3.readthedocs.io/en/latest/steps/DDECal.html)).

    **Parameters**
    infile: string
        The name of the gain solutions h5 file. The solutions have to be 
        calculated in diagonal mode because all the rotation effects should have been 
        corrected for (differential Faraday rotation).
    outfile: string
        The name of the h5 file where the solutions should be saved in the table amplitude000
    folder: string
        The folder where the plots and solutions will be solved


    **Returns**
    It does not return any values but it saves a folder called {folder} 
    where the phase, amplitude, and banspass solutions can be looked at.
    It saves a h5 file called {outfile} where the bandpass values are written and will be applied to the data.
    '''

    logging.info(f"Reading and plotting the solutions. The plots are in the folder {folder}")
    f = h5py.File(infile)
    freq = np.array(((f['sol000/phase000/freq'])))
    phases = np.array(((f['sol000/phase000/val'])))
    amplitudes = np.array(((f['sol000/amplitude000/val'])))
    antennas = np.array(((f['sol000/phase000/ant'])))
    time = f['sol000/phase000/time']

    freq = np.array(freq)
    phases = np.array(phases)
    amplitudes = np.array(amplitudes)
    time = np.array(time)

    phases = phases[:,:,:,0,:]
    amplitudes = amplitudes[:,:,:,0,:]

    if not os.path.exists(folder):
        os.mkdir(folder)

    phases_ref = np.empty_like(phases)
    for ant in range(phases.shape[2]):   
        phases_ref[:,:,ant,:] = np.angle(np.exp(1j*phases[:,:,ant,:])*np.exp(-1j*phases[:,:,0,:]))

    logging.info(f"The amplitudes are of shape:{amplitudes.shape}")


    rows = 3
    subplots_num = int(phases.shape[2]/rows)

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((phases_ref[:,:,antenna,0]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'hsv', origin = 'lower', vmin = -np.pi, vmax = np.pi)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Phases xx', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/phases_xx.png', format = 'png', bbox_inches = 'tight')

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((phases_ref[:,:,antenna,1]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'hsv', origin = 'lower', vmin = -np.pi, vmax = np.pi)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Phases yy', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/phases_yy.png', format = 'png', bbox_inches = 'tight')


    med = np.nanmedian(amplitudes, axis = (0,1,2))
    mad = np.nanmedian (np.absolute(amplitudes - med), axis = (0,1,2))
    vmin = med - 3*mad
    vmax = med + 3*mad

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((amplitudes[:,:,antenna,0]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'plasma', origin = 'lower', vmin = vmin[0], vmax = vmax[0])
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Amplitudes xx', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/amplitudes_xx.png', format = 'png', bbox_inches = 'tight')

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((amplitudes[:,:,antenna,1]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'plasma', origin = 'lower', vmin = vmin[1], vmax = vmax[1])
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Amplitudes yy', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/amplitudes_yy.png', format = 'png', bbox_inches = 'tight')


    logging.info(f"Flagging bad apmplitudes")
    amplitudes_median = np.nanmedian(amplitudes, axis=(0,2))
    mask = np.zeros_like(amplitudes, dtype = bool)
    weight = np.ones_like(amplitudes, dtype = bool)

    for f in range(amplitudes.shape[1]):
        for pol in range(amplitudes.shape[3]):
            mask[:,f,:,pol][amplitudes[:,f,:,pol]>3*amplitudes_median[f,pol]] = 1
            weight[:,f,:,pol][amplitudes[:,f,:,pol]>3*amplitudes_median[f,pol]] = 0

    amplitudes_masked = np.ma.masked_array(amplitudes, mask = mask)
    amplitudes_plot = np.copy(amplitudes)
    amplitudes_plot[mask] = 0

    logging.info(f"Plotting flagged apmplitudes")
    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((amplitudes_plot[:,:,antenna,0]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'plasma', origin = 'lower', vmin = vmin[0], vmax = vmax[0])
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Amplitudes flagged xx', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/amplitudes_flagged_xx.png', format = 'png', bbox_inches = 'tight')

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((amplitudes_plot[:,:,antenna,1]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'plasma', origin = 'lower', vmin = vmin[1], vmax = vmax[1])
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Amplitudes flagged yy', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/amplitudes_flagged_yy.png', format = 'png', bbox_inches = 'tight')

    logging.info(f"Calculating the bandpass")
    bandpass = np.nanmean(amplitudes_masked, axis = 0)

    logging.info(f"Plotting the bandpass")
    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].plot(freq/10**6,(bandpass[:,antenna,0]), label = 'xx')
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].plot(freq/10**6,(bandpass[:,antenna,1]), label = 'yy')
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Amplitude [a.u.]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].legend()
    subfigs.suptitle('Bandpass', fontsize=12)
    plt.savefig(f'{folder}/bandpass.png', format = 'png', bbox_inches = 'tight')

    logging.info(f"Saving the solutions on a h5 file ready to apply")
    h5r=h5py.File(infile, 'r')
    with h5py.File(outfile, 'w') as h5w:
        for obj in h5r.keys():        
            h5r.copy(obj, h5w )       
    h5r.close()
    f = h5py.File(outfile, 'r+')
    amplitudes = np.array(f['sol000/amplitude000/val'])
    data = np.ones_like(amplitudes)
    for t in range(data.shape[0]):
        for ant in range(data.shape[2]):
            data[t,:,ant,0,:] = bandpass[:,ant,:]
    f['sol000/amplitude000/val'][:] = data
    f.close()



def func(freq, delay, tec, offset, tec3):
     '''
     A function that make a theoretical curve for clock, tec and tec3 effects.
     '''
     c = 3*10**8
     model = 2*np.pi*freq*(delay*10**(-9)) + -8.4479745e9/freq*tec + offset + (c/freq)**3*(tec3)
     fit = np.angle(np.exp(1j*model))
     return fit

def calculate_clockTEC3(infile, folder, outfile):
    '''
    Function that separates the contribution of the clock from that of the ionosphere (clocktec separation)
    using a new function that works with phasors, and not with phases. This makes the handling of phasewrap easier. 
    This function calculates solutions for all the directions in ddecal (https://dp3.readthedocs.io/en/latest/steps/DDECal.html)).

    **Parameters**
    infile: string
        The name of the gain solutions h5 file. The solutions have to be 
        calculated in diagonal mode because the clocktec separation requires a diagonal matrix.
    outfile: string
        The name of the h5 file where the solutions should be saved in the table phase000
    folder: string
        The folder where the plots and solutions will be solved

    **Returns**
    It does not return any values but it saves a folder called {folder} 
    where the phase, amplitude, and phase fitting can be looked at.
    It saves a h5 file called {outfile} where the new phase values are written and will be applied to the data.
    '''

    logging.info(f"Reading and plotting the solutions. The plots are in the folder {folder}")
    f = h5py.File(infile)
    freq = np.array(((f['sol000/phase000/freq'])))
    phases = np.array(((f['sol000/phase000/val'])))
    amplitudes = np.array(((f['sol000/amplitude000/val'])))
    weight = np.array(((f['sol000/amplitude000/weight'])))
    antennas = np.array(((f['sol000/phase000/ant'])))
    time = f['sol000/phase000/time']

    freq = np.array(freq)
    phases = np.array(phases)
    amplitudes = np.array(amplitudes)
    time = np.array(time)
    weight = np.array(weight)

    phases = phases[:,:,:,0,:]
    amplitudes = amplitudes[:,:,:,0,:]

    if not os.path.exists(folder):
        os.mkdir(folder)

    phases_ref = np.empty_like(phases)
    for ant in range(phases.shape[2]):   
        phases_ref[:,:,ant,:] = np.angle(np.exp(1j*phases[:,:,ant,:])*np.exp(-1j*phases[:,:,0,:]))

    logging.info(f"The phases have a shape {phases_ref.shape}")

    rows = 3
    subplots_num = int(phases.shape[2]/rows)

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((phases_ref[:,:,antenna,0]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'hsv', origin = 'lower', vmin = -np.pi, vmax = np.pi)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Phases xx', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/phases_xx.png', format = 'png', bbox_inches = 'tight')

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((phases_ref[:,:,antenna,1]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'hsv', origin = 'lower', vmin = -np.pi, vmax = np.pi)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Phases yy', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/phases_yy.png', format = 'png', bbox_inches = 'tight')

    med = np.nanmedian(amplitudes, axis = (0,1,2))
    mad = np.nanmedian (np.absolute(amplitudes - med), axis = (0,1,2))
    vmin = med - 3*mad
    vmax = med + 3*mad

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((amplitudes[:,:,antenna,0]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'plasma', origin = 'lower', vmin = vmin[0], vmax = vmax[0])
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Amplitudes xx', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/amplitudes_xx.png', format = 'png', bbox_inches = 'tight')

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((amplitudes[:,:,antenna,1]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'plasma', origin = 'lower', vmin = vmin[1], vmax = vmax[1])
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Amplitudes yy', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/amplitudes_yy.png', format = 'png', bbox_inches = 'tight')

    logging.info(f"Fitting for clock, tec, and tec3 with own function. Only works for core stations with delays less than 15 ns.")
    params = np.zeros((phases_ref.shape[0],phases_ref.shape[2],phases_ref.shape[3], 4))

    for t in range(phases_ref.shape[0]):
        for pol in range(phases_ref.shape[3]):
            for ant in range(phases_ref.shape[2]):     
                phases_ref[t,np.isnan(phases_ref[t,:,ant,pol])==True,ant,pol] = 0
                param_bounds=([-15,-0.01,-2*np.pi,-50],[15,0.01,2*np.pi,50])
                popt, _ = curve_fit(func, freq, phases_ref[t,:,ant,pol],bounds=param_bounds)
                params[t,ant,pol] = popt

    t = 2
    c = 3*10**8
    phase_to_fit = np.zeros_like(phases_ref)

    for f in range(freq.shape[0]):
        phase_to_fit[:,f,:,:] = 2*np.pi*freq[f]*(params[:,:,:,0]*10**(-9)) -8.4479745e9/freq[f]*(params[:,:,:,1]) + params[:,:,:,2] + (c/freq[f])**3*(params[:,:,:,3])
    fit = np.angle(np.exp(1j*phase_to_fit))
 
    logging.info(f"Plotting the resulting fits")
    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases_ref.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].plot(freq/10**6,(phases_ref[t,:,antenna,0]))
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].plot(freq/10**6,(fit[t,:,antenna,0]))
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Phase [rad]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Fit one time slot xx', fontsize=12)
    plt.savefig(f'{folder}/phases_oneTime_fit_xx.png', format = 'png', bbox_inches = 'tight')

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases_ref.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].plot(freq/10**6,(phases_ref[t,:,antenna,1]))
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].plot(freq/10**6,(fit[t,:,antenna,1]))
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Phase [rad]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Fit one time slot xx', fontsize=12)
    plt.savefig(f'{folder}/phases_oneTime_fit_yy.png', format = 'png', bbox_inches = 'tight')


    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases_ref.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((fit[:,:,antenna,0]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'hsv', origin = 'lower', vmin = -np.pi/2, vmax = np.pi/2)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Fit phases xx', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/fit_phases_xx.png', format = 'png', bbox_inches = 'tight')

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases_ref.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((fit[:,:,antenna,1]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'hsv', origin = 'lower', vmin = -np.pi/2, vmax = np.pi/2)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Fit phases yy', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/fit_phases_yy.png', format = 'png', bbox_inches = 'tight')


    logging.info(f"Ploting the residuals")
    residuals = np.angle(np.exp(1j*(phases_ref-fit)))
    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases_ref.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((residuals[:,:,antenna,0]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'hsv', origin = 'lower', vmin = -np.pi/4, vmax = np.pi/4)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Residuals xx', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/residual_phases_xx.png', format = 'png', bbox_inches = 'tight')

    fig = plt.figure(layout='constrained', figsize=(11, 5.5))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(3, 8, sharey=True)
    for antenna in range(phases_ref.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((residuals[:,:,antenna,1]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'hsv', origin = 'lower', vmin = -np.pi/4, vmax = np.pi/4)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Residuals yy', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/residuals_phases_yy.png', format = 'png', bbox_inches = 'tight')

    logging.info(f"Flagging bad amplitudes")
    amplitudes_median = np.nanmedian(amplitudes, axis=(0,2))
    mask = np.zeros_like(amplitudes, dtype = bool)
    weight = weight[:,:,:,0,:]

    for f in range(amplitudes.shape[1]):
        for pol in range(amplitudes.shape[3]):
            mask[:,f,:,pol][amplitudes[:,f,:,pol]>2*amplitudes_median[f,pol]] = 1
            weight[:,f,:,pol][amplitudes[:,f,:,pol]>2*amplitudes_median[f,pol]] = 0

    amplitudes_plot = np.copy(amplitudes)
    amplitudes_plot[mask] = 0

    logging.info(f"Ploting flagged amplitudes")
    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((amplitudes_plot[:,:,antenna,0]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'plasma', origin = 'lower', vmin = vmin[0], vmax = vmax[0])
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Amplitudes flagged xx', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/amplitudes_flagged_xx.png', format = 'png', bbox_inches = 'tight')

    fig = plt.figure(layout='constrained', figsize=(20,10))
    subfigs = fig.subfigures(1,1, wspace=0.02)
    axs = subfigs.subplots(rows, subplots_num, sharey=True)
    for antenna in range(phases.shape[2]):
        pc = axs[int(antenna/subplots_num),antenna%subplots_num].imshow((amplitudes_plot[:,:,antenna,1]).T, aspect= 'auto', extent=(0, (time[-1]-time[0])/3600,freq[0]/10**6, freq[-1]/10**6), cmap = 'plasma', origin = 'lower', vmin = vmin[1], vmax = vmax[1])
        axs[int(antenna/subplots_num),antenna%subplots_num].set_xlabel('Observing time [s]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_ylabel('Frequency [MHz]', fontsize = 9)
        axs[int(antenna/subplots_num),antenna%subplots_num].set_title(str(antennas[antenna]), fontsize = 9)
    subfigs.suptitle('Amplitudes flagged yy', fontsize=12)
    subfigs.colorbar(pc, shrink=0.6, ax=axs, location='bottom')
    plt.savefig(f'{folder}/amplitudes_flagged_yy.png', format = 'png', bbox_inches = 'tight')

    logging.info(f"Saving the solutions on a h5 file ready to apply")
    h5r=h5py.File(infile, 'r')
    with h5py.File(outfile, 'w') as h5w:
        for obj in h5r.keys():        
            h5r.copy(obj, h5w )       
    h5r.close()
    f = h5py.File(outfile, 'r+')

    phases = np.array(f['sol000/phase000/val'])
    weights = np.array(f['sol000/amplitude000/weight'])
    amplitudes = np.array(f['sol000/amplitude000/val'])

    phases[:,:,:,0,:] = fit
    weights[:,:,:,0,:] = weight
    amplitudes[:,:,:,0,:] = 1

    f['sol000/amplitude000/weight'][:] = weights
    f['sol000/amplitude000/val'][:] = amplitudes
    f['sol000/phase000/val'][:] = phases
    
    f.close()



def flag_demix(dirs_cal, dirs_target, root_name_cal, root_name_target, demix_A_team, channels_avg, time_avg, folder_logs, start, ind):
    '''
    Function that flags the raw LOFAR data and removes the bright A-team sources (CasA, CygA, HerA, VirA and TauA). 
    The function runs the processes in parallel for the calibrator and target so the output might be confusing.

    **Parameters**
    dirs_cal: numpy array of strings
        The name of the MS files of the calibrator.

    dirs_target: numpy array of strings
        The name of the MS files of the target.

    root_name_cal: string
        The common string for all the calibrator MS files (this is calculated in the main function).
    
    root_name_target: string
        The common string for all the target MS files (this is calculated in the main function).

    demix_A_team: bool
        If the A-team sources should be removed or the data only averaged.
    
    channels_avg: int
        The number of channels that have to be averged during demixing or during averaging.

    time_avg: int
        The number of time slots that have to be averged during demixing or during averaging.
    
    folder_logs: str
        The name of the folder where the logs files will be saved.

    start: str
        The step that the pipeline has to start with

    ind: 0 or 1
        This specifies if the calibrator or the target files are being flagged/demixed.

    ***Returns***
    Does not return anything, but saves the flagged and demixed files sepparately.
    '''
    if ind == 0:
        if start == 'Flagging':
            for i  in range(0,dirs_cal.shape[0]):
            # Flag data
                logging.info(f"Started flagging the calibrator file {dirs_cal[i]}")
                file = open("flag_calibrator.parset","w")
                L = dedent(f'''
                msin = {dirs_cal[i]}
                msout = {dirs_cal[i]}.flag
                msout.overwrite = True
                steps=[preflagger,aoflagger]
                preflagger.corrtype=auto
                preflagger.type=preflagger
                aoflagger.autocorr=F
                aoflagger.keepstatistics=T
                aoflagger.memorymax=0
                aoflagger.memoryperc=98
                aoflagger.overlapperc=2
                aoflagger.timewindow=10
                aoflagger.type=aoflagger
                ''')
                file.write(L)
                file.close() 

                command = f'DP3 flag_calibrator.parset > {folder_logs}/flag_{dirs_cal[i]}'
                logging.info(f"Running the command: {command}")
                run_process(command)
                start = 'Flagging'
                np.save('start.npy', np.array(start))
        if start == 'Flagging':
            start = 'Demixing/Averaging'
            np.save('start.npy', np.array(start))

        if start == 'Demixing/Averaging':
            for i  in range(0,dirs_cal.shape[0],2):
                # Demix and/or average data
                if demix_A_team == True:
                    logging.info(f"Started demixing the calibrator file {dirs_cal[i]}")
                    file = open("demix_calibrator.parset","w")
                    L = dedent(f'''
                    msin = [{dirs_cal[i]}.flag,{dirs_cal[i+1]}.flag]
                    msout = {dirs_cal[i]}.flag.avg
                    msout.overwrite = True
                    steps=[demixer]
                    demixer.freqstep={channels_avg}
                    demixer.timestep={time_avg}
                    demixer.demixfreqstep={channels_avg}
                    demixer.demixtimestep={time_avg}
                    demixer.skymodel = Ateam_LBA_CC.skymodel
                    demixer.subtractsources=[CasA, CygA, VirA, HerA, TauA]
                    demixer.type=demixer
                    demixer.maxiter = 4000
                    demixer.ntimechunk = 64
                    demixer.target = 3c295
                    demixer.instrumentmodel = instrument_{dirs_cal[i][:-3]}
                    ''')
                    file.write(L)
                    file.close() 

                    command = f'DP3 demix_calibrator.parset > {folder_logs}/demix_{dirs_cal[i]}'
                    logging.info(f"Running the command: {command}")
                    run_process(command)
                    start = 'Demixing/Averaging'
                    np.save('start.npy', np.array(start))

                else:
                    logging.info(f"Started averaging the calibrator file {dirs_cal[i]}")
                    file = open("average_calibrator.parset","w")

                    L = dedent(f'''
                    msin = {dirs_cal[i]}.flag
                    msout = {dirs_cal[i]}.flag.avg
                    msout.overwrite = True
                    steps=[averager]
                    averager.freqstep={channels_avg}
                    averager.timestep={time_avg}
                    averager.type=averager
                    ''')
                    file.write(L)
                    file.close() 

                    command = f'DP3 average_calibrator.parset > {folder_logs}/avg_{dirs_cal[i]}'
                    logging.info(f"Running the command: {command}")
                    run_process(command)

                    start = 'Demixing/Averaging'
                    np.save('start.npy', np.array(start))
        if start == 'Demixing/Averaging':
            start = 'Merging'
            np.save('start.npy', np.array(start))

        
        if start == 'Merging':
            logging.info(f"Started merging the calibrator files")
            file = open("merge_calibrator.parset","w")
            L = dedent(f'''
            msin = {root_name_cal}*.MS.flag.avg
            msout = {root_name_cal[:-3]}.MS.flag.avg
            msout.overwrite = True
            steps = []
            ''')
            file.write(L)
            file.close() 

            command = f'DP3 merge_calibrator.parset > {folder_logs}/merge_calibrator'
            logging.info(f"Running the command: {command}")
            run_process(command)


    else:
        if start == 'Flagging':
            for i  in range(dirs_target.shape[0]):
                logging.info(f"Started flagging the target file {dirs_target[i]}")
                file = open("flag_target.parset","w")
                L = dedent(f'''
                msin = {dirs_target[i]}
                msout = {dirs_target[i]}.flag
                msout.overwrite = True
                steps=[preflagger,aoflagger]
                preflagger.corrtype=auto
                preflagger.type=preflagger
                aoflagger.autocorr=F
                aoflagger.count.save=F
                aoflagger.keepstatistics=T
                aoflagger.memorymax=0
                aoflagger.memoryperc=98
                aoflagger.overlapperc=2
                aoflagger.timewindow=10
                aoflagger.type=aoflagger
                ''')
                file.write(L)
                file.close() 

                command = f'DP3 flag_target.parset > {folder_logs}/flag_{dirs_target[i]}'
                logging.info(f"Running the command: {command}")
                run_process(command)
                start = 'Flagging'
                np.save('start.npy', np.array(start))

        if start == 'Flagging':
            start = 'Demixing/Averaging'
            np.save('start.npy', np.array(start))

        if start == 'Demixing/Averaging':
            for i  in range(0,dirs_target.shape[0],2):
                if demix_A_team == True:
                    logging.info(f"Started demixing the target file {dirs_target[i]}")
                    file = open("demix_target.parset","w")
                    L = dedent(f'''
                    msin = [{dirs_target[i]}.flag,{dirs_target[i+1]}.flag]
                    msout = {dirs_target[i]}.flag.avg
                    msout.overwrite = True
                    steps=[demixer]
                    demixer.freqstep={channels_avg}
                    demixer.timestep={time_avg}
                    demixer.demixfreqstep={channels_avg}
                    demixer.demixtimestep={time_avg}
                    demixer.skymodel = Ateam_LBA_CC.skymodel
                    demixer.subtractsources=[CasA, CygA, VirA, HerA, TauA]
                    demixer.type=demixer
                    demixer.target = 3c295
                    demixer.maxiter = 4000
                    demixer.ntimechunk = 64
                    demixer.instrumentmodel = instrument_{dirs_target[i][:-3]}
                    ''')
                    file.write(L)
                    file.close() 

                    command = f'DP3 demix_target.parset > {folder_logs}/demix_{dirs_target[i]}'
                    logging.info(f"Running the command: {command}")
                    run_process(command)
                    start = 'Demixing/Averaging'
                    np.save('start.npy', np.array(start))

                else:
                    logging.info(f"Started averaging the target file {dirs_target[i]}")
                    file = open("average_target.parset","w")
                    L = dedent(f'''
                    msin = {dirs_target[i]}.flag
                    msout = {dirs_target[i]}.flag.avg
                    msout.overwrite = True
                    steps=[averager]
                    averager.freqstep={channels_avg}
                    averager.timestep={time_avg}
                    averager.type=averager
                    ''')
                    file.write(L)
                    file.close() 

                    command = f'DP3 average_target.parset  > {folder_logs}/avg_{dirs_target[i]}'
                    logging.info(f"Running the command: {command}")
                    run_process(command)
                    start = 'Demixing/Averaging'
                    np.save('start.npy', np.array(start))
        if start == 'Demixing/Averaging':
            start = 'Merging'
            np.save('start.npy', np.array(start))
            
        if start == 'Merging':
            logging.info(f"Started merging the target files")
            file = open("merge_target.parset","w")
            L = dedent(f'''
            msin = {root_name_target}*.MS.flag.avg
            msout = {root_name_target[:-3]}.MS.flag.avg
            msout.overwrite = True
            steps = []
            ''')
            file.write(L)
            file.close() 

            command = f'DP3 merge_target.parset  > {folder_logs}/merge_target'
            logging.info(f"Running the command: {command}")
            run_process(command)



def make_model_fulljones(nchans):
    '''
    A function that uses the selfcalibrated images to make a frequency dependent model for the calibration of the polarization leakage. 
    The pipeline should make the images automatically.

    **Parameters**
    nchans: int
        The number of channels used for imaging

    ***Returns***
    Nothing, but it saves a model that is frequency dependent.
    '''
    img_list = []

    for ind in range(nchans):
        if ind<10:
            img = fits.getdata('selfcal_small-000'+str(ind)+'-I-image.fits')
        else:
            img = fits.getdata('selfcal_small-00'+str(ind)+'-I-image.fits')
        img_list.append(img)

    halfchans = int(nchans/2)+1
    if halfchans < 10:
        file = f'selfcal_small-000{halfchans}-I-image.fits'
    else:
        file = f'selfcal_small-00{halfchans}-I-image.fits'
    data, header = fits.getdata(file, header=True)

    header["NAXIS3"] = nchans
    header["CRPIX3"] = int(nchans/2)+1

    # Cast the list into a numpy array
    img_array = np.array(img_list)
    img_array = np.swapaxes(img_array,0,2)[0]

    # Save the array as fits - it will save it as an image cube
    fits.writeto('mycube_selfcal.fits', img_array,header, overwrite=True)

    filename = 'mycube_selfcal.fits'
    img = bdsf.process_image(filename, thresh_isl = 3, thresh_pix = 5, spectralindex_do = True)
    img.write_catalog(outfile = 'sources_target_field_selfcal.skymodel', bbs_patches = 'single', format = 'bbs', catalog_type = 'gaul', clobber = True)


    
def create_fulljones_solutions(infile, outfile):
    '''
    A function that prepares the soltutions for the polarization leakage correction. 
    This function only keeps the anti-diagonal values and puts 1 in rest.

    ***Parameters***
    infile: string
        The name of the gain solutions h5 file. The solutions have to be 
        calculated in fulljones mode because the polarization leakage is in the antidiagonal terms.
    outfile: string
        The name of the h5 file where the solutions should be saved in the table phase000.

    ***Returns***
    Nothing, but it creates and updates the file {outfile}.
    '''

    h5r=h5py.File(infile)
    with h5py.File(outfile,'w') as h5w:
        for obj in h5r.keys():
            h5r.copy(obj, h5w )
    h5r.close()
    f = h5py.File(outfile,'r+')

    phases = np.array(f['sol000/phase000/val'])
    amplitudes = np.array(f['sol000/amplitude000/val'])


    phases[:,:,:,:,3] = 0
    phases[:,:,:,:,0] = 0

    amplitudes[:,:,:,:,3] = 1
    amplitudes[:,:,:,:,0] = 1

    f['sol000/phase000/val'][:] = phases
    f['sol000/amplitude000/val'][:] = amplitudes
    f.close()



def main(args):

    # make a log folder and a log file
    now = datetime.datetime.now()
    folder_logs = f'logs_{now}'
    folder_logs = folder_logs.replace(' ','_')

    if not os.path.exists(folder_logs):
        os.mkdir(folder_logs)

    logging.basicConfig(level=logging.DEBUG, filename=f"{folder_logs}/general_logs", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(f"The arguments are {args}")

    # read the function's arguments
    path = args.path
    model_cal = args.model_cal
    model_target = args.model_target
    demix_A_team = args.demix_A_team
    sap_target = args.sap_target
    channels_avg = args.channels_avg
    time_avg = args.time_avg
    channels_image = args.channels_image
    time_const_target = args.time_selfcal_target
    subbands_const_target = args.channels_selfcal_target
    dynspec = args.dynspec
    time_slots_avg_dynspec = args.time_slots_avg_dynspec
    freq_slots_avg_dynspec = args.freq_slots_avg_dynspec
    start = args.start

    if start == 'Nothing':
        if not os.path.exists('start.npy'):
            start = 'Flagging'
        else:
            start = np.load('start.npy')
    
    # read MS files
    dirs = []
    # Iterate directory
    for file in os.listdir(path):
        # only append ms directory names
        if file.endswith('_uv.MS'):
            dirs.append(file)
    dirs.sort()
    logging.info(f"The files that are found are {dirs}")


    # separate calibrator from target's files
    sap_calibrator = 1-sap_target
    dirs_cal = []
    dirs_target = []
    for dir_name in dirs:
       if 'SAP00'+str(sap_calibrator) in dir_name:
           dirs_cal.append(dir_name)
       elif 'SAP00'+str(sap_target) in dir_name:
           dirs_target.append(dir_name)

    dirs_target = np.array(dirs_target)
    dirs_cal = np.array(dirs_cal)

    # Find the root of the names (without the SB{subband}_uv). This can be buggy depending on the observation's id number
    root_name_cal = dirs_cal[0][:17]
    root_name_target = dirs_target[0][:17]
    logging.info(f"The root of the calibrator is: {root_name_cal}")
    logging.info(f"The root of the target is: {root_name_target}")

    # Start flagging and demixing/averaging the files. This takes very long. The target and calibrator are done in parallel as it's a bit faster.
    if start == 'Flagging' or start == 'Demixing/Averaging' or start == 'Merging':
        logging.info(f"Flagging, demixing/averaging and/or merging the files")
        pool = multiprocessing.Pool()
        pool = multiprocessing.Pool(processes=2)
        inputs = np.arange(0, 2)
        pool.map(partial(flag_demix, dirs_cal, dirs_target, root_name_cal, root_name_target, demix_A_team, channels_avg, time_avg, folder_logs, start), inputs)
        start = np.load('start.npy')

    if start == 'Merging':
        start = 'SecondFlagging'
        np.save('start.npy', start)
    # An extra round of flagging to remove borad band effects
    if start == 'SecondFlagging':
        logging.info(f"Second round of flagging after merging the files. If you need to remake this step, you should start from merge to have the initial flags in the files.")
        logging.info(f"The calibrator is flagging now")
        tab = table(f'{root_name_cal[:-3]}.MS.flag.avg', readonly=False)
        dynspec = np.array(tablecolumn(tab,'DATA'))
        flag = np.array(tablecolumn(tab,'FLAG'))
        np.save(f'DONOTDELETE_{root_name_cal[:-3]}_initial_flag', flag)


        dynspec[flag] = np.nan
        for f in range(0,dynspec.shape[1]):
            for pol in range(dynspec.shape[2]):
                flag[np.abs(dynspec[:,f,pol])>2*np.nanmedian(np.abs(dynspec[:,f,pol])),f,pol] = 1

        tab.putcol("FLAG",flag)
        np.save('better_flag', flag)
        tab.flush()
        tab.close()

        logging.info(f"The target is flagging now")
        tab = table(f'{root_name_target[:-3]}.MS.flag.avg', readonly=False)
        dynspec = np.array(tablecolumn(tab,'DATA'))
        flag = np.array(tablecolumn(tab,'FLAG'))
        np.save(f'DONOTDELETE_{root_name_target[:-3]}_initial_flag', flag)

        dynspec[flag] = np.nan
        for f in range(0,dynspec.shape[1]):
            for pol in range(dynspec.shape[2]):
                flag[np.abs(dynspec[:,f,pol])>2*np.nanmedian(np.abs(dynspec[:,f,pol])),f,pol] = 1

        tab.putcol("FLAG",flag)
        np.save('better_flag', flag)
        tab.flush()
        tab.close()
    if start == 'SecondFlagging':
        start = 'PolAlign'
        np.save('start.npy', start)

    # Calculate polAlign using Losoto
    if start == 'PolAlign':
        logging.info(f"Calculate polarization missalignment using the calibrator")
        file = open("calibrate_cal.parset","w")

        L = dedent(f'''
        msin = {root_name_cal[:-3]}.MS.flag.avg
        msout = .
        steps = [cal]
        cal.type = ddecal
        cal.mode = rotation+diagonal
        cal.h5parm = rot_diag_forPolAlign_cal.h5
        cal.maxiter = 50
        cal.solint = 1
        cal.nchan = 1
        cal.sourcedb = {model_cal}
        cal.usebeammodel = True
        ''')
        file.write(L)
        file.close() 

        command = f'DP3 calibrate_cal.parset > {folder_logs}/calibrate_polAlign'
        logging.info(f"Running the command: {command}")
        run_process(command) 

        logging.info(f"Entering the plotting function")
        calculatePolAlign('rot_diag_forPolAlign_cal.h5', 'PolAlignPlots')

        logging.info(f"Apply polarization missalignment on calibrator")
        file = open("apply_cal.parset","w")
        L = dedent(f'''
        msin = {root_name_cal[:-3]}.MS.flag.avg
        msout = {root_name_cal[:-3]}.MS.flag.avg.polAlign
        msout.overwrite=True
        steps = [apply,applyBeam]
        apply.type = applycal
        apply.parmdb = rot_diag_forPolAlign_cal.h5
        apply.steps = [phase]
        apply.phase.correction=phasediff
        applyBeam.type = applybeam
        ''')
        file.write(L)
        file.close() 

        command = f'DP3 apply_cal.parset > {folder_logs}/apply_polAlign_calibrator'
        logging.info(f"Running the command: {command}")
        run_process(command) 
    
        logging.info(f"Apply polarization missalignment on target")
        file = open("apply_target.parset","w")
        L = dedent(f'''
        msin = {root_name_target[:-3]}.MS.flag.avg
        msout = {root_name_target[:-3]}.MS.flag.avg.polAlign
        msout.overwrite=True
        steps = [apply,applyBeam]
        apply.type = applycal
        apply.parmdb = rot_diag_forPolAlign_cal.h5
        apply.steps = [phase]
        apply.phase.correction=phasediff
        applyBeam.type = applybeam
        ''')
        file.write(L)
        file.close() 

        command = f'DP3 apply_target.parset > {folder_logs}/apply_polAlign_target'
        logging.info(f"Running the command: {command}")
        run_process(command) 

    if start == 'PolAlign':
        start = 'Bandpass'
        np.save('start.npy', start)


    # Calculate bandpass - use own code: median with RFI ignored
    if start == 'Bandpass':
        logging.info(f"Calculate bandpass using the calibrator")
        file = open("calibrate_cal.parset","w")
        L = dedent(f'''
        msin = {root_name_cal[:-3]}.MS.flag.avg.polAlign
        msout = .
        steps = [cal]
        cal.type = ddecal
        cal.mode = diagonal
        cal.h5parm = diag_forBP_cal.h5
        cal.maxiter = 100
        cal.solint = 1
        cal.nchan = 1
        cal.sourcedb = {model_cal}   
        ''')
        file.write(L)
        file.close() 

        command = f'DP3 calibrate_cal.parset > {folder_logs}/calibrate_bandpass'
        logging.info(f"Running the command: {command}")
        run_process(command) 

        logging.info(f"Entering the solution processing and plotting function")
        calculate_bandpass('diag_forBP_cal.h5', 'diag_ApplyBP_cal.h5','BPPlots')

        logging.info(f"Apply bandpass on calibrator")
        file = open("apply_cal.parset","w")

        L = dedent(f'''
        msin = {root_name_cal[:-3]}.MS.flag.avg.polAlign
        msout = {root_name_cal[:-3]}.MS.flag.avg.polAlign.BP
        msout.overwrite=True
        steps = [apply]
        apply.type = applycal
        apply.parmdb = diag_ApplyBP_cal.h5
        apply.steps = [ampl]
        apply.ampl.correction=amplitude000
        ''')
        file.write(L)
        file.close() 

        command = f'DP3 apply_cal.parset > {folder_logs}/apply_bandpass_calibrator'
        logging.info(f"Running the command: {command}")
        run_process(command) 

    
        logging.info(f"Apply bandpass on target")
        file = open("apply_target.parset","w")
        L = dedent(f'''
        msin = {root_name_target[:-3]}.MS.flag.avg.polAlign
        msout = {root_name_target[:-3]}.MS.flag.avg.polAlign.BP
        msout.overwrite=True
        steps = [apply]
        apply.type = applycal
        apply.parmdb = diag_ApplyBP_cal.h5
        apply.steps = [ampl]
        apply.ampl.correction=amplitude000
        ''')
        file.write(L)
        file.close() 

        command = f'DP3 apply_target.parset > {folder_logs}/apply_bandpass_target'
        logging.info(f"Running the command: {command}")
        run_process(command) 

    if start == 'Bandpass':
        start = 'ClockTEC3'
        np.save('start.npy', start)


    # Calibrate for clock/TEC with own code. Losoto doesn't work for this type of data.
    if start == 'ClockTEC3':
        logging.info(f"Calculate Clock/TEC/TEC3 effects using the calibrator")
        file = open("calibrate_cal.parset","w")
        L = dedent(f'''
        msin = {root_name_cal[:-3]}.MS.flag.avg.polAlign.BP
        msout = .
        steps = [cal]
        cal.type = ddecal
        cal.mode = diagonal
        cal.h5parm = diag_forClockTEC3_cal.h5
        cal.maxiter = 100
        cal.smoothnessconstraint = 200000
        cal.solint = 1
        cal.nchan = 1
        cal.sourcedb = {model_cal}
        ''')
        file.write(L)
        file.close() 

        command = f'DP3 calibrate_cal.parset > {folder_logs}/calibrate_clock_tec_tec3'
        logging.info(f"Running the command: {command}")
        run_process(command) 


        logging.info(f"Entering the solution processing and plotting function")
        calculate_clockTEC3(infile='diag_forClockTEC3_cal.h5', outfile='diag_ApplyClockTEC3_cal.h5', folder = 'ClockTEC3Plots')

        logging.info(f"Apply Clock/TEC/TEC3 effects on calibrator")
        file = open("apply_cal.parset","w")

        L = dedent(f'''
        msin = {root_name_cal[:-3]}.MS.flag.avg.polAlign.BP
        msout = {root_name_cal[:-3]}.MS.flag.avg.polAlign.BP.clockTEC3
        msout.overwrite=True
        steps = [apply]
        apply.type = applycal
        apply.parmdb = diag_ApplyClockTEC3_cal.h5
        apply.steps = [phase, ampl]
        apply.phase.correction=phase000
        apply.ampl.correction=amplitude000
        ''')
        file.write(L)
        file.close() 

        command = f'DP3 apply_cal.parset > {folder_logs}/apply_clock_tec_tec3_calibrator'
        logging.info(f"Running the command: {command}")
        run_process(command) 

    
        logging.info(f"Apply Clock/TEC/TEC3 effects on target")
        file = open("apply_target.parset","w")
        L = dedent(f'''
        msin = {root_name_target[:-3]}.MS.flag.avg.polAlign.BP
        msout = {root_name_target[:-3]}.MS.flag.avg.polAlign.BP.clockTEC3
        msout.overwrite=True
        steps = [apply]
        apply.type = applycal
        apply.parmdb = diag_ApplyClockTEC3_cal.h5
        apply.steps = [phase, ampl]
        apply.phase.correction=phase000
        apply.ampl.correction=amplitude000
        ''')
        file.write(L)
        file.close() 

        command = f'DP3 apply_target.parset > {folder_logs}/apply_clock_tec_tec3_target'
        logging.info(f"Running the command: {command}")
        run_process(command) 

    if start == 'ClockTEC3':
        start = 'FirstImaging'
        np.save('start.npy', start)

    # Clean and image both calibrator and target field - for now only one image for the whole time
    if start == 'FirstImaging':
        logging.info(f"Image the calibrator in multiple narrow channels")
        command = f'\
        wsclean -name test -multiscale -use-wgridder -niter 1000000 -mgain 0.8 \
        -parallel-deconvolution 2000 -pol iv -scale 15asec -auto-mask 4  -auto-threshold 0.5 -size 4000 4000 -j 128 \
        -channels-out {channels_image}  {root_name_cal[:-3]}.MS.flag.avg.polAlign.BP.clockTEC3 > {folder_logs}/image_calibrator_narrow_bands'
        logging.info(f"Running the command: {command}")
        run_process(command)


        logging.info(f"Image the calibrator in full bandwidth")
        command = f'\
        wsclean -name first_small_cal -multiscale -use-wgridder -niter 1000000 -mgain 0.8 \
        -parallel-deconvolution 2000 -pol iv -scale 15asec -auto-mask 4  -auto-threshold 0.5 -size 4000 4000 -j 128 \
        {root_name_cal[:-3]}.MS.flag.avg.polAlign.BP.clockTEC3 > {folder_logs}/image_calibrator_full_bandwidth'
        logging.info(f"Running the command: {command}")
        run_process(command)

        logging.info(f"Image the target in multiple narrow channels")
        command = f'\
        wsclean -name test -multiscale -use-wgridder -niter 1000000 -mgain 0.8 \
        -parallel-deconvolution 2000 -pol iv -scale 15asec -auto-mask 4  -auto-threshold 0.5 -size 4000 4000 -j 128 \
        -channels-out {channels_image} {root_name_target[:-3]}.MS.flag.avg.polAlign.BP.clockTEC3 > {folder_logs}/first_image_target_narrow_bands'
        logging.info(f"Running the command: {command}")
        run_process(command)

        logging.info(f"Image the target in full bandwidth")
        command = f'\
        wsclean -name first_small_target -multiscale -use-wgridder -niter 1000000 -mgain 0.8 \
        -parallel-deconvolution 2000 -pol iv -scale 15asec -auto-mask 4  -auto-threshold 0.5 -size 4000 4000 -j 128 \
        {root_name_target[:-3]}.MS.flag.avg.polAlign.BP.clockTEC3 > {folder_logs}/first_image_target_full_bandwidth'
        logging.info(f"Running the command: {command}")
        run_process(command)

    if start=='FirstImaging':
        start = 'ThirdFlagging'
        np.save('start.npy', start)

    # Do another flagging step because the RFI is quite bad at these frequencies.
    if start == 'ThirdFlagging':
        logging.info(f"Selfcalibration of the target field")
        name = f'{root_name_target[:-3]}.MS.flag.avg.polAlign.BP.clockTEC3'

        logging.info(f"Another round of flagging for the target field")
        tab = table(f'{name}', readonly=False)  
        dynspec = np.array(tablecolumn(tab,'DATA'))
        flag = np.array(tablecolumn(tab,'FLAG'))
        np.save(f'DONOTDELETE_SELFCAL_{name[:-2]}_initial_flag', flag)

        dynspec[flag] = np.nan
        for f in range(0,dynspec.shape[1]):
            for pol in range(dynspec.shape[2]):
                flag[np.abs(dynspec[:,f,pol])>2*np.nanmedian(np.abs(dynspec[:,f,pol])),f,pol] = 1

        tab.putcol("FLAG",flag)
        tab.flush()
    tab.close()

    if start == 'ThirdFlagging':
        start = 'TEC3'
        np.save('start.npy', start)


    # Calculate TEC corrections
    if start == 'TEC3':
        logging.info(f"Calculate TEC/TEC3 effects on the target field using selfcalibration")
        file = open("calibrate_selfcal.parset","w")
        L = dedent(f'''
        msin = {name}
        msout = .
        steps = [cal]
        cal.type = ddecal
        cal.sourcedb = {model_target}
        cal.mode = diagonal
        cal.h5parm = diag_forSelfcal.h5
        cal.maxiter = 100
        cal.solint = {time_const_target}
        cal.nchan = {subbands_const_target}
        ''')
        file.write(L)
        file.close() 

        command = f'DP3 calibrate_selfcal.parset > {folder_logs}/selfcalibrate_tec'
        logging.info(f"Running the command: {command}")
        run_process(command) 

        logging.info(f"Entering the solution processing and plotting function")
        calculate_clockTEC3(infile='diag_forSelfcal.h5', outfile='diag_ApplySelfcal.h5', folder='ClockTEC3Plots_selfcal')

        logging.info(f"Apply tec on the target field")
        file = open("apply_selfcal.parset","w")

        L = dedent(f'''
        msin = {name}
        msout = {name}.selfcal
        msout.overwrite=True
        steps = [apply]
        apply.type = applycal
        apply.parmdb = diag_ApplySelfcal.h5
        apply.steps = [phase,ampl]
        apply.phase.correction=phase000
        apply.ampl.correction=amplitude000
        ''')
        file.write(L)
        file.close() 

        command = f'DP3 apply_selfcal.parset > {folder_logs}/apply_tec'
        logging.info(f"Running the command: {command}")
        run_process(command)

    if start == 'TEC3':
        start = 'SecondImaging'
        np.save('start.npy', start)

    # Clean and image the selfcalibreted target field - requried for the polarization misalignment calibration
    if start == 'SecondImaging':
        logging.info(f"Image the selfcalibrated target in multiple narrow channels")
        command = f'\
        wsclean -name selfcal_small -multiscale -use-wgridder -niter 1000000  -mgain 0.8 \
        -parallel-deconvolution 2000 -pol iv -mem 90 -scale 15asec -auto-mask 4  -auto-threshold 0.5 -size 4000 4000 -j 128 \
        -channels-out 20 {name}.selfcal > {folder_logs}/selfcal_image_target_narrow_bands'
        logging.info(f"Running the command: {command}")
        run_process(command)


        logging.info(f"Image the selfcalibrated target in full bandwidth")
        command = f'\
        wsclean -name selfcal_small -multiscale -use-wgridder -niter 1000000  -mgain 0.8 \
        -parallel-deconvolution 2000 -pol iv -mem 90 -scale 15asec -auto-mask 4  -auto-threshold 0.5 -size 4000 4000 -j 128 \
        {name}.selfcal > {folder_logs}/selfcal_image_target_full_bandwidth'
        logging.info(f"Running the command: {command}")
        run_process(command)

    if start == 'SecondImaging':
        start = 'PolLeakage'
        np.save('start.npy', start)


    # Fix the polarization leakage
    if start == 'PolLeakage':
        logging.info(f"Calculate the polarization leakage using selfcalibration")
        make_model_fulljones(channels_image)

        file = open("calibrate_selfcal.parset","w")
        L = dedent(f'''
        msin = {name}.selfcal
        msout = .
        steps = [cal]
        cal.type = ddecal
        cal.sourcedb = sources_target_field_selfcal.skymodel
        cal.mode = fulljones
        cal.h5parm = diag_forSelfcal_fulljones.h5
        cal.maxiter = 100
        cal.solint = 0 
        cal.nchan = 3
        ''')
        file.write(L)
        file.close()
        command = f'DP3 calibrate_selfcal.parset > {folder_logs}/selfcalibrate_polarization_leakage'
        logging.info(f"Running the command: {command}")
        run_process(command)

        logging.info(f"Entering the solution processing function")
        create_fulljones_solutions('diag_forSelfcal_fulljones.h5', 'diag_forSelfcal_fulljones_apply.h5') 


        logging.info(f"Apply the polarization leakage correction")
        file = open("apply_selfcal.parset","w")

        L = dedent(f'''
        msin = {name}.selfcal
        msout = {name}.selfcal.fulljones
        msout.overwrite=True
        steps = [apply]
        apply.type = applycal
        apply.parmdb = diag_forSelfcal_fulljones_apply.h5
        apply.correction = fulljones
        ''')
        file.write(L)
        file.close()

        command = f'DP3 apply_selfcal.parset > {folder_logs}/apply_polarization_leakage'
        logging.info(f"Running the command: {command}")
        run_process(command)

    if start == 'PolLeakage':
        start = 'ThirdImaging'
        np.save('start.npy', start)


    # Clean and image the target field final images - for now only one image for the whole time
    if start == 'ThirdImaging':
        logging.info(f"Image the polarization leakage fixed and selfcalibrated target in full bandwidth")
        command = f'\
        wsclean -name best_small -multiscale -use-wgridder -niter 1000000 -mgain 0.8 -weight briggs -0.5\
        -parallel-deconvolution 2000 -pol iv -mem 90 -scale 15asec -auto-mask 4  -auto-threshold 0.5 -size 4000 4000 -j 128 \
        {name}.selfcal.fulljones > {folder_logs}/leakage_fixed_image_target_full_bandwidth'
        logging.info(f"Running the command: {command}")
        run_process(command)

        logging.info(f"Image the polarization leakage fixed and selfcalibrated target in multiple narrow channels Stokes V")
        command = f'\
        wsclean -name best_small-V -multiscale -use-wgridder -niter 1000000 -mgain 0.8 -weight briggs -0.5\
        -parallel-deconvolution 2000 -pol v -mem 90 -scale 15asec -auto-mask 4  -auto-threshold 0.5 -size 4000 4000 -j 128 \
        -channels-out {channels_image}  {name}.selfcal.fulljones > {folder_logs}/leakage_fixed_image_target_narrow_bands'
        logging.info(f"Running the command: {command}")
        run_process(command)

        logging.info(f"Image the polarization leakage fixed and selfcalibrated target in multiple narrow channels Stokes I. This step is essential for Dynspec because it generates the correct model that will later on be subrtacted")
        command = f'\
        wsclean -name best_small-I -multiscale -use-wgridder -niter 1000000 -mgain 0.8 -weight briggs -0.5\
        -parallel-deconvolution 2000 -pol i -mem 90 -scale 15asec -auto-mask 4  -auto-threshold 0.5 -size 4000 4000 -j 128 \
        -channels-out {channels_image}  {name}.selfcal.fulljones > {folder_logs}/leakage_fixed_image_target_narrow_bands'
        logging.info(f"Running the command: {command}")
        run_process(command)


    # If requested, make a dynamic spectra for the phase center
    if dynspec == True:
        if start == 'ThirdImaging':
            start = 'Dynspec'
            np.save('start.npy', start)
        if start == 'Dynspec':
            logging.info(f"Calculating the dynamic spectrum at the phase center")
            name = f'{name}.selfcal.fulljones'

            flag_limit = 10

            name_root = name[49:56]


            if not os.path.exists(f'DynSpec_{name_root}'):
                os.mkdir(f'DynSpec_{name_root}')
            else:
                command = f'rm -r DynSpec_{name_root}'
                logging.info(f"Running the command: {command}")
                run_process(command)
                os.mkdir(f'DynSpec_{name_root}')

            logging.info(f"Copy original MS to subtract stokes I")
            command = f'cp -r {name} dynspec_subtracted_{name_root}.MS'
            logging.info(f"Running the command: {command}")
            run_process(command)

            logging.info(f"Subtract sources. This requires that the Stokes I frequency dependent images are made last in the pipeline.")
            command = f'taql update dynspec_subtracted_{name_root}.MS set DATA=DATA-MODEL_DATA'
            logging.info(f"Running the command: {command}")
            run_process(command)
        
            command = f'mv dynspec_subtracted_{name_root}.MS DynSpec_{name_root}'
            logging.info(f"Running the command: {command}")
            run_process(command)

            logging.info(f"Extract the calibrated and subtracted visibilities")
            tab = table(f'DynSpec_{name_root}/dynspec_subtracted_{name_root}.MS')
            logging.info('Table opened...')
            data_init = np.array(tablecolumn(tab, 'DATA'))
            logging.info('Data extracted...')
            weight_init = np.array(tablecolumn(tab, 'WEIGHT_SPECTRUM'))
            logging.info('Weights extracted...')
            flag_init = np.array(tablecolumn(tab, 'FLAG'))
            logging.info('Flags extracted...')

            logging.info(f"Save the data for faster usage later. This can be quite large")
            np.save(f'DynSpec_{name_root}/data_dynspec.npy', data_init)
            np.save(f'DynSpec_{name_root}/weight_dynspec.npy', weight_init)
            np.save(f'DynSpec_{name_root}/flag_dynspec.npy', flag_init)

            
            dynspec = np.load(f'DynSpec_{name_root}/data_dynspec.npy')
            dynweights = np.load(f'DynSpec_{name_root}/weight_dynspec.npy')
            flag = np.load(f'DynSpec_{name_root}/flag_dynspec.npy')

            dynspec[flag] = np.nan+1j*np.nan
            count = 0
            ant = int(np.sqrt(dynspec.shape[0])*2)
            vis = int(ant*(ant+1)/2)
            spectra = np.zeros((ant,ant,int(dynspec.shape[0]/vis),dynspec.shape[1],dynspec.shape[2]), dtype = np.complex64) # the shape is ant1, ant2, time, freq, pol
            weights = np.zeros((ant,ant,int(dynspec.shape[0]/vis),dynspec.shape[1],dynspec.shape[2]), dtype = np.complex64)
            for t in range(int(dynspec.shape[0]/vis)):
                for i in range(ant):
                    for j in range(i+1):
                        spectra[i,j,t,:,:] = dynspec[count,:,:]
                        weights[i,j,t,:,:] = dynweights[count,:,:]
                        count += 1
            spectra[np.abs(spectra) == 0] = np.nan+np.nan*1j
            weights[np.abs(spectra) == 0] = 0


            logging.info(f"Another round of flagging. This is required to obtain reliable dynamic spectra.")
            spectra_flagged = np.copy(spectra)
            weight_flagged = np.copy(weights)

            mean = np.abs(np.nanmean(spectra_flagged,axis = (0,1,2,4)))

            for f in range(spectra_flagged.shape[3]):
                spectrum = spectra_flagged[:,:,:,f,:]
                weight =  weight_flagged[:,:,:,f,:]

                spectrum[np.abs(spectrum)>flag_limit*mean[f]] = np.nan+1j*np.nan
                weight[np.abs(spectrum)>flag_limit*mean[f]] = np.nan+1j*np.nan

                spectra_flagged[:,:,:,f,:] = spectrum
                weight_flagged[:,:,:,f,:] = weight


            spectra_avg = np.nanmean(spectra_flagged*weight_flagged, axis = (0,1))/np.nanmean(weight_flagged,axis = (0,1))


            spectra_v = np.real(1j*spectra_avg[:,:,1]+1j*spectra_avg[:,:,2])/2
            imag_v = np.imag(1j*spectra_avg[:,:,1]+1j*spectra_avg[:,:,2])/2
            spectra_i= np.real(spectra_avg[:,:,0]+spectra_avg[:,:,3])/2
            imag_i = np.imag(spectra_avg[:,:,0]+spectra_avg[:,:,3])/2

            logging.info(f"Average in time and frequency")
            v_spectra_bins=np.zeros([int(spectra_v.shape[0]/time_slots_avg_dynspec), int(spectra_v.shape[1]/freq_slots_avg_dynspec)])
            v_imag_bins=np.zeros([int(spectra_v.shape[0]/time_slots_avg_dynspec), int(spectra_v.shape[1]/freq_slots_avg_dynspec)])
            i_spectra_bins=np.zeros([int(spectra_v.shape[0]/time_slots_avg_dynspec), int(spectra_v.shape[1]/freq_slots_avg_dynspec)])
            i_imag_bins=np.zeros([int(spectra_v.shape[0]/time_slots_avg_dynspec), int(spectra_v.shape[1]/freq_slots_avg_dynspec)])

            v_curve_time = np.zeros((int(spectra_v.shape[0]/time_slots_avg_dynspec)))
            v_imag_time = np.zeros((int(spectra_v.shape[0]/time_slots_avg_dynspec)))
            i_curve_time = np.zeros((int(spectra_v.shape[0]/time_slots_avg_dynspec)))
            i_imag_time = np.zeros((int(spectra_v.shape[0]/time_slots_avg_dynspec)))


            v_curve_freq = np.zeros((int(spectra_v.shape[1]/freq_slots_avg_dynspec)))
            v_imag_freq = np.zeros((int(spectra_v.shape[1]/freq_slots_avg_dynspec)))
            i_curve_freq = np.zeros((int(spectra_v.shape[1]/freq_slots_avg_dynspec)))
            i_imag_freq = np.zeros((int(spectra_v.shape[1]/freq_slots_avg_dynspec)))

            factor = 0.5

            for i in range(0,spectra_v.shape[0]-time_slots_avg_dynspec,time_slots_avg_dynspec):
                for f in range(0,spectra_v.shape[1]-freq_slots_avg_dynspec,freq_slots_avg_dynspec): 
                    v_spectra_bins[int(i/time_slots_avg_dynspec), int(f/freq_slots_avg_dynspec)]=np.nanmean(spectra_v[i:i+time_slots_avg_dynspec, f+freq_slots_avg_dynspec])
                    v_imag_bins[int(i/time_slots_avg_dynspec), int(f/freq_slots_avg_dynspec)]=np.nanstd(imag_v[i:i+time_slots_avg_dynspec, f+freq_slots_avg_dynspec])/np.sqrt(np.sum(np.where(np.isnan(imag_v[i:i+time_slots_avg_dynspec, f+freq_slots_avg_dynspec])==False)))		
                    i_spectra_bins[int(i/time_slots_avg_dynspec), int(f/freq_slots_avg_dynspec)]=np.nanmean(spectra_i[i:i+time_slots_avg_dynspec, f+freq_slots_avg_dynspec])
                    i_imag_bins[int(i/time_slots_avg_dynspec), int(f/freq_slots_avg_dynspec)]=np.nanstd(imag_i[i:i+time_slots_avg_dynspec, f+freq_slots_avg_dynspec])/np.sqrt(np.sum(np.where(np.isnan(imag_i[i:i+time_slots_avg_dynspec, f+freq_slots_avg_dynspec])==False)))


                    v_curve_time[int(i/time_slots_avg_dynspec)]=np.array([np.nanmean(spectra_v[i:i+time_slots_avg_dynspec, :])])
                    v_imag_time[int(i/time_slots_avg_dynspec)]=np.array([np.nanstd(imag_v[i:i+time_slots_avg_dynspec, :])/np.sqrt(np.sum((np.isnan(imag_v[i:i+time_slots_avg_dynspec, :])==False)))])
                    i_curve_time[int(i/time_slots_avg_dynspec)]=np.array([np.nanmean(spectra_i[i:i+time_slots_avg_dynspec, :])])
                    i_imag_time[int(i/time_slots_avg_dynspec)]=np.array([np.nanstd(imag_i[i:i+time_slots_avg_dynspec, :])/np.sqrt(np.sum((np.isnan(imag_i[i:i+time_slots_avg_dynspec, :])==False)))])


                    v_curve_freq[int(f/freq_slots_avg_dynspec)]=np.array([np.nanmean(spectra_v[:,f+freq_slots_avg_dynspec])])
                    v_imag_freq[int(f/freq_slots_avg_dynspec)]=np.array([np.nanstd(imag_v[:,f+freq_slots_avg_dynspec])/np.sqrt(np.sum((np.isnan(imag_v[:,f+freq_slots_avg_dynspec])==False)))])
                    i_curve_freq[int(f/freq_slots_avg_dynspec)]=np.array([np.nanmean(spectra_i[:,f+freq_slots_avg_dynspec])])
                    i_imag_freq[int(f/freq_slots_avg_dynspec)]=np.array([np.nanstd(imag_i[:,f+freq_slots_avg_dynspec])/np.sqrt(np.sum((np.isnan(imag_i[:,f+freq_slots_avg_dynspec])==False)))])


            v_spectra_bins[v_imag_bins>(np.nanmedian(v_imag_bins)+factor*np.nanstd(v_imag_bins))]=np.nan
            i_spectra_bins[i_imag_bins>(np.nanmedian(i_imag_bins)+factor*np.nanstd(i_imag_bins))]=np.nan

            logging.info(f"Make the plots")
            fig, axes=plt.subplots(2,2, sharex='col', sharey='row', width_ratios=[4,1], height_ratios=[1,4])
            img=axes[1,0].imshow(v_spectra_bins.T, aspect='auto',interpolation='none', vmin = -2 ,vmax  = 2, extent=[0,8,15,40], cmap='bwr', origin='lower')
            axes[1,0].set_xlabel('Time (h)')
            axes[1,0].set_ylabel('Frequency (MHz)')
            axes[0,0].plot(np.linspace(0,8,v_curve_time.shape[0]), v_curve_time)
            axes[1,1].plot((v_curve_freq),np.linspace(15,40,v_curve_freq.shape[0])) 
            plt.subplots_adjust(wspace=0, hspace=0)
            cbar = plt.colorbar(img,ax=axes.ravel().tolist()) 
            cbar.set_label('Flux density (Jy)', rotation=270, labelpad=12)
            fig.delaxes(axes[0,1])
            plt.savefig(f'DynSpec_{name_root}/dynspec_v_avg_jy.png', format = 'png', bbox_inches = 'tight')



            fig, axes=plt.subplots(2,2, sharex='col', sharey='row', width_ratios=[4,1], height_ratios=[1,4])
            img=axes[1,0].imshow((v_spectra_bins/v_imag_bins).T, aspect='auto',interpolation='none', vmin = -5 ,vmax  = 5, extent=[0,8,15,40], cmap='bwr', origin='lower')
            axes[1,0].set_xlabel('Time (h)')
            axes[1,0].set_ylabel('Frequency (MHz)')
            axes[0,0].plot(np.linspace(0,8,v_curve_time.shape[0]), v_curve_time/v_imag_time)
            axes[1,1].plot((v_curve_freq/v_imag_freq),np.linspace(15,40,v_curve_freq.shape[0])) 
            plt.subplots_adjust(wspace=0, hspace=0)
            cbar = plt.colorbar(img,ax=axes.ravel().tolist()) 
            cbar.set_label('Signal/Noise', rotation=270, labelpad=12)
            fig.delaxes(axes[0,1])
            plt.savefig(f'DynSpec_{name_root}/dynspec_v_avg_ratio.png', format = 'png', bbox_inches = 'tight')


            fig, axes=plt.subplots(2,2, sharex='col', sharey='row', width_ratios=[4,1], height_ratios=[1,4])
            img=axes[1,0].imshow(i_spectra_bins.T, aspect='auto',interpolation='none', vmin = -5 ,vmax  = 5, extent=[0,8,15,40], cmap='bwr', origin='lower')
            axes[1,0].set_xlabel('Time (h)')
            axes[1,0].set_ylabel('Frequency (MHz)')
            axes[0,0].plot(np.linspace(0,8,i_curve_time.shape[0]), v_curve_time)
            axes[1,1].plot((i_curve_freq),np.linspace(15,40,i_curve_freq.shape[0])) 
            plt.subplots_adjust(wspace=0, hspace=0)
            cbar = plt.colorbar(img,ax=axes.ravel().tolist())
            cbar.set_label('Flux density (Jy)', rotation=270, labelpad=12)
            fig.delaxes(axes[0,1])
            plt.savefig(f'DynSpec_{name_root}/dynspec_i_avg_jy.png', format = 'png', bbox_inches = 'tight')


            fig, axes=plt.subplots(2,2, sharex='col', sharey='row', width_ratios=[4,1], height_ratios=[1,4])
            img=axes[1,0].imshow((i_spectra_bins/i_imag_bins).T, aspect='auto',interpolation='none', vmin = -5 ,vmax  = 5, extent=[0,8,15,40], cmap='bwr', origin='lower')
            axes[1,0].set_xlabel('Time (h)')
            axes[1,0].set_ylabel('Frequency (MHz)')
            axes[0,0].plot(np.linspace(0,8,i_curve_time.shape[0]), i_curve_time/i_imag_time)
            axes[1,1].plot((i_curve_freq/i_imag_freq),np.linspace(15,40,i_curve_freq.shape[0])) 
            plt.subplots_adjust(wspace=0, hspace=0)
            cbar = plt.colorbar(img,ax=axes.ravel().tolist())
            cbar.set_label('Signal/Noise', rotation=270, labelpad=12)
            fig.delaxes(axes[0,1])
            plt.savefig(f'DynSpec_{name_root}/dynspec_i_avg_ratio.png', format = 'png', bbox_inches = 'tight')

    logging.info(f"Pipeline finished successfully")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This is a short and simple pipeline that can calibrate LOFAR LBA data below 40 MHz when the dataset only contains core stations. It is important that the delay between stations is less than 10 ns. The observations from 2022 and later have two stations with delays around 250 ns. Their origin is unknown. This code does not deal with that unfortunately. A new code will come soon which can deal with this issues and can calibrate remote stations as well. Stay tuned. To run this, you need a basis container with all the lofar software. I can recomment you take a look here https://tikk3r.github.io/flocs/."
    )
    parser.add_argument("--path", required=True, type=str, help = 'The path to the folder with MS files. The folder should contain both the MS files with the name structure of L{id}_SAP00{source_id}_SB_{subband_numer}_uv.MS. The subband number should alawys have 4 digits. For example SB0001 for subband 1 or SB0233 for subband 233, etc. The source id is 0 or 1 depending if it is the calibrator or targer')
    parser.add_argument("--model_cal", required=True, type=str, help = 'The path and name of the calibrator model file in .skymodel format. You can use https://lcs165.lofar.eu/ to generate it and copy/paste the output.')
    parser.add_argument("--model_target", required=True, type=str, help = 'The path and name of the calibrator model file in .skymodel format. You can use https://lcs165.lofar.eu/  to generate it and copy/paste the output.')
    parser.add_argument("--sap_target", required=True, type=int, help = 'The SAP number of the target. Can either be 0 or 1.')
    parser.add_argument("--demix_A_team", required=False, type=bool, default = True, help = 'Says if the A-team sources will be removed. If not, the data will only be averaged. Default: True')
    parser.add_argument("--channels_avg", required=False, type=int, default = 128, help = 'The number of channels that you want for demixing/averaging. Default: 128')
    parser.add_argument("--time_avg", required=False, type=int, default = 30, help = 'The number of time slots that you want for demixing/averaging. Default: 30')
    parser.add_argument("--channels_image", required=False, type=int, default = 20, help = 'The number of channels that you want when imaging in shorter frequency bands. Default: 20')
    parser.add_argument("--channels_selfcal_target", required=False, type=int, default = 30, help = 'The number of channels that you want constant for selfcalibration. Default: 5. Important: This is after the data is averaged, so it is the number of averaged channels.')
    parser.add_argument("--time_selfcal_target", required=False, type=int, default = 30, help = 'The number of time slots that you want constant for selfcalibration. Default: 30. Important!: This is after the data is averaged, so it is the number of averaged time slots.')
    parser.add_argument("--start", required=False, type=str, default = 'Nothing', help = 'The steps at which you want to start. If none, it starts at the most recent step that did not finish. The steps are: Flagging, Demixing/Averaging, Merging, SecondFlagging, PolAlign, Bandpass, ClockTEC3, FirstImaging, ThirdFlagging, TEC3, SecondImaging, PolLeakage, ThirdImaging, Dynspec. Dynspec works only if dynspec == True')
    parser.add_argument("--dynspec", required=False, type=bool, default = True, help = 'Says if a dynamic spectra of the phase center should be made. Useful for transient studies if the target is in the phase center. Default: True')
    parser.add_argument("--time_slots_avg_dynspec", required=False, type=int, default = 24, help = 'The number of time slots that should be averaged in the dynamic spectrum. Default: 24')
    parser.add_argument("--freq_slots_avg_dynspec", required=False, type=int, default = 10, help = 'The number of frequency bands that should be averaged in the dynamic spectrum. Default: 10')

    args = parser.parse_args()
    main(args)

