"""
Gets DeepMETv1 training data from PFNano NanoAOD file and saves them to HDF5
output.

Help Message
------------
usage: convertNanoToHDF5.py [-h] [-v] [-l <leptons>] [-p <npf>]
                            [--auto_npf] [-f <fill>]
                            <inputfile> <outputfile>

positional arguments:
  <inputfile>           input NanoAOD file.
  <outputfile>          output HDF5 file

options:
  -h, --help            show this help message and exit
  -v, --verbose         print logs
  -l <leptons>, --leptons <leptons>
                        number of leptons to remove from pfcands (default
                        is 2)
  -p <npf>, --npf <npf>
                        max number of pfcands per event (default is 4500)
  --auto_npf            determine npf based on input events max npf
                        (overrides -p/--npf)
  -f <fill>, --fill <fill>
                        value used to pad and fill empty training data
                        entries (default is -999)
"""
# Native libraries
import sys
import logging
import warnings
import argparse
# Third-party libraries
import numpy as np
import awkward as ak
import h5py
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import NanoAODSchema

def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile',
        metavar='<inputfile>', type=str,
        help='input NanoAOD file.')
    parser.add_argument('outputfile',
        metavar='<outputfile>', type=str,
        help='output HDF5 file')
    parser.add_argument('-v', '--verbose',
        action='store_true',
        help='show logs')
    parser.add_argument('-l', '--leptons',
        metavar='<leptons>', type=int, default=2,
        help='number of leptons to remove from pfcands (default is 2)')
    parser.add_argument('-p', '--npf',
        metavar='<npf>', type=int, default=4500,
        help='max number of pfcands per event (default is 4500)')
    parser.add_argument('--auto_npf', action='store_true',
        help='determine npf based on input events max npf '
             '(overrides -p/--npf)')
    parser.add_argument('-f', '--fill',
        metavar='<fill>', type=float, default=-999,
        help='value used to pad and fill empty training data entries '
             '(default is -999)')
    return parser.parse_args()

def delta_phi(obj1, obj2):
    """Returns deltaPhi between two objects in range [-pi,pi)"""
    return (obj1.phi - obj2.phi + np.pi) % (2*np.pi) - np.pi

def delta_r(obj1, obj2):
    """Returns deltaR between two objects"""
    deta = obj1.eta - obj2.eta
    dphi = delta_phi(obj1, obj2)
    return np.hypot(deta, dphi)

def px(obj):
    """Returns object px"""
    return obj.pt * np.cos(obj.phi)

def py(obj):
    """Returns object py"""
    return obj.pt * np.sin(obj.phi)

def remove_lepton(pfcands, lepton, r_max=0.001):
    """
    Remove deltaR matched lepton from pfcands. A lepton is matched to the
    nearest pfcand if they are closer than a deltaR of r_max.
    """
    dr = delta_r(pfcands, lepton)
    ipf = ak.local_index(dr)
    imin = ak.argmin(dr, axis=-1)
    is_match = (ipf == imin) & (dr < r_max)
    return pfcands[np.invert(is_match)]

def main():
    """main"""
    # Suppress irrelevant warnings from coffea. Warnings have to do with
    # the naming convention of some branches not relevant to DeepMET.
    # The offending branches are 'Jet_*' and 'FatJet_*'.
    warnings.filterwarnings('ignore', message='Found duplicate branch .*Jet_')

    input_fields = [
        'd0',
        'dz',
        'eta',
        'mass',
        'pt',
        'puppiWeight',
        'px',
        'py',
        'pdgId',
        'charge',
        #'fromPV'
    ]
    output_fields = ['px', 'py']
    # Labels for fields with discrete values
    labels = {
       'charge':{  -1.0: 0,   0.0: 1,   1.0: 2},
       'pdgId': {-211.0: 0, -13.0: 1, -11.0: 2,   0.0: 3,   1.0: 4,  2.0: 5,
                   11.0: 6,  13.0: 7,  22.0: 8, 130.0: 9, 211.0:10},
       'fromPV':{   0.0: 0,   1.0: 1,   2.0: 2,   3.0: 3}
    }
    args = get_args()

    if args.verbose:
        logging.basicConfig(
            format='[%(asctime)s] %(levelname)s: %(message)s',
            level=logging.INFO
        )

    # Getting events from NanoAOD
    logging.info('Fetching events')
    events = NanoEventsFactory.from_root(
        args.inputfile,
        schemaclass=NanoAODSchema
    ).events()

    # Event selection
    logging.info(f'Num events before selection: {len(events)}')
    n_lep = ak.num(events.Muon) + ak.num(events.Electron)
    events = events[n_lep >= args.leptons]
    logging.info(f'Num events after selection:  {len(events)}')

    # Getting training data collections and leading leptons
    pfcands = events.PFCands
    genMET = events.GenMET
    leptons = ak.concatenate([events.Muon, events.Electron], axis=1)
    leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
    leptons = leptons[:,:args.leptons]

    # DeltaR matching
    logging.info('Removing leading leptons from pfcand list')
    for ilep in range(args.leptons):
        pfcands = remove_lepton(pfcands, leptons[:,ilep])

    # Computing additional quantites
    logging.info('Computing px and py')
    pfcands['px'] = px(pfcands)
    pfcands['py'] = py(pfcands)
    genMET['px'] = px(genMET)
    genMET['py'] = py(genMET)

    # Training inputs
    logging.info('Preparing training inputs')
    pfcands_fields = []
    fields = ak.unzip(pfcands[input_fields])
    npf = ak.max(ak.num(pfcands)) if args.auto_npf else args.npf

    for field_name, field in zip(input_fields, fields):
        logging.info(f'Processing PFCands_{field_name}')
        if field_name in list(labels):
            pass # todo: conversion to labels
        field = ak.pad_none(field, npf, axis=-1, clip=True)
        field = ak.fill_none(field, args.fill)
        pfcands_fields.append(field)

    # Training outputs
    logging.info('Preparing training outputs')
    genMET_fields = ak.unzip(genMET[output_fields])

    # Saving data to file
    logging.info('Converting to numpy arrays')
    X = np.array(pfcands_fields)
    Y = np.array(genMET_fields)

    with h5py.File(args.outputfile, 'w') as h5f:
        h5f.create_dataset('X', data=X, compression='lzf')
        h5f.create_dataset('Y', data=Y, compression='lzf')

    logging.info(f'Inputs shape:  {np.shape(X)}')   # (nfields,nevents,npf)
    logging.info(f'Outputs shape: {np.shape(Y)}')   # (nfields,nevents)
    logging.info(f'Training data saved to {args.outputfile}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit('\nStopping early.')
