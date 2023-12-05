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
from coffea.nanoevents.schemas import PFNanoAODSchema

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

class LoggingFormatter(logging.Formatter):
    """Log formatting customizations"""

    def format(self, record):
        """Convert relativeCreated from milliseconds to seconds"""
        record.relativeCreated = record.relativeCreated / 1000
        return super().format(record)

def remove_lepton(pfcands, lepton, r_max=0.001):
    """
    Remove deltaR matched lepton from pfcands. A lepton is matched to the
    nearest pfcand if they are closer than a deltaR of r_max.
    """
    dr = pfcands.delta_r(lepton)
    ipf = ak.local_index(dr)
    imin = ak.argmin(dr, axis=-1, mask_identity=False)
    match = (ipf == imin) & (dr < r_max)
    return pfcands[~match]

def main(args):
    """main program"""
    # Suppress irrelevant warnings from coffea. Warnings have to do with
    # the naming convention of some branches not relevant to DeepMET.
    # The offending branches are 'Jet_*' and 'FatJet_*'.
    warnings.filterwarnings('ignore', message='Found duplicate branch .*Jet_')

    # PFCands and GenMET fields, respectively, to be saved in HDF5 file
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

    # Configure logging if enabled
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        formatter = LoggingFormatter(
            '[%(relativeCreated)03ds] %(levelname)s: %(message)s')
        logging.root.handlers[0].setFormatter(formatter)

    # Get events from NanoAOD
    logging.info('Fetching events')
    events = NanoEventsFactory.from_root(
        args.inputfile,
        schemaclass=PFNanoAODSchema
    ).events()

    # Event selection
    logging.info(f'Num events before selection: {len(events)}')
    n_lep = ak.num(events.Muon) + ak.num(events.Electron)
    events = events[n_lep >= args.leptons]
    logging.info(f'Num events after selection:  {len(events)}')

    # Get training data collections and leading leptons
    pfcands = events.PFCands
    genMET = events.GenMET
    leptons = ak.concatenate([events.Muon, events.Electron], axis=1)
    leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
    leptons = leptons[:,:args.leptons]

    # DeltaR matching
    logging.info('Removing leading leptons from pfcand list')
    for ilep in range(args.leptons):
        pfcands = remove_lepton(pfcands, leptons[:,ilep])
    logging.info('Lepton matching completed')

    # px and py are not computed or saved until they are initialized
    logging.info('Computing additional quantities')
    pfcands['px'] = pfcands.px
    pfcands['py'] = pfcands.py
    genMET['px'] = genMET.px
    genMET['py'] = genMET.py
    logging.info(f'Additional computations completed')

    # Format training data
    logging.info('Preparing training inputs')
    pfcands_fields = []
    npf = ak.max(ak.num(pfcands)) if args.auto_npf else args.npf

    for field_name in input_fields:
        logging.info(f'Processing PFCands_{field_name}')
        field = pfcands[field_name]
        if field_name in list(labels):
            pass # todo: conversion to labels
        field = ak.pad_none(field, npf, axis=-1, clip=True)
        field = ak.fill_none(field, args.fill)
        pfcands_fields.append(field)

    logging.info('Preparing training outputs')
    genMET_fields = ak.unzip(genMET[output_fields])

    # Save data to file
    logging.info('Converting to numpy arrays')
    X = np.array(pfcands_fields)
    Y = np.array(genMET_fields)

    logging.info('Saving to HDF5 file')
    with h5py.File(args.outputfile, 'w') as h5f:
        h5f.create_dataset('X', data=X, compression='lzf')
        h5f.create_dataset('Y', data=Y, compression='lzf')

    logging.info(f'Inputs shape:  {np.shape(X)}')   # (nfields,nevents,npf)
    logging.info(f'Outputs shape: {np.shape(Y)}')   # (nfields,nevents)
    logging.info(f'Training data saved to {args.outputfile}')

if __name__ == '__main__':
    try:
        args = get_args()
        main(args)
    except KeyboardInterrupt:
        sys.exit('\nStopping early.')
