"""
convertNanoToHDF5.py

Usage: python convertNanoToHDF5.py [-h] -i <file> -o <file> [-n <num>] [--data]

This script converts PFNano NanoAOD files to HDF5 files to be used in training
DeepMETv1.
"""
import sys
import warnings
import argparse
import numpy as np
import awkward as ak
import h5py
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import NanoAODSchema

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input NanoAOD file', metavar='<file>', required=True, type=str)
    parser.add_argument('-o', '--output', help='Output HDF5 file', metavar='<file>', required=True, type=str)
    return parser.parse_args()

def delta_phi(obj1, obj2):
    """Returns deltaPhi between two objects in range [-pi,pi)"""
    return (obj1.phi - obj2.phi + np.pi) % (2*np.pi) - np.pi

def delta_r(obj1, obj2):
    """Returns deltaR between two objects"""
    deta = obj1.eta - obj2.eta
    dphi = delta_phi(obj1, obj2)
    return np.hypot(deta, dphi)

def input_field(n_obj, field, value=0):
    """
    Converts field from a ragged array to a regular array to be saved as input
    for training.
    """
    return ak.fill_none(ak.pad_none(field, n_obj, clip=True), max_entries)

def main():
    args = get_args()
    print('Fetching events')
    events = NanoEventsFactory.from_root(
        args.input,
        schemaclass=NanoAODSchema
    ).events()
    print('Num events before selection:', len(events))
    n_leptons = ak.num(events.Muon) + ak.num(events.Electron)
    events = events[n_leptons >= 2]
    print('Num events after selection: ', len(events))

    # Get pf cands and the two leading leptons
    pfcands = events.PFCands
    leptons = ak.concatenate([events.Muon, events.Electron], axis=1)
    leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
    leptons = leptons[:,:2]

    # Compute delta R between each PF candidate and each lepton
    pf, lep = ak.unzip(ak.cartesian([pfcands, leptons], nested=True))
    dr = delta_r(pf, lep)

    # Get PF candidates that are closest to a lepton (one per lepton)
    ipf = ak.local_index(pf, axis=1)
    ipf_closest = ak.argmin(dr, axis=1)
    ipf, ipf_closest = ak.unzip(ak.cartesian([ipf, ipf_closest], nested=True))
    is_closest = ak.any(ipf==ipf_closest, axis=2)
    # Cut on min delta R
    in_cone = (ak.min(dr, axis=2) < 0.001)

    # Remove closest lepton match from PF candidates
    pfcands = pfcands[np.invert(is_closest & in_cone)]

    sys.exit('DEBUG')

    #max_npf = ak.max(ak.num(pfcands))  # Value used in DeepMETv2 inputs
    max_npf = 4500                      # Value used in DeepMETv1 Run2 training

    # inputs: d0, dz, eta, mass, pt, puppi, px, py, pdgId, charge, fromPV
    training_inputs = ak.concatenate([
        [input_field(max_npf, pfcands.d0)],
        [input_field(max_npf, pfcands.dz)],
        [input_field(max_npf, pfcands.eta)],
        [input_field(max_npf, pfcands.mass)],
        [input_field(max_npf, pfcands.pt)],
        [input_field(max_npf, pfcands.puppiWeight)],
        [input_field(max_npf, pfcands.pt * np.cos(pfcands.phi))],
        [input_field(max_npf, pfcands.pt * np.sin(pfcands.phi))],
        [input_field(max_npf, pfcands.pdgId)],
        [input_field(max_npf, pfcands.charge)],
        [input_field(max_npf, pfcands.fromPV)],
    ])

    '''
    # Dictionaries to assign labels to discrete values
    d_encoding = {
       b'PFCands_charge':{-1.0: 0, 0.0: 1, 1.0: 2},
       b'PFCands_pdgId':{-211.0: 0, -13.0: 1, -11.0: 2, 0.0: 3, 1.0: 4, 2.0: 5, 11.0: 6, 13.0: 7, 22.0: 8, 130.0: 9, 211.0: 10},
    }

        # truth info
        Y[e][0] += tree['GenMET_pt'][e] * np.cos(tree['GenMET_phi'][e])
        Y[e][1] += tree['GenMET_pt'][e] * np.sin(tree['GenMET_phi'][e])

    with h5py.File(args.output, 'w') as h5f:
        h5f.create_dataset('X',    data=X,    compression='lzf')
        h5f.create_dataset('Y',    data=Y,    compression='lzf')
        h5f.create_dataset('EVT',  data=EVT,  compression='lzf')
        h5f.create_dataset('XLep', data=XLep, compression='lzf')
    '''

if __name__ == '__main__':
    try:
        # Suppress irrelevant warnings from coffea. Warnings have to do with
        # the naming convention of some branches not relevant to DeepMET.
        # 'Jet_*' and 'FatJet_*'
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Found duplicate branch .*Jet_')
            main()
    except KeyboardInterrupt:
        sys.exit('\nStopping early.')
