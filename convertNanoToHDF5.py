"""
Gets DeepMETv1 training inputs from PFNano NanoAOD file and saves them to HDF5
output.
Usage:
    python convertNanoToHDF5.py -i/--input <file.root> -o/--output <file.h5>
                                [-p/--npf <num>] [--auto_npf]
Help:
    python convertNanoToHDF5.py -h/--help
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
    parser.add_argument('-p', '--npf', help='Max number of PF candidates per event. Default is 4500', metavar='<num>', default=4500, type=int)
    parser.add_argument('--auto_npf', help='Determine npf based on input events max npf. Overrides -p/--npf', action='store_true')
    return parser.parse_args()

def delta_phi(obj1, obj2):
    """Returns deltaPhi between two objects in range [-pi,pi)"""
    return (obj1.phi - obj2.phi + np.pi) % (2*np.pi) - np.pi

def delta_r(obj1, obj2):
    """Returns deltaR between two objects"""
    deta = obj1.eta - obj2.eta
    dphi = delta_phi(obj1, obj2)
    return np.hypot(deta, dphi)

def remove_match(pfcands, lepton, r_max=0.001):
    """Remove deltaR matched lepton from pfcand list"""
    dr = delta_r(pfcands, lepton)
    ipf = ak.local_index(dr)
    imin = ak.argmin(dr, axis=-1)
    # Match lepton to closest pfcand inside r_max cone
    is_match = (ipf == imin) & (dr < r_max)
    return pfcands[np.invert(is_match)]

def main():
    args = get_args()
    # Paramters
    n_lep = 2                       # num leptons matched/removed from pfcands
    pad_value = 0                   # padding for empty training data
    output_fields = ['px', 'py']    # GenMET quantities saved as outputs
    input_fields = [                # PFCands quantities saved as inputs
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
        #'fromPV',
    ]

    print('Fetching events')
    events = NanoEventsFactory.from_root(
        args.input,
        schemaclass=NanoAODSchema
    ).events()
    print('Num events before selection:', len(events))
    n_lep_all = ak.num(events.Muon) + ak.num(events.Electron)
    events = events[n_lep_all >= n_lep]
    print('Num events after selection: ', len(events))

    # Get pf cands and leading leptons
    pfcands = events.PFCands
    leptons = ak.concatenate([events.Muon, events.Electron], axis=1)
    leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
    leptons = leptons[:,:n_lep]

    for ilep in range(n_lep):
        print(f'DeltaR matching lepton {ilep+1}')
        pfcands = remove_match(pfcands, leptons[:,ilep])

    print('Computing PFCands_px')
    pfcands['px'] = pfcands.pt * np.cos(pfcands.phi)
    print('Computing PFCands_py')
    pfcands['py'] = pfcands.pt * np.cos(pfcands.phi)

    print('Preparing training input data:')
    training_inputs = []
    fields = ak.unzip(pfcands[input_fields])
    npf = args.npf if not args.auto_npf else ak.max(ak.num(pfcands))
    for i,field in enumerate(fields):
        print(f'Processing PFCands_{input_fields[i]}')
        field = ak.pad_none(field, npf, axis=-1, clip=True)
        field = ak.fill_none(field, pad_value)
        training_inputs.append(field)

    # Training input format: training_inputs[ifield][ievt][ipf]
    training_inputs = np.array(training_inputs)

    print(np.shape(training_inputs))
    sys.exit('DEBUG')

    #print('Saving training outputs')
    #genMET_outputs = ak.concatenate([
    #   [events.GenMET.pt * np.cos(events.GenMET.phi)]       # px
    #   [events.GenMET.pt * np.sin(events.GenMET.phi)]       # py
    #], axis=1)

#    # Dictionaries to assign labels to discrete values
#    d_encoding = {
#       b'PFCands_charge':{-1.0: 0, 0.0: 1, 1.0: 2},
#       b'PFCands_pdgId':{-211.0: 0, -13.0: 1, -11.0: 2, 0.0: 3, 1.0: 4, 2.0: 5, 11.0: 6, 13.0: 7, 22.0: 8, 130.0: 9, 211.0: 10},
#    }
#
#        # truth info
#        Y[e][0] += tree['GenMET_pt'][e] * np.cos(tree['GenMET_phi'][e])
#        Y[e][1] += tree['GenMET_pt'][e] * np.sin(tree['GenMET_phi'][e])
#
#    with h5py.File(args.output, 'w') as h5f:
#        h5f.create_dataset('X',    data=X,    compression='lzf')
#        h5f.create_dataset('Y',    data=Y,    compression='lzf')
#        h5f.create_dataset('EVT',  data=EVT,  compression='lzf')
#        h5f.create_dataset('XLep', data=XLep, compression='lzf')

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
