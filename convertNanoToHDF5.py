'''
convertNanoToHDF5.py
This script converts PFNano NanoAOD files to HDF5 files to be used in training
DeepMETv1.
'''
import sys
import warnings
import argparse
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import NanoAODSchema
import numpy as np
import h5py
import progressbar

def get_args():
    """
    Usage: convertNanoToHDF5.py [-h] -i <file> -o <file> [-n <num>] [--data]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input NanoAOD file', metavar='<file>', required=True, type=str)
    parser.add_argument('-o', '--output', help='Output HDF5 file', metavar='<file>', required=True, type=str)
    parser.add_argument('-n', '--nEvents', help='Max events (default: -1)', metavar='<num>', default=-1, type=int)
    parser.add_argument('--data', help='Input is data (default: MC)', action='store_true')
    return parser.parse_args()

def deltaR(eta1, phi1, eta2, phi2):
    """Calculate deltaR"""
    dphi = (phi1-phi2)
    while dphi >  np.pi: dphi -= 2*np.pi
    while dphi < -np.pi: dphi += 2*np.pi
    deta = eta1-eta2
    return np.hypot(deta, dphi)

# This function was copied over from DeepMETv2.
# https://github.com/DeepMETv2/DeepMETv2/blob/master/data_dytt/generate_npz.py
def run_deltar_matching(store,
                        target,
                        drname='deltaR',
                        radius=0.4,
                        unique=False,
                        sort=False):
    """
    Running a delta R matching of some object collection "store" of dimension NxS
    with some target collection "target" of dimension NxT, The return object will
    have dimension NxSxT' where objects in the T' contain all "target" objects
    within the delta R radius. The delta R between the store and target object will
    be stored in the field `deltaR`. If the unique flag is turned on, then objects
    in the target collection will only be associated to the closest object. If the
    sort flag is turned on, then the target collection will be sorted according to
    the computed `deltaR`.
    """
    _, target = ak.unzip(ak.cartesian([store.eta, target], nested=True))
    target[drname] = delta_r(store, target)
    if unique:  # Additional filtering
        t_index = ak.argmin(target[drname], axis=-2)
        s_index = ak.local_index(store.eta, axis=-1)
        _, t_index = ak.unzip(ak.cartesian([s_index, t_index], nested=True))
        target = target[s_index == t_index]

    # Cutting on the computed delta R
    target = target[target[drname] < radius]

    # Sorting according to the computed delta R
    if sort:
        idx = ak.argsort(target[drname], axis=-1)
        target = target[idx]
    return target


def main():
    args = get_args()
    if args.data:
        print('Processing data')
    else:
        print('Processing MC')

    # Dictionaries to assign labels to discrete values
    d_encoding = {
       b'PFCands_charge':{-1.0: 0, 0.0: 1, 1.0: 2},
       b'PFCands_pdgId':{-211.0: 0, -13.0: 1, -11.0: 2, 0.0: 3, 1.0: 4, 2.0: 5, 11.0: 6, 13.0: 7, 22.0: 8, 130.0: 9, 211.0: 10},
    }

    # Get events, but suppress irrelevant warnings from coffea
    # Warnings have to do with the naming convention of some unused branches
    warnings.filterwarnings('ignore', message='Found duplicate branch .*Jet_')
    events = NanoEventsFactory.from_root(args.input, schemaclass=NanoAODSchema).events()

    # Testing coffea and awkward
    muons = events.Muon[:]
    print("nEvents:", len(events))
    import awkward as ak
    nMuonMax = max(ak.num(muons, axis=1))
    print("Max nMuons:", nMuonMax)

    sys.exit('Debug') #########################################################

    # general setup
    maxNPF = 4500
    nFeatures = 14

    maxEntries = len(tree['nPFCands']) if args.nEvents==-1 else args.nEvents
    # input PF candidates
    X = np.zeros(shape=(maxEntries,maxNPF,nFeatures), dtype=float, order='F')
    # recoil estimators
    Y = np.zeros(shape=(maxEntries,2), dtype=float, order='F')
    # leptons
    XLep = np.zeros(shape=(maxEntries, 2, nFeatures), dtype=float, order='F')
    # event-level information
    EVT = np.zeros(shape=(maxEntries,len(varList_evt)), dtype=float, order='F')
    print(X.shape)

    widgets=[
        progressbar.SimpleProgress(), ' - ',
        progressbar.Timer(), ' - ',
        progressbar.Bar(), ' - ',
        progressbar.AbsoluteETA()
    ]

    # loop over events
    for e in progressbar.progressbar(range(maxEntries), widgets=widgets):
        Leptons = []
        for ilep in range(min(2, tree['nMuon'][e])):
            Leptons.append( (tree['Muon_pt'][e][ilep], tree['Muon_eta'][e][ilep], tree['Muon_phi'][e][ilep]) )

        # get momenta
        ipf = 0
        ilep = 0
        for j in range(tree['nPFCands'][e]):
            if ipf == maxNPF:
                break

            pt = tree['PFCands_pt'][e][j]
            #if pt < 0.5:
            #    continue
            eta = tree['PFCands_eta'][e][j]
            phi = tree['PFCands_phi'][e][j]

            pf = X[e][ipf]

            isLep = False
            for lep in Leptons:
                if deltaR(eta, phi, lep[1], lep[2])<0.001 and abs(pt/lep[0]-1.0)<0.4:
                    # pfcand matched to the muon
                    # fill into XLep instead
                    isLep = True
                    pf = XLep[e][ilep]
                    ilep += 1
                    Leptons.remove(lep)
                    break
            if not isLep:
                ipf += 1

            # 4-momentum
            pf[0] = pt
            pf[1] = pt * np.cos(phi)
            pf[2] = pt * np.sin(phi)
            pf[3] = eta
            pf[4] = phi
            pf[5] = tree['PFCands_d0'][e][j]
            pf[6] = tree['PFCands_dz'][e][j]
            pf[7] = tree['PFCands_puppiWeightNoLep'][e][j]
            pf[8] = tree['PFCands_mass'][e][j]
            pf[9] = tree['PFCands_puppiWeight'][e][j]
            # encoding
            pf[10] = d_encoding[b'PFCands_pdgId' ][float(tree['PFCands_pdgId' ][e][j])]
            pf[11] = d_encoding[b'PFCands_charge'][float(tree['PFCands_charge'][e][j])]

        # truth info
        Y[e][0] += tree['GenMET_pt'][e] * np.cos(tree['GenMET_phi'][e])
        Y[e][1] += tree['GenMET_pt'][e] * np.sin(tree['GenMET_phi'][e])

        EVT[e][0] = tree['Rho_fixedGridRhoFastjetAll'][e]
        EVT[e][1] = tree['Rho_fixedGridRhoFastjetCentralCalo'][e]
        EVT[e][2] = tree['PV_npvs'][e]
        EVT[e][3] = tree['PV_npvsGood'][e]
        EVT[e][4] = tree['nMuon'][e]

    with h5py.File(args.output, 'w') as h5f:
        h5f.create_dataset('X',    data=X,    compression='lzf')
        h5f.create_dataset('Y',    data=Y,    compression='lzf')
        h5f.create_dataset('EVT',  data=EVT,  compression='lzf')
        h5f.create_dataset('XLep', data=XLep, compression='lzf')

### Run Script ###

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit('\nStopping early.')

