"""
convertNanoToHDF5.py

Usage: convertNanoToHDF5.py [-h] -i <file> -o <file> [-n <num>] [--data]

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
    #parser.add_argument('-n', '--nevt', help='Max events (default: -1)', metavar='<num>', default=-1, type=int)
    parser.add_argument('--data', help='Input is data (default: MC)', action='store_true')
    return parser.parse_args()

def delta_phi(obj1, obj2):
    return (obj1.phi - obj2.phi + np.pi) % (2 * np.pi) - np.pi

def delta_r(obj1, obj2):
#    return np.sqrt((obj1.eta - obj2.eta) ** 2 + delta_phi(obj1, obj2) ** 2)
    return np.hypot(obj1.eta - obj2.eta, delta_phi(obj1, obj2))

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
    # target = muon = n*[m1, m2]
    # store = pf = n*[pf1,pf2,...,pf_npf]

    # target = n*[[m1,m2],[m1,m2],...npf times...,[m1,m2]]
    _, target = ak.unzip(ak.cartesian([store.eta, target], nested=True))

    # target.dR = n*[[p1m1,p1m2],[p2m1,p2m2],[p3m1,p3m2],...]]
    target[drname] = delta_r(store, target)

    if unique:  # Additional filtering

        # t_idx = n*[ipf_min1, ipf_min2] for each muon, idx of closest pf
        t_index = ak.argmin(target[drname], axis=-2)

        # s_idx = n*[ipf1, ipf2, ...] # list of pf indicies
        s_index = ak.local_index(store.eta, axis=-1)

        # t_idx = n*[[imin1, imin2],...npf times...,[imin1,imin2]] # copies of "mu's closest pf"
        _, t_index = ak.unzip(ak.cartesian([s_index, t_index], nested=True))

        # target = n*[[closest_mu_for_pf1],[closest_mu_for_pf2],...] # list of closest mu for each pf
        target = target[s_index == t_index]

    # Cutting on the computed delta R

    # target(list of muons) = n*[[mu_close_pf1],[mu_close_pf2],...]
    # for each pf, closest mu if it's inside the radius
    # for each pf, [mu_close_pf] has 1 or 0 muons (if unique==True)
    # only two pf idxs should be nonzero
    target = target[target[drname] < radius]

    # Sorting according to the computed delta R
    if sort:
        idx = ak.argsort(target[drname], axis=-1)
        target = target[idx]
    return target

def main():
    """
    main() for convertNanoToHDF5.py.
    Reads NanoAOD input file using coffea and handles input data with awkward  #
    arrays. Events with at least two reconstructed muons are selected. The two
    muons with the greatest pT are matched to the corresponding PF candidates
    with the closest delta R. PF candidate information used for training is
    saved, with muon PF candidates saved separately. The generator level MET
    is saved to be used as the training target. The saved data is stored
    in an output HDF5 file.
    """
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

    events = NanoEventsFactory.from_root(args.input, schemaclass=NanoAODSchema).events()
    events_selected = events[ak.num(events.Muon) >= 2]

    '''
    Testing coffea delta_r functionality
    '''

    print('Running delta R matching')
    pf_cands = events_selected.PFCands
    muons = events_selected.Muon[:,:2]

    ipf = ak.local_index(pf_cands, axis=1)
    print(ipf)

    dR_0 = muons[:,0].delta_r(pf_cands)
    print('delta r:', dR_0)
    min_dR1 = ak.argmin(dR_0, axis=1)
    print('min dR idx:', min_dR1)
    is_not_muon = (ipf!=min_dR1)
    print('mask:', is_not_muon)

    print(ak.num(pf_cands))
    pf_cands = pf_cands[is_not_muon]
    print(ak.num(pf_cands))

    sys.exit('Debug') #########################################################

    '''
    Processing MC
    Running delta R matching
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, ... 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861]]
    delta r: [[3.14, 2.93, 1.79, 1.36, 1.24, 1.33, 1.86, ... 6.64, 6.93, 6.66, 6.84, 6.71, 6.76]]
    min dR idx: [587, 1336, 414, 955, 818, 943, 1090, 1023, ... 760, 1026, 1069, 674, 1068, 579, 718]
    mask: [[True, True, True, True, True, True, True, ... True, True, True, True, True, True]]
    [1643, 2737, 1299, 2219, 1958, 2166, 2429, ... 2228, 2345, 1690, 2417, 1620, 1862]
    [1642, 2736, 1298, 2218, 1957, 2165, 2428, ... 2227, 2344, 1689, 2416, 1619, 1861]
    Debug
    '''

    dR_1 = muons[:,1].delta_r(pf_cands)
    print(dR_1)
    min_dR2 = ak.argmin(dR_1, axis=1)
    print(min_dR2)

    # Are we using Muon[:2] or inverting PFCands selection for training target?

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
