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
    #parser.add_argument('--data', help='Input is data (default: MC)', action='store_true')
    return parser.parse_args()

def remove_from_pf(pf_cands, muon, dR_max=0.001):
    """
    This is columnar operation over events.

    Matches a muon to the nearest PF candidate in pf_cands and removes it from
    the list. Returns a tuple with the filtered pf_cands and the matched muon
    candidate pf_muon (empty if there is no match), respectively. The quantity
    pf_muon.deltaR contains the distance to the matched muon.

    If the distance to the nearest PF candidate is greater than dR_max, then no
    PF candidate is removed, and pf_muon is an empty list.
    """
    dR = muon.delta_r(pf_cands)

    dR_nearest = ak.min(dR, axis=1)
    ipf_nearest = ak.argmin(dR, axis=1)
    ipf = ak.local_index(pf_cands, axis=1)
    pf_is_muon = (ipf_nearest == ipf) & (dR_nearest < dR_max)

    pf_cands = pf_cands[np.invert(pf_is_muon)]
    pf_muon = pf_cands[pf_is_muon]
    pf_muon['deltaR'] = dR[pf_is_muon]
    return pf_cands, pf_muon

def main():
    """
    main() for convertNanoToHDF5.py.

    Reads NanoAOD input file using coffea and handles input data with awkward
    arrays. Events with at least two reconstructed muons are selected. The two
    muons with the greatest pT are matched to the corresponding PF candidates
    with the closest delta R. PF candidate information used for training is
    saved, with muon PF candidates saved separately. The generator level MET
    is saved to be used as the training target. The saved data is stored
    in an output HDF5 file.
    """
    args = get_args()
    print('Fetching events')
    events = NanoEventsFactory.from_root(
        args.input,
        schemaclass=NanoAODSchema
    ).events()
    print('Total events:', len(events))

    events['Lepton'] = ak.concatenate([events.Muon, events.Electron], axis=1)
    events.Lepton = events.Lepton[ak.argsort(events.Lepton.pt ,axis=1, ascending=False)]

    print('Requiring two leptons')
    events = events[ak.num(events.Lepton) >= 2]
    print('Num events:', len(events))

    # New Lepton objects no longer have intrinsic delta_r function

    sys.exit('Debug') #########################################################

    print('Matching leading two leptons to PFCands')
    events.PFCands, events['LepPF0'] = remove_from_pf(events.PFCands, events.Lepton[:,0])
    events.PFCands, events['LepPF1'] = remove_from_pf(events.PFCands, events.Lepton[:,1])

    print('Requiring leading two leptons to be PF matched')
    #events = events[(ak.num(pf_lep_0)!=0) & (ak.num(pf_lep_1)!=0)]
    events = events[ak.num(events.LepPF0)!=0 & ak.num(events.LepPF1)!=0]
    print('Num events:', len(events))

    # Are we using Muon[:2] or inverting PFCands selection for training target?


    '''
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
'''
    # Dictionaries to assign labels to discrete values
    d_encoding = {
       b'PFCands_charge':{-1.0: 0, 0.0: 1, 1.0: 2},
       b'PFCands_pdgId':{-211.0: 0, -13.0: 1, -11.0: 2, 0.0: 3, 1.0: 4, 2.0: 5, 11.0: 6, 13.0: 7, 22.0: 8, 130.0: 9, 211.0: 10},
    }

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

'''
