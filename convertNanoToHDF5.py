#!/usr/bin/env python3
import sys
import uproot
import numpy as np
import h5py
import progressbar
import os
import argparse

### Options ###

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input NanoAOD file', metavar='<file>', required=True, type=str)
parser.add_argument('-o', '--output', help='output HDF5 file', metavar='<file>', required=True, type=str)
parser.add_argument('-n', '--nEvents', help='max number of events (default: -1)', metavar='<num>', default=-1, type=int)
parser.add_argument('--data', help='input is data (default: MC)', action='store_true')
args = parser.parse_args()

### Variables saved ###

varList = [
    'nMuon', 'Muon_pt', 'Muon_eta', 'Muon_phi',
    'nPFCands', 'PFCands_pt', 'PFCands_eta', 'PFCands_phi',
    'PFCands_pdgId', 'PFCands_charge', 'PFCands_mass',
    'PFCands_d0', 'PFCands_dz',
    'PFCands_puppiWeightNoLep', 'PFCands_puppiWeight',
]

# Event-level variables
varList_evt = [
    'Rho_fixedGridRhoFastjetAll', 'Rho_fixedGridRhoFastjetCentralCalo',
    'PV_npvs', 'PV_npvsGood', 'nMuon'
]

# MC-only variables
varList_mc = [
    'GenMET_pt', 'GenMET_phi',
]

if not args.data:
    varList = varList + varList_mc
varList = varList + varList_evt

# Dictionaries for labeling discrete values
d_encoding = {
    b'PFCands_charge':{-1.0: 0, 0.0: 1, 1.0: 2},
    b'PFCands_pdgId':{-211.0: 0, -13.0: 1, -11.0: 2, 0.0: 3, 1.0: 4, 2.0: 5, 11.0: 6, 13.0: 7, 22.0: 8, 130.0: 9, 211.0: 10},
}

### Functions ###

def deltaR(eta1, phi1, eta2, phi2):
    """ calculate deltaR """
    dphi = (phi1-phi2)
    while dphi >  np.pi: dphi -= 2*np.pi
    while dphi < -np.pi: dphi += 2*np.pi
    deta = eta1-eta2
    return np.hypot(deta, dphi)

### Main ###

def main():
    uptree = uproot.open(args.input + ':Events')
    tree = uptree.arrays(varList)

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
        progressbar.SimpleProgress(),' - ',progressbar.Timer(),' - ',progressbar.Bar(),' - ',progressbar.AbsoluteETA()
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

