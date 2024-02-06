"""
Gets DeepMETv1 training data from a PFNano NanoAOD file and saves them to HDF5
output.
"""
# Help Message
# ------------
# usage: convertNanoToHDF5.py [-h] [-v] [-l <leptons>] <inputfile> <outputfile>
#
# positional arguments:
#   <inputfile>           input NanoAOD file.
#   <outputfile>          output HDF5 file
#
# options:
#   -h, --help            show this help message and exit
#   -v, --verbose         show logs
#   -l <leptons>, --leptons <leptons>
#                         number of leptons to remove from pfcands (default is 2)
import sys
import logging
import warnings
import argparse
import numpy as np
import awkward as ak
import h5py
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import PFNanoAODSchema

### Parameters ###

# PFCands and GenMET fields, respectively, to be saved in HDF5 file
input_fields = [
    "d0",
    "dz",
    "eta",
    "mass",
    "pt",
    "puppiWeight",
    "px",
    "py",
    "pdgId",
    "charge",
    #'fromPV'
]
output_fields = ["px", "py"]

# Labels for fields with discrete values
labels = {
    "charge": {-1.0: 0, 0.0: 1, 1.0: 2},
    "pdgId": {
        -211.0: 0,
        -13.0: 1,
        -11.0: 2,
        0.0: 3,
        1.0: 4,
        2.0: 5,
        11.0: 6,
        13.0: 7,
        22.0: 8,
        130.0: 9,
        211.0: 10,
    },
    "fromPV": {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3},
}

auto_npf = False  # If true, overwrites npf_max at runtime
npf_max = 4500  # Number of PFCands entries in output file (fixed axis size)
padding = -999  # Padding value used to fill empty PFCands entries

### Logging ###

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputfile", metavar="<inputfile>", type=str, help="input NanoAOD file."
    )
    parser.add_argument(
        "outputfile", metavar="<outputfile>", type=str, help="output HDF5 file"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show logs")
    parser.add_argument(
        "-l",
        "--leptons",
        metavar="<leptons>",
        type=int,
        default=2,
        help="number of leptons to remove from pfcands (default is 2)",
    )
    return parser.parse_args()


def delta_r_matching(pfcands, lepton, r_max=0.001):
    """
    Remove deltaR matched lepton from pfcands. A lepton is matched to the
    nearest pfcand if they are closer than a deltaR of r_max.
    """
    dr = pfcands.delta_r(lepton)
    ipf = ak.local_index(dr)
    imin = ak.argmin(dr, axis=1, mask_identity=False)
    match = (ipf == imin) & (dr < r_max)
    return pfcands[~match]


def convert_file(inputfile, outputfile, n_leptons=2, verbose=False):
    """Get training data from a NanoAOD and save it to an HDF5 file"""
    # Configure logger if enabled
    if verbose:
        logger.setLevel(level=logging.INFO)

    # Supress warnings about names of unused branches from coffea
    warnings.filterwarnings("ignore", message="Found duplicate branch .*Jet_")

    # Get events from NanoAOD
    logger.info("Fetching events")
    events = NanoEventsFactory.from_root(
        inputfile, schemaclass=PFNanoAODSchema
    ).events()

    # Event selection
    logger.info(f"Num events before selection: {len(events)}")
    n_lep = ak.num(events.Muon) + ak.num(events.Electron)
    events = events[n_lep >= n_leptons]
    logger.info(f"Num events after selection:  {len(events)}")

    # Get training data collections and leading leptons
    pfcands = events.PFCands
    genMET = events.GenMET
    leptons = ak.concatenate([events.Muon, events.Electron], axis=1)
    leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
    leptons = leptons[:, :n_leptons]

    # DeltaR matching
    logger.info("Removing leading leptons from pfcand list")
    for ilep in range(n_leptons):
        logger.info(f"DeltaR matching lepton {ilep+1}")
        pfcands = delta_r_matching(pfcands, leptons[:, ilep])
    logger.info("Lepton matching completed")

    # px and py are not computed or saved until they are initialized
    logger.info("Computing additional quantities")
    pfcands["px"] = pfcands.px
    pfcands["py"] = pfcands.py
    genMET["px"] = genMET.px
    genMET["py"] = genMET.py
    logger.info(f"Additional computations completed")

    # Format training data
    logger.info("Preparing training inputs")
    pfcands_fields = []
    npf = ak.max(ak.num(pfcands)) if auto_npf else npf_max

    for field_name in input_fields:
        logger.info(f"Processing PFCands_{field_name}")
        field = pfcands[field_name]
        if field_name in list(labels):
            pass  # todo: conversion to labels
        field = ak.pad_none(field, npf, axis=-1, clip=True)
        field = ak.fill_none(field, padding)
        pfcands_fields.append(field)

    logger.info("Preparing training outputs")
    genMET_fields = ak.unzip(genMET[output_fields])

    # Save data to file
    logger.info("Converting to numpy arrays")
    X = np.array(pfcands_fields)
    Y = np.array(genMET_fields)

    logger.info("Saving to HDF5 file")
    with h5py.File(outputfile, "w") as h5f:
        h5f.create_dataset("X", data=X, compression="lzf")
        h5f.create_dataset("Y", data=Y, compression="lzf")

    logger.info(f"Inputs shape:  {np.shape(X)}")  # (nfields,nevents,npf)
    logger.info(f"Outputs shape: {np.shape(Y)}")  # (nfields,nevents)
    logger.info(f"Training data saved to {outputfile}")


if __name__ == "__main__":
    try:
        args = get_args()
        convert_file(args.inputfile, args.outputfile, args.leptons, args.verbose)
    except KeyboardInterrupt:
        sys.exit("\nStopping early.")
