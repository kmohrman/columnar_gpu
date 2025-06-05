import time
import awkward as ak
import cupy as cp
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

import torch
import cudf
from ak_from_cudf import cudf_to_awkward

import pyarrow.parquet as pq
import fastparquet
from hepconvert import root_to_parquet

import uproot
from coffea.jitters import hist as gpu_hist
import hist
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import vector

import pandas as df

print("cudf version", cudf.__version__)


#######################################################
### Test with GPU hist ###

def check_hists():

    N_dims = 4

    test_gpu = ak.Array(cp.random.multivariate_normal(
        mean=np.zeros(shape=N_dims),
        cov=np.eye(N_dims),
        size=(200_000_000//N_dims),
    ))


    hist_gpu = gpu_hist.Hist(
        "test",
        gpu_hist.Bin("x", "x coordinate", 32, -5, 5),
        gpu_hist.Bin("y", "y coordinate", 32, -5, 5),
        gpu_hist.Bin("z", "z coordinate", 32, -5, 5),
        gpu_hist.Bin("t", "t coordinate", 32, -5, 5),
    )

    hist_gpu.fill(x=test_gpu[:, 0], y=test_gpu[:,1], z=test_gpu[:,2], t=test_gpu[:,3])

    hist_gpu_cupy = cp.histogramdd(
        ak.to_cupy(test_gpu),
        bins=(32, 32, 32, 32),
        range=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)]
    )
    hist_gpu.values()[()].get()
    #gpu_hist.plot1d(hist_gpu)
    gpu_hist.plot1d(hist_gpu.project("z"))


    test_cpu = ak.to_backend(test_gpu, "cpu")
    hist_cpu = hist.new.Reg(32, -5, 5).Reg(32, -5, 5).Reg(32, -5, 5).Reg(32, -5, 5).Weight()
    hist_cpu.fill(test_cpu[:,0], test_cpu[:,1], test_cpu[:,2], test_cpu[:,3])


####################################################################################################
### Write out to parquet ###

def write_to_parquet():

    print("Starting root_to_parquet")
    root_to_parquet(in_file = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu.root",
        out_file = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_03.parquet",
        tree="Events",
        compression = "zstd",
        compression_level = 3,
        extensionarray=False,
        parquet_version="2.6",
        parquet_page_version="2.0",
        parquet_extra_options = {"column_encoding": "PLAIN"}
    )

####################################################################################################
### Lindsey debugging a problem with numba.cuda ? ###

def lindsey_debugging_numba_cuda():

    ak.numba.register_and_check()

    @nb.vectorize(
        [
            nb.float32(nb.float32),
            nb.float64(nb.float64),
        ]
    )
    def _square(x):
        return x * x

    @nb.vectorize(
        [
            nb.float32(nb.float32),
            nb.float64(nb.float64),
        ],
        target="cuda",
    )
    def _square_cuda(x):
        return x * x

    def square_cuda_wrapped(x):
        counts = x.layout.offsets.data[1:] - x.layout.offsets.data[:-1]
        return ak.unflatten(cp.array(_square_cuda(ak.flatten(x))), counts)

    counts = cp.random.poisson(lam=3, size=5000000)
    flat_values = cp.random.normal(size=int(counts.sum()))

    values = ak.unflatten(flat_values, counts)

    values2_cpu = _square(ak.to_backend(values, "cpu"))

    print(values2_cpu)

    #values2 = square_cuda_wrapped(values) # Gives errror: "AttributeError: 'CUDATypingContext' object has no attribute 'resolve_argument_type'. Did you mean: 'resolve_value_type'?"
    #print(values2)

    #########

    counts = cp.random.poisson(lam=3, size=5000000)
    flat_values = cp.random.normal(size=int(counts.sum()))

    values = ak.unflatten(flat_values, counts)

    np_vals = np.abs(values)
    print(np_vals, ak.backend(np_vals))

    #cp_vals = cp.abs(values) # Gives error: "TypeError: Unsupported type <class 'awkward.highlevel.Array'>"
    #print(cp_vals, ak.backend(cp_vals))

    values

    dir(nb.cuda)

    #dir(values2)

    cp.float32 == np.float32


####################################################################################################
### Check combinations ###

def check_combinations():

    jetmet = uproot.open(
        "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu.root:Events"
    ).arrays(
        ["Jet_pt","MET_pt"],
    )

    print("\n\nRunning the ak combinations stuff")
    print(time.time())
    Jet_pt = ak.to_backend(jetmet.Jet_pt, "cuda")
    #Jet_pt = ak.to_backend(jetmet.Jet_pt, "cpu")
    Jet_pt = Jet_pt[:100] # TMP

    print("len Jet_pt",len(Jet_pt))
    print("type Jet_pt",type(Jet_pt))

    t_before_comb = time.time()
    print("HERE 1 before combinations",t_before_comb)
    jet_comb_out = ak.combinations(Jet_pt, 2)

    t_after_comb = time.time()
    print("HERE 1 after combinations",t_after_comb)
    print("Time for ak comb",t_after_comb-t_before_comb,"\n")


    print("jet_comb_out",jet_comb_out)
    print("len jet_comb_out",len(jet_comb_out))
    print("type jet_comb_out",type(jet_comb_out))

    print("comb part done, moving on.....")





####################################################################################################
####################################################################################################
####################################################################################################


# Q1 query GPU
# Fill hist with met for all events
def query1_gpu(filepath,makeplot=False):

    print("\nStarting Q1 code on gpu..")

    # Get met pt and fill hist

    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["MET_pt"])
    t_after_read = time.time() # Time

    MET_pt = cudf_to_awkward(table["MET_pt"])
    t_after_load = time.time() # Time

    q1_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
    )
    q1_hist.fill(met=MET_pt)
    t1 = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q1_hist.plot1d(flow="none");
        fig.savefig("fig_q1_gpu.png")

    print(f"Time for q1: {t1-t0}")
    print(f"    Time for reading: {t_after_read-t0} ({np.round(100*(t_after_read-t0)/(t1-t0),1)}%)")
    print(f"    Time for loading: {t_after_load-t_after_read} ({np.round(100*(t_after_load-t_after_read)/(t1-t0),1)}%)")
    print(f"    Time for computing and histing: {t1-t_after_load} ({np.round(100*(t1-t_after_load)/(t1-t0),1)}%)")
    return(q1_hist,t1-t0)



# Q1 query CPU
# Fill hist with met for all events
def query1_cpu(filepath,makeplot=False):

    # Fill hist with met for all events
    print("\nStarting Q1 code on cpu..")

    # Get met pt and fill hist

    t0 = time.time()

    table = df.read_parquet(filepath, columns = ["MET_pt"])
    t_after_read = time.time() # Time

    MET_pt = ak.Array(table["MET_pt"])
    t_after_load = time.time() # Time

    q1_hist = hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]").Double()
    q1_hist.fill(met=MET_pt)
    t1 = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q1_hist.plot1d(flow="none");
        fig.savefig("fig_q1_cpu.png")

    print(f"Time for q1: {t1-t0}")
    print(f"    Time for reading: {t_after_read-t0} ({np.round(100*(t_after_read-t0)/(t1-t0),1)}%)")
    print(f"    Time for loading: {t_after_load-t_after_read} ({np.round(100*(t_after_load-t_after_read)/(t1-t0),1)}%)")
    print(f"    Time for computing and histing: {t1-t_after_load} ({np.round(100*(t1-t_after_load)/(t1-t0),1)}%)")
    return(q1_hist,t1-t0)



# Q2 query GPU
# Fill hist with pt for all jets
def query2_gpu(filepath,makeplot=False):

    print("\nStarting Q2 code on gpu..")

    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["Jet_pt"])
    t_after_read = time.time() # Time

    Jet_pt = cudf_to_awkward(table["Jet_pt"])
    t_after_load = time.time() # Time

    q2_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("ptj", "Jet $p_{T}$ [GeV]", 100, 0, 200),
    )
    q2_hist.fill(ptj=ak.flatten(Jet_pt))
    t1 = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q2_hist.to_hist().plot1d(flow="none");
        fig.savefig("fig_q2_gpu.png")

    print(f"Time for q2: {t1-t0}")
    print(f"    Time for reading: {t_after_read-t0} ({np.round(100*(t_after_read-t0)/(t1-t0),1)}%)")
    print(f"    Time for loading: {t_after_load-t_after_read} ({np.round(100*(t_after_load-t_after_read)/(t1-t0),1)}%)")
    print(f"    Time for computing and histing: {t1-t_after_load} ({np.round(100*(t1-t_after_load)/(t1-t0),1)}%)")
    return(q2_hist,t1-t0)



# Q2 query CPU
# Fill hist with pt for all jets
def query2_cpu(filepath,makeplot=False):

    print("\nStarting Q2 code on cpu..")

    t0 = time.time()

    table = df.read_parquet(filepath, columns = ["Jet_pt"])
    t_after_read = time.time() # Time

    Jet_pt = ak.Array(table["Jet_pt"])
    t_after_load = time.time() # Time

    q2_hist = hist.new.Reg(100, 0, 200, name="ptj", label="Jet $p_{T}$ [GeV]").Double()
    q2_hist.fill(ptj=ak.flatten(Jet_pt))
    t1 = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q2_hist.plot1d(flow="none");
        fig.savefig("fig_q2_cpu.png")

    print(f"Time for q2: {t1-t0}")
    print(f"    Time for reading: {t_after_read-t0} ({np.round(100*(t_after_read-t0)/(t1-t0),1)}%)")
    print(f"    Time for loading: {t_after_load-t_after_read} ({np.round(100*(t_after_load-t_after_read)/(t1-t0),1)}%)")
    print(f"    Time for computing and histing: {t1-t_after_load} ({np.round(100*(t1-t_after_load)/(t1-t0),1)}%)")
    return(q2_hist,t1-t0)



# Q3 query GPU
# Fill a hist with pt of jets with eta less than 1
def query3_gpu(filepath,makeplot=False):

    print("\nStarting Q3 code on gpu..")

    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["Jet_pt", "Jet_eta"])
    t_after_read = time.time() # Time

    Jet_pt = cudf_to_awkward(table["Jet_pt"])
    Jet_eta = cudf_to_awkward(table["Jet_eta"])
    t_after_load = time.time() # Time

    q3_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("ptj", "Jet $p_{T}$ [GeV]", 100, 0, 200),
    )
    q3_hist.fill(ptj=ak.flatten(Jet_pt[abs(Jet_eta) < 1.0]))
    t1 = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q3_hist.to_hist().plot1d(flow="none");
        fig.savefig("fig_q3_gpu.png")

    print(f"Time for q3: {t1-t0}")
    print(f"    Time for reading: {t_after_read-t0} ({np.round(100*(t_after_read-t0)/(t1-t0),1)}%)")
    print(f"    Time for loading: {t_after_load-t_after_read} ({np.round(100*(t_after_load-t_after_read)/(t1-t0),1)}%)")
    print(f"    Time for computing and histing: {t1-t_after_load} ({np.round(100*(t1-t_after_load)/(t1-t0),1)}%)")
    return(q3_hist,t1-t0)



# Q3 query CPU
# Fill a hist with pt of jets with eta less than 1
def query3_cpu(filepath,makeplot=False):

    print("\nStarting Q3 code on cpu..")

    t0 = time.time()

    table = df.read_parquet(filepath, columns = ["Jet_pt", "Jet_eta"])
    t_after_read = time.time() # Time

    Jet_pt = ak.Array(table["Jet_pt"])
    Jet_eta = ak.Array(table["Jet_eta"])
    t_after_load = time.time() # Time

    q3_hist = hist.new.Reg(100, 0, 200, name="ptj", label="Jet $p_{T}$ [GeV]").Double()
    q3_hist.fill(ptj=ak.flatten(Jet_pt[abs(Jet_eta) < 1.0]))

    t1 = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q3_hist.plot1d(flow="none");
        fig.savefig("fig_q3_cpu.png")

    print(f"Time for q3: {t1-t0}")
    print(f"    Time for reading: {t_after_read-t0} ({np.round(100*(t_after_read-t0)/(t1-t0),1)}%)")
    print(f"    Time for loading: {t_after_load-t_after_read} ({np.round(100*(t_after_load-t_after_read)/(t1-t0),1)}%)")
    print(f"    Time for computing and histing: {t1-t_after_load} ({np.round(100*(t1-t_after_load)/(t1-t0),1)}%)")
    return(q3_hist,t1-t0)



# Q4 query GPU
# Fill a hist with MET of events that have at least two jets with pt>40
def query4_gpu(filepath,makeplot=False):

    print("\nStarting Q4 code on gpu..")

    t0 = time.time()
    table = cudf.read_parquet(filepath, columns = ["Jet_pt", "MET_pt"])
    t_after_read = time.time() # Time

    Jet_pt = cudf_to_awkward(table["Jet_pt"])
    MET_pt = cudf_to_awkward(table["MET_pt"])
    t_after_load = time.time() # Time

    q4_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
    )
    has2jets = ak.sum(Jet_pt > 40, axis=1) >= 2
    q4_hist.fill(met=MET_pt[has2jets])
    t1 = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q4_hist.to_hist().plot1d(flow="none");
        fig.savefig("fig_q4_gpu.png")

    print(f"Time for q4: {t1-t0}")
    print(f"    Time for reading: {t_after_read-t0} ({np.round(100*(t_after_read-t0)/(t1-t0),1)}%)")
    print(f"    Time for loading: {t_after_load-t_after_read} ({np.round(100*(t_after_load-t_after_read)/(t1-t0),1)}%)")
    print(f"    Time for computing and histing: {t1-t_after_load} ({np.round(100*(t1-t_after_load)/(t1-t0),1)}%)")
    return(q4_hist,t1-t0)


# Q4 query CPU
# Fill a hist with MET of events that have at least two jets with pt>40
def query4_cpu(filepath,makeplot=False):

    print("\nStarting Q4 code on cpu..")

    t0 = time.time()

    table = df.read_parquet(filepath, columns = ["Jet_pt", "MET_pt"])
    t_after_read = time.time() # Time

    Jet_pt = ak.Array(table["Jet_pt"])
    MET_pt = ak.Array(table["MET_pt"])
    t_after_load = time.time() # Time

    q4_hist = hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]").Double()
    has2jets = ak.sum(Jet_pt > 40, axis=1) >= 2
    q4_hist.fill(met=MET_pt[has2jets])

    t1 = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q4_hist.plot1d(flow="none");
        fig.savefig("fig_q4_cpu.png")

    print(f"Time for q4: {t1-t0}")
    print(f"    Time for reading: {t_after_read-t0} ({np.round(100*(t_after_read-t0)/(t1-t0),1)}%)")
    print(f"    Time for loading: {t_after_load-t_after_read} ({np.round(100*(t_after_load-t_after_read)/(t1-t0),1)}%)")
    print(f"    Time for computing and histing: {t1-t_after_load} ({np.round(100*(t1-t_after_load)/(t1-t0),1)}%)")
    return(q4_hist,t1-t0)



# Q5 query GPU
# Fill a hist with MET For events that have an OS muon pair with an invariant mass between 60 and 120 GeV
def query5_gpu(filepath,makeplot=False):

    print("\nStarting Q5 code on gpu..")

    t0 = time.time()
    table = cudf.read_parquet(
        filepath,
        columns = [
            "MET_pt",
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            "Muon_mass",
            "Muon_charge",
        ]
    )
    t_after_read = time.time() # Time

    MET_pt = cudf_to_awkward(table["MET_pt"])
    Muon_pt = cudf_to_awkward(table["Muon_pt"])
    Muon_eta = cudf_to_awkward(table["Muon_eta"])
    Muon_phi = cudf_to_awkward(table["Muon_phi"])
    Muon_mass = cudf_to_awkward(table["Muon_mass"])
    Muon_charge = cudf_to_awkward(table["Muon_charge"])
    t_after_load = time.time() # Time

    q5_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
    )

    Muon = ak.zip(
        {
            "pt": Muon_pt,
            "eta": Muon_eta,
            "phi": Muon_phi,
            "mass": Muon_mass,
            "charge": Muon_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )
    #)[0:10000]


    mupair = ak.combinations(Muon, 2, fields=["mu1", "mu2"])
    pairmass = (mupair.mu1 + mupair.mu2).mass
    goodevent = ak.any(
        (pairmass > 60)
        & (pairmass < 120)
        & (mupair.mu1.charge == -mupair.mu2.charge),
        axis=1,
    )

    q5_hist.fill(met=MET_pt[goodevent])
    t1 = time.time()


    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q5_hist.to_hist().plot1d(flow="none");
        fig.savefig("fig_q5_gpu.png")

    print(f"Time for q5: {t1-t0}")
    print(f"    Time for reading: {t_after_read-t0} ({np.round(100*(t_after_read-t0)/(t1-t0),1)}%)")
    print(f"    Time for loading: {t_after_load-t_after_read} ({np.round(100*(t_after_load-t_after_read)/(t1-t0),1)}%)")
    print(f"    Time for computing and histing: {t1-t_after_load} ({np.round(100*(t1-t_after_load)/(t1-t0),1)}%)")
    return(q5_hist,t1-t0)



# Q5 query CPU
# Fill a hist with MET For events that have an OS muon pair with an invariant mass between 60 and 120 GeV
def query5_cpu(filepath,makeplot=False):

    print("\nStarting Q5 code on cpu..")

    t0 = time.time()
    table = df.read_parquet(
        filepath,
        columns = [
            "MET_pt",
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            "Muon_mass",
            "Muon_charge",
        ]
    )
    t_after_read = time.time() # Time

    MET_pt      = ak.Array(table["MET_pt"])
    Muon_pt     = ak.Array(table["Muon_pt"])
    Muon_eta    = ak.Array(table["Muon_eta"])
    Muon_phi    = ak.Array(table["Muon_phi"])
    Muon_mass   = ak.Array(table["Muon_mass"])
    Muon_charge = ak.Array(table["Muon_charge"])
    t_after_load = time.time() # Time

    q5_hist = hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]").Double()

    Muon = ak.zip(
        {
            "pt": Muon_pt,
            "eta": Muon_eta,
            "phi": Muon_phi,
            "mass": Muon_mass,
            "charge": Muon_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )
    #)[0:10000]

    mupair = ak.combinations(Muon, 2, fields=["mu1", "mu2"])
    pairmass = (mupair.mu1 + mupair.mu2).mass
    goodevent = ak.any(
        (pairmass > 60)
        & (pairmass < 120)
        & (mupair.mu1.charge == -mupair.mu2.charge),
        axis=1,
    )

    q5_hist.fill(met=MET_pt[goodevent])
    t1 = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q5_hist.plot1d(flow="none");
        fig.savefig("fig_q5_cpu.png")

    print(f"Time for q5: {t1-t0}")
    print(f"    Time for reading: {t_after_read-t0} ({np.round(100*(t_after_read-t0)/(t1-t0),1)}%)")
    print(f"    Time for loading: {t_after_load-t_after_read} ({np.round(100*(t_after_load-t_after_read)/(t1-t0),1)}%)")
    print(f"    Time for computing and histing: {t1-t_after_load} ({np.round(100*(t1-t_after_load)/(t1-t0),1)}%)")
    return(q5_hist,t1-t0)


# Q6 query GPU
# Select events at least 3 jets
#   - Fill hist with pt of tri-jet system closest to top mass
#   - Fill hist with max b-tag score of the jets in the system
def query6_gpu(filepath,makeplot=False):

    print("\nStarting Q6 code on gpu..")

    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_btag",])

    Jet_pt = cudf_to_awkward(table["Jet_pt"])
    Jet_eta = cudf_to_awkward(table["Jet_eta"])
    Jet_phi = cudf_to_awkward(table["Jet_phi"])
    Jet_mass = cudf_to_awkward(table["Jet_mass"])
    Jet_btag = cudf_to_awkward(table["Jet_btag"])

    jets = ak.zip(
        {
            "pt": Jet_pt,
            "eta": Jet_eta,
            "phi": Jet_phi,
            "mass": Jet_mass,
            "btag": Jet_btag,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=candidate.behavior,
    )

    # Get the pt of the trijet system closest to top
    trijet = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])
    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3

    trijet = ak.flatten(
        trijet[ak.singletons(ak.argmin(abs(trijet.p4.mass - 172.5), axis=1))]
    )

    # Get max btag of the trijet system
    maxBtag = np.maximum(
        trijet.j1.btag,
        np.maximum(
            trijet.j2.btag,
            trijet.j3.btag,
        ),
    )

    q6_hist_1 = gpu_hist.Hist("Counts", gpu_hist.Bin("pt3j", "Trijet $p_{T}$ [GeV]", 100, 0, 200))
    q6_hist_1.fill(pt3j=trijet.p4.pt)

    q6_hist_2 = gpu_hist.Hist("Counts", gpu_hist.Bin("btag", "Max jet b-tag score", 100, 0, 1))
    q6_hist_2.fill(btag=maxBtag)

    t1 = time.time()

    # Plotting
    if makeplot:
        # First hist
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q6_hist_1.to_hist().plot1d(flow="none");
        fig.savefig("fig_q6p1_gpu.png")
        # Second hist
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q6_hist_2.to_hist().plot1d(flow="none");
        fig.savefig("fig_q6p2_gpu.png")

    return(q6_hist_1,q6_hist_2,t1-t0)


# Q6 query CPU
# Select events at least 3 jets
#   - Fill hist with pt of tri-jet system closest to top mass
#   - Fill hist with max b-tag score of the jets in the system
def query6_cpu(filepath,makeplot=False):

    print("\nStarting Q6 code on cpu..")

    t0 = time.time()

    table = df.read_parquet(filepath, columns = ["Jet_pt","Jet_eta","Jet_phi","Jet_mass","Jet_btag"])

    Jet_pt = ak.Array(table["Jet_pt"])
    Jet_eta = ak.Array(table["Jet_eta"])
    Jet_phi = ak.Array(table["Jet_phi"])
    Jet_mass = ak.Array(table["Jet_mass"])
    Jet_btag = ak.Array(table["Jet_btag"])

    jets = ak.zip(
        {
            "pt": Jet_pt,
            "eta": Jet_eta,
            "phi": Jet_phi,
            "mass": Jet_mass,
            "btag": Jet_btag,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=candidate.behavior,
    )

    # Get the pt of the trijet system closest to top
    trijet = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])
    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3

    trijet = ak.flatten(
        trijet[ak.singletons(ak.argmin(abs(trijet.p4.mass - 172.5), axis=1))]
    )

    # Get max btag of the trijet system
    maxBtag = np.maximum(
        trijet.j1.btag,
        np.maximum(
            trijet.j2.btag,
            trijet.j3.btag,
        ),
    )

    q6_hist_1 = hist.new.Reg(100, 0, 200, name="pt3j", label="Trijet $p_{T}$ [GeV]").Double()
    q6_hist_1.fill(pt3j=trijet.p4.pt)

    q6_hist_2 = hist.new.Reg(100, 0, 1, name="btag", label="Max jet b-tag score").Double()
    q6_hist_2.fill(btag=maxBtag)

    t1 = time.time()

    # Plotting
    if makeplot:
        # First hist
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q6_hist_1.plot1d(flow="none");
        fig.savefig("fig_q6p1_cpu.png")
        # Second hist
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q6_hist_2.plot1d(flow="none");
        fig.savefig("fig_q6p2_cpu.png")

    return(q6_hist_1,q6_hist_2,t1-t0)


# Q7 query GPU
# Fill hist with HT of jets
#   - Jets have pt>30 and far (dR>0.4) from leptons
#   - Leptons have pt>10
def query7_gpu(filepath,makeplot=False):

    print("\nStarting Q7 code on gpu..")

    t0 = time.time()

    #table = df.read_parquet(filepath, columns = ["Muon_pt", "Electron_pt", "Jet_pt", "Jet_metric_table"])
    table = cudf.read_parquet(filepath, columns = [
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_charge",
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass"
    ])
    t_after_read = time.time() # Time

    Jet_pt   = cudf_to_awkward(table["Jet_pt"])
    Jet_eta  = cudf_to_awkward(table["Jet_eta"])
    Jet_phi  = cudf_to_awkward(table["Jet_phi"])
    Jet_mass = cudf_to_awkward(table["Jet_mass"])

    Muon_pt     = cudf_to_awkward(table["Muon_pt"])
    Muon_eta    = cudf_to_awkward(table["Muon_eta"])
    Muon_phi    = cudf_to_awkward(table["Muon_phi"])
    Muon_mass   = cudf_to_awkward(table["Muon_mass"])
    Muon_charge = cudf_to_awkward(table["Muon_charge"])

    Electron_pt     = cudf_to_awkward(table["Electron_pt"])
    Electron_eta    = cudf_to_awkward(table["Electron_eta"])
    Electron_phi    = cudf_to_awkward(table["Electron_phi"])
    Electron_mass   = cudf_to_awkward(table["Electron_mass"])
    Electron_charge = cudf_to_awkward(table["Electron_charge"])

    t_after_load = time.time() # Time

    jets = ak.zip(
        {
            "pt": Jet_pt,
            "eta": Jet_eta,
            "phi": Jet_phi,
            "mass": Jet_mass,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=candidate.behavior,
    )

    Electron = ak.zip(
        {
            "pt": Electron_pt,
            "eta": Electron_eta,
            "phi": Electron_phi,
            "mass": Electron_mass,
            "charge": Electron_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    Muon = ak.zip(
        {
            "pt": Muon_pt,
            "eta": Muon_eta,
            "phi": Muon_phi,
            "mass": Muon_mass,
            "charge": Muon_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    # Get good leptons
    leptons = ak.with_name(ak.concatenate([Electron,Muon],axis=1),'PtEtaPhiMCandidate')
    leptons_good = leptons[leptons.pt>10]

    # Get good jets and sum pt to get HT
    jet_nearest_to_any_lep, dr = jets.nearest(leptons,return_metric=True)
    jets_good = jets[(jets.pt>30) & (dr>0.4)]
    ht = ak.sum(jets_good.pt,axis=1)

    # Fill hist
    q7_hist = gpu_hist.Hist("Counts", gpu_hist.Bin("sumjetpt", "Scalar sum of jet $p_{T}$ [GeV]", 100, 0, 200))
    q7_hist.fill(sumjetpt=ht)

    t1 = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q7_hist.plot1d(flow="none");
        fig.savefig("fig_q7_cpu.png")

    print(f"Time for q7: {t1-t0}")
    print(f"    Time for reading: {t_after_read-t0} ({np.round(100*(t_after_read-t0)/(t1-t0),1)}%)")
    print(f"    Time for loading: {t_after_load-t_after_read} ({np.round(100*(t_after_load-t_after_read)/(t1-t0),1)}%)")
    print(f"    Time for computing and histing: {t1-t_after_load} ({np.round(100*(t1-t_after_load)/(t1-t0),1)}%)")
    return(q7_hist,t1-t0)


# Q7 query CPU
# Fill hist with HT of jets
#   - Jets have pt>30 and far (dR>0.4) from leptons
#   - Leptons have pt>10
def query7_cpu(filepath,makeplot=False):

    print("\nStarting Q7 code on cpu..")

    t0 = time.time()

    #table = df.read_parquet(filepath, columns = ["Muon_pt", "Electron_pt", "Jet_pt", "Jet_metric_table"])
    table = df.read_parquet(filepath, columns = [
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_charge",
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass"
    ])
    t_after_read = time.time() # Time

    Jet_pt   = ak.Array(table["Jet_pt"])
    Jet_eta  = ak.Array(table["Jet_eta"])
    Jet_phi  = ak.Array(table["Jet_phi"])
    Jet_mass = ak.Array(table["Jet_mass"])

    Muon_pt     = ak.Array(table["Muon_pt"])
    Muon_eta    = ak.Array(table["Muon_eta"])
    Muon_phi    = ak.Array(table["Muon_phi"])
    Muon_mass   = ak.Array(table["Muon_mass"])
    Muon_charge = ak.Array(table["Muon_charge"])

    Electron_pt     = ak.Array(table["Electron_pt"])
    Electron_eta    = ak.Array(table["Electron_eta"])
    Electron_phi    = ak.Array(table["Electron_phi"])
    Electron_mass   = ak.Array(table["Electron_mass"])
    Electron_charge = ak.Array(table["Electron_charge"])

    t_after_load = time.time() # Time

    jets = ak.zip(
        {
            "pt": Jet_pt,
            "eta": Jet_eta,
            "phi": Jet_phi,
            "mass": Jet_mass,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=candidate.behavior,
    )

    Electron = ak.zip(
        {
            "pt": Electron_pt,
            "eta": Electron_eta,
            "phi": Electron_phi,
            "mass": Electron_mass,
            "charge": Electron_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    Muon = ak.zip(
        {
            "pt": Muon_pt,
            "eta": Muon_eta,
            "phi": Muon_phi,
            "mass": Muon_mass,
            "charge": Muon_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    # Get good leptons
    leptons = ak.with_name(ak.concatenate([Electron,Muon],axis=1),'PtEtaPhiMCandidate')
    leptons_good = leptons[leptons.pt>10]

    # Get good jets and sum pt to get HT
    jet_nearest_to_any_lep, dr = jets.nearest(leptons,return_metric=True)
    jets_good = jets[(jets.pt>30) & (dr>0.4)]
    ht = ak.sum(jets_good.pt,axis=1)

    # Fill hist
    q7_hist = hist.new.Reg(100, 0, 200, name="sumjetpt", label="Scalar sum of jet $p_{T}$ [GeV]").Double()
    q7_hist.fill(sumjetpt=ht)

    t1 = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q7_hist.plot1d(flow="none");
        fig.savefig("fig_q7_cpu.png")

    print(f"Time for q7: {t1-t0}")
    print(f"    Time for reading: {t_after_read-t0} ({np.round(100*(t_after_read-t0)/(t1-t0),1)}%)")
    print(f"    Time for loading: {t_after_load-t_after_read} ({np.round(100*(t_after_load-t_after_read)/(t1-t0),1)}%)")
    print(f"    Time for computing and histing: {t1-t_after_load} ({np.round(100*(t1-t_after_load)/(t1-t0),1)}%)")
    return(q7_hist,t1-t0)


# Q8 query CPU
# Select events with at least 3 leptons, that inlude a SFOS pair
# Plot MT of the system of the leading non-Z lepton and MET
def query8_cpu(filepath,makeplot=False):

    print("\nStarting Q8 code on cpu..")

    t0 = time.time()

    #table = df.read_parquet(filepath, columns = ["Muon_pt", "Electron_pt", "Jet_pt", "Jet_metric_table"])
    table = df.read_parquet(filepath, columns = [
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_charge",
        "MET_pt", "MET_phi",
    ])
    t_after_read = time.time() # Time

    Muon_pt     = ak.Array(table["Muon_pt"])
    Muon_eta    = ak.Array(table["Muon_eta"])
    Muon_phi    = ak.Array(table["Muon_phi"])
    Muon_mass   = ak.Array(table["Muon_mass"])
    Muon_charge = ak.Array(table["Muon_charge"])

    Electron_pt     = ak.Array(table["Electron_pt"])
    Electron_eta    = ak.Array(table["Electron_eta"])
    Electron_phi    = ak.Array(table["Electron_phi"])
    Electron_mass   = ak.Array(table["Electron_mass"])
    Electron_charge = ak.Array(table["Electron_charge"])

    MET_pt = ak.Array(table["MET_pt"])
    MET_phi = ak.Array(table["MET_phi"])

    t_after_load = time.time() # Time

    MET = ak.zip(
        {
            "pt": MET_pt,
            "phi": MET_phi,
        },
        with_name="PolarTwoVector",
        behavior=vector.behavior,
    )

    Electron = ak.zip(
        {
            "pt": Electron_pt,
            "eta": Electron_eta,
            "phi": Electron_phi,
            "mass": Electron_mass,
            "charge": Electron_charge,
            "pdgId": -11 * Electron_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    Muon = ak.zip(
        {
            "pt": Muon_pt,
            "eta": Muon_eta,
            "phi": Muon_phi,
            "mass": Muon_mass,
            "charge": Muon_charge,
            "pdgId": -13 * Muon_charge,
        },
        with_name="PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )

    # Get good leptons
    leptons = ak.with_name(ak.concatenate([Electron,Muon],axis=1),'PtEtaPhiMCandidate')
    leptons = leptons[leptons.pt>10]

    # Attatch index to each lepton
    leptons['idx'] = ak.local_index(leptons, axis=1)

    # Get pairs of leptons
    ll_pairs = ak.combinations(leptons, 2, fields=["l0","l1"])
    ll_pairs_idx = ak.argcombinations(leptons, 2, fields=["l0","l1"])

    # Get distance from Z
    dist_from_z_all_pairs = abs((ll_pairs.l0+ll_pairs.l1).mass - 91.2)

    # Mask out the pairs that are not SFOS (so that we don't include them when finding the one that's closest to Z)
    # And then of the SFOS pairs, get the index of the one that's cosest to the Z
    sfos_mask = (ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId)
    dist_from_z_sfos_pairs = ak.mask(dist_from_z_all_pairs,sfos_mask)
    sfos_pair_closest_to_z_idx = ak.argmin(dist_from_z_sfos_pairs,axis=-1,keepdims=True)

    # Build a mask (of the shape of the original lep array) corresponding to the leps that are part of the Z candidate
    mask_is_z_lep = (leptons.idx == ak.flatten(ll_pairs_idx.l0[sfos_pair_closest_to_z_idx]))
    mask_is_z_lep = (mask_is_z_lep | (leptons.idx == ak.flatten(ll_pairs_idx.l1[sfos_pair_closest_to_z_idx])))
    mask_is_z_lep = ak.fill_none(mask_is_z_lep, False)

    # Get ahold of the leading non-Z lepton
    leps_not_from_z_candidate = leptons[~mask_is_z_lep]
    lead_lep_not_from_z_candidate = leps_not_from_z_candidate[ak.argmax(leps_not_from_z_candidate.pt, axis=1, keepdims=True)]
    lead_lep_not_from_z_candidate = lead_lep_not_from_z_candidate[:,0] # Go from e.g. [None,[lepton object]] to [None,lepton object]

    # Get the MT
    print("met phi",MET.phi)
    print("l3 phi",lead_lep_not_from_z_candidate.phi)
    print("d phi",MET.delta_phi(lead_lep_not_from_z_candidate))
    mt = np.sqrt(2 * lead_lep_not_from_z_candidate.pt * MET_pt * (1 - np.cos(MET.delta_phi(lead_lep_not_from_z_candidate))))

    # Apply 3l SFOS selection
    has_3l = ak.num(leptons) >=3
    has_sfos = ak.any(sfos_mask,axis=1)
    mt = mt[has_3l & has_sfos]

    # Fill hist
    q8_hist = hist.new.Reg(100, 0, 200, name="mt_lep_met", label="$\ell$-MET transverse mass [GeV]").Double()
    q8_hist.fill(mt_lep_met=mt)

    t1 = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q8_hist.plot1d(flow="none");
        fig.savefig("fig_q8_cpu.png")

    return(q8_hist,t1-t0)


####################################################################################################


def main():

    # Misc tests
    #check_hists()
    #check_combinations()
    #write_to_parquet()
    #lindsey_debugging_numba_cuda()

    # File paths
    # https://github.com/CoffeaTeam/coffea-benchmarks/blob/master/coffea-adl-benchmarks.ipynb
    #root_filepath = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu.root:Events"
    #filepath = "test_pq_10.parquet"
    #filepath = "test_pq_100.parquet"
    #filepath = "test_pq_1k.parquet"
    #filepath = "test_pq_100k.parquet"
    filepath = "test_pq_1M.parquet"
    #filepath = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN.parquet"

    # Dump just the first 100k events from Lindsey's file into a smaller file
    if 0:
        from pyarrow.parquet import ParquetFile
        import pyarrow as pa
        pf = ParquetFile(filepath)
        first_ten_rows = next(pf.iter_batches(batch_size = 10000000))
        df = pa.Table.from_batches([first_ten_rows]).to_pandas()
        df.to_parquet("test_pq_10M.parquet")
        print("Done")
        exit()

    # Run the benchmark queries on GPU
    hist_q1_gpu, t_q1_gpu = query1_gpu(filepath)
    hist_q2_gpu, t_q2_gpu = query2_gpu(filepath)
    hist_q3_gpu, t_q3_gpu = query3_gpu(filepath)
    hist_q4_gpu, t_q4_gpu = query4_gpu(filepath)
    hist_q5_gpu, t_q5_gpu = query5_gpu(filepath)
    hist_q6p1_gpu, hist_q6p2_gpu, t_q6_gpu = 0,0,0 #query6_gpu(filepath)
    hist_q7_gpu, t_q7_gpu = query7_gpu(filepath,makeplot=True)
    hist_q8_gpu, t_q8_gpu = 0, 0

    # Run the benchmark queries on CPU
    hist_q1_cpu, t_q1_cpu = query1_cpu(filepath)
    hist_q2_cpu, t_q2_cpu = query2_cpu(filepath)
    hist_q3_cpu, t_q3_cpu = query3_cpu(filepath)
    hist_q4_cpu, t_q4_cpu = query4_cpu(filepath)
    hist_q5_cpu, t_q5_cpu = query5_cpu(filepath)
    hist_q6p1_cpu, hist_q6p2_cpu, t_q6_cpu = query6_cpu(filepath)
    hist_q7_cpu, t_q7_cpu = query7_cpu(filepath,makeplot=True)
    hist_q8_cpu, t_q8_cpu = query8_cpu(filepath,makeplot=True)


    # Print the times
    print("gpu",[t_q1_gpu,t_q2_gpu,t_q3_gpu,t_q4_gpu,t_q5_gpu, t_q6_gpu, t_q7_gpu, t_q8_gpu])
    print("cpu",[t_q1_cpu,t_q2_cpu,t_q3_cpu,t_q4_cpu,t_q5_cpu, t_q6_cpu, t_q7_cpu, t_q8_cpu])

    exit()

    # Plotting the query outputs

    # Q1
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    hist_q1_cpu.plot1d(linewidth=3,color="orange",flow="none",label="cpu");
    hist_q1_gpu.to_hist().plot1d(linewidth=1,color="blue",flow="none",label="gpu");
    ax.legend(fontsize="12",framealpha=1)
    fig.savefig("fig_q1.png")

    # Q2
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    hist_q2_cpu.plot1d(linewidth=3,color="orange",flow="none",label="cpu");
    hist_q2_gpu.to_hist().plot1d(linewidth=1,color="blue",flow="none",label="gpu");
    ax.legend(fontsize="12",framealpha=1)
    fig.savefig("fig_q2.png")

    # Q3
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    hist_q3_cpu.plot1d(linewidth=3,color="orange",flow="none",label="cpu");
    hist_q3_gpu.to_hist().plot1d(linewidth=1,color="blue",flow="none",label="gpu");
    ax.legend(fontsize="12",framealpha=1)
    fig.savefig("fig_q3.png")

    # Q4
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    hist_q4_cpu.plot1d(linewidth=3,color="orange",flow="none",label="cpu");
    hist_q4_gpu.to_hist().plot1d(linewidth=1,color="blue",flow="none",label="gpu");
    ax.legend(fontsize="12",framealpha=1)
    fig.savefig("fig_q4.png")

    # Q5
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    hist_q5_cpu.plot1d(linewidth=3,color="orange",flow="none",label="cpu");
    hist_q5_gpu.to_hist().plot1d(linewidth=1,color="blue",flow="none",label="gpu");
    ax.legend(fontsize="12",framealpha=1)
    fig.savefig("fig_q5.png")

    '''
    # Q6

    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    hist_q6p1_cpu.plot1d(linewidth=3,color="orange",flow="none",label="cpu");
    hist_q6p1_gpu.to_hist().plot1d(linewidth=1,color="blue",flow="none",label="gpu");
    ax.legend(fontsize="12",framealpha=1)
    fig.savefig("fig_q6p1.png")

    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    hist_q6p2_cpu.plot1d(linewidth=3,color="orange",flow="none",label="cpu");
    hist_q6p2_gpu.to_hist().plot1d(linewidth=1,color="blue",flow="none",label="gpu");
    ax.legend(fontsize="12",framealpha=1)
    fig.savefig("fig_q6p2.png")
    '''



main()




