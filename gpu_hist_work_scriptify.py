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

### Q1 query GPU ###
def query1_gpu(filepath):

    print("\nStarting Q1 code..")

    # Get met pt and fill hist
    t0 = time.time()
    MET_pt = cudf_to_awkward(cudf.read_parquet(filepath, columns = ["MET_pt"])["MET_pt"])
    q1_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
    )
    q1_hist.fill(met=MET_pt)
    t1 = time.time()

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    q1_hist.to_hist().plot1d(flow="none");
    fig.savefig("fig_q1.png")

    print(f"Time for q1: {t1-t0}")
    return(t1-t0)



### Q2 query GPU ###
def query2_gpu(filepath):

    print("\nStarting Q2 code..")

    t0 = time.time()
    #Jet_pt = ak.to_backend(uproot.open("/uscms_data/d2/lagray/Run2012B_SingleMu.root:Events")["Jet_pt"].array(), "cuda")
    #Jet_pt = ak.to_backend(ak.from_arrow(pq.read_table(filepath, columns=["Jet_pt"])["Jet_pt"]), "cuda")
    Jet_pt = cudf_to_awkward(cudf.read_parquet(filepath, columns = ["Jet_pt"])["Jet_pt"])
    q2_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("ptj", "Jet $p_{T}$ [GeV]", 100, 0, 200),
    )
    q2_hist.fill(ptj=ak.flatten(Jet_pt))
    t1 = time.time()

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    q2_hist.to_hist().plot1d(flow="none");
    fig.savefig("fig_q2.png")

    print(f"Time for q2: {t1-t0}")
    return(t1-t0)


### Q3 query GPU ###
def query3_gpu(filepath):

    print("\nStarting Q3 code..")

    t0 = time.time()
    table = cudf.read_parquet(filepath, columns = ["Jet_pt", "Jet_eta"])
    Jet_pt = cudf_to_awkward(table["Jet_pt"])
    Jet_eta = cudf_to_awkward(table["Jet_eta"])

    q3_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("ptj", "Jet $p_{T}$ [GeV]", 100, 0, 200),
    )
    q3_hist.fill(ptj=ak.flatten(Jet_pt[abs(Jet_eta) < 1.0]))
    t1 = time.time()

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    q3_hist.to_hist().plot1d(flow="none");
    fig.savefig("fig_q3.png")

    print(f"Time for q3: {t1-t0}")
    return(t1-t0)


### Q4 query GPU ###

def query4_gpu(filepath):

    print("\nStarting Q4 code..")

    t0 = time.time()
    table = cudf.read_parquet(filepath, columns = ["Jet_pt", "MET_pt"])
    Jet_pt = cudf_to_awkward(table["Jet_pt"])
    MET_pt = cudf_to_awkward(table["MET_pt"])

    q4_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
    )
    has2jets = ak.sum(Jet_pt > 40, axis=1) >= 2
    q4_hist.fill(met=MET_pt[has2jets])
    t1 = time.time()

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    q4_hist.to_hist().plot1d(flow="none");
    fig.savefig("fig_q4.png")

    print(f"Time for q4: {t1-t0}")
    return(t1-t0)



### Q5 query GPU ###

def query5_gpu(filepath):

    print("\nStarting Q5 code..")

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
    MET_pt = cudf_to_awkward(table["MET_pt"])
    Muon_pt = cudf_to_awkward(table["Muon_pt"])
    Muon_eta = cudf_to_awkward(table["Muon_eta"])
    Muon_phi = cudf_to_awkward(table["Muon_phi"])
    Muon_mass = cudf_to_awkward(table["Muon_mass"])
    Muon_charge = cudf_to_awkward(table["Muon_charge"])

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
    #)
    #)[0:10]
    )[0:10000]


    mupair = ak.combinations(Muon, 2, fields=["mu1", "mu2"])
    pairmass = (mupair.mu1 + mupair.mu2).mass
    goodevent = ak.any(
        (pairmass > 60)
        & (pairmass < 120)
        & (mupair.mu1.charge == -mupair.mu2.charge),
        axis=1,
    )


    #q5_hist.fill(MET_pt[goodevent])
    q5_hist.fill(met=MET_pt[goodevent])
    t1 = time.time()


    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    q5_hist.to_hist().plot1d(flow="none");
    fig.savefig("fig_q5.png")

    print(f"Time for q5: {t1-t0}")
    return(t1-t0)



####################################################################################################


def main():

    #check_hists()
    #check_combinations()
    #write_to_parquet()
    #lindsey_debugging_numba_cuda()

    # Benchmark queries
    # https://github.com/CoffeaTeam/coffea-benchmarks/blob/master/coffea-adl-benchmarks.ipynb
    root_filepath = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu.root:Events"
    filepath = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN.parquet"
    t_q1_gpu = query1_gpu(filepath)
    t_q2_gpu = query2_gpu(filepath)
    t_q3_gpu = query3_gpu(filepath)
    t_q4_gpu = query4_gpu(filepath)
    t_q5_gpu = query5_gpu(filepath)



main()




