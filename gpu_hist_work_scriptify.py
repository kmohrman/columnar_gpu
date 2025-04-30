import time
import awkward as ak
import cupy as cp
import numpy as np

from coffea.jitters import hist as gpu_hist

###########################

N_dims = 4

test_gpu = ak.Array(cp.random.multivariate_normal(
    mean=np.zeros(shape=N_dims), 
    cov=np.eye(N_dims),
    size=(200_000_000//N_dims),
))

###########################

hist_gpu = gpu_hist.Hist(
    "test", 
    gpu_hist.Bin("x", "x coordinate", 32, -5, 5),
    gpu_hist.Bin("y", "y coordinate", 32, -5, 5),
    gpu_hist.Bin("z", "z coordinate", 32, -5, 5),
    gpu_hist.Bin("t", "t coordinate", 32, -5, 5),
)


###########################

hist_gpu.fill(x=test_gpu[:, 0], y=test_gpu[:,1], z=test_gpu[:,2], t=test_gpu[:,3])

###########################

hist_gpu_cupy = cp.histogramdd(
    ak.to_cupy(test_gpu),
    bins=(32, 32, 32, 32),
    range=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)]
)

###########################

import hist

###########################

test_cpu = ak.to_backend(test_gpu, "cpu")

###########################

hist_cpu = hist.new.Reg(32, -5, 5).Reg(32, -5, 5).Reg(32, -5, 5).Reg(32, -5, 5).Weight()

###########################

hist_cpu.fill(test_cpu[:,0], test_cpu[:,1], test_cpu[:,2], test_cpu[:,3])

###########################

hist_gpu.values()[()].get()

###########################

#gpu_hist.plot1d(hist_gpu)
gpu_hist.plot1d(hist_gpu.project("z"))

###########################

import cudf
cudf.__version__

###########################

import uproot
import awkward as ak
#import cupy as cp
import cudf
import pyarrow.parquet as pq
import numpy as np
from coffea.jitters import hist

from ak_from_cudf import cudf_to_awkward

root_filepath = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu.root:Events"
#filepath = "/uscms_data/d3/fstrug/temp/Run2012B_SingleMu_compressed_zstd.parquet"
#filepath = "/uscms_data/d3/fstrug/temp/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN.parquet"
#filepath = "/uscms_data/d3/fstrug/temp/Run2012B_SingleMu_compressed_zstdlv3_Pv2-6_PPv2-0_PLAIN.parquet"
#filepath = "/uscms_data/d2/lagray/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN.parquet"
filepath = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN.parquet"

###########################

# Q1

#MET_pt = ak.to_backend(ak.from_arrow(pq.read_table(filepath, columns=["MET_pt"])["MET_pt"]), "cuda")
MET_pt = cudf_to_awkward(cudf.read_parquet(filepath, columns = ["MET_pt"])["MET_pt"])
q1_hist = hist.Hist(
    "Counts",
    hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
)
q1_hist.fill(met=MET_pt)

q1_hist.to_hist().plot1d(flow="none");

###########################

# Q2

#Jet_pt = ak.to_backend(uproot.open("/uscms_data/d2/lagray/Run2012B_SingleMu.root:Events")["Jet_pt"].array(), "cuda")
#Jet_pt = ak.to_backend(ak.from_arrow(pq.read_table(filepath, columns=["Jet_pt"])["Jet_pt"]), "cuda")
Jet_pt = cudf_to_awkward(cudf.read_parquet(filepath, columns = ["Jet_pt"])["Jet_pt"])
q2_hist = hist.Hist(
    "Counts",
    hist.Bin("ptj", "Jet $p_{T}$ [GeV]", 100, 0, 200),
)
q2_hist.fill(ptj=ak.flatten(Jet_pt))

q2_hist.to_hist().plot1d(flow="none");

###########################

# Q3

#jets = uproot.open(
#    "/uscms_data/d2/lagray/Run2012B_SingleMu.root:Events"
#).arrays(
#    ["Jet_pt","Jet_eta"],
#)
#Jet_pt = ak.to_backend(jets.Jet_pt, "cuda")
#Jet_eta = ak.to_backend(jets.Jet_eta, "cuda")

#table = ak.to_backend(ak.from_arrow(pq.read_table(filepath, columns=["Jet_pt", "Jet_eta"])), "cuda")
#Jet_pt = table.Jet_pt
#Jet_eta = table.Jet_eta

table = cudf.read_parquet(filepath, columns = ["Jet_pt", "Jet_eta"])
Jet_pt = cudf_to_awkward(table["Jet_pt"])
Jet_eta = cudf_to_awkward(table["Jet_eta"])

q3_hist = hist.Hist(
    "Counts",
    hist.Bin("ptj", "Jet $p_{T}$ [GeV]", 100, 0, 200),
)
q3_hist.fill(ptj=ak.flatten(Jet_pt[abs(Jet_eta) < 1.0]))

q3_hist.to_hist().plot1d(flow="none");

###########################

# Q4

#jetmet = uproot.open(
#    "/uscms_data/d2/lagray/Run2012B_SingleMu.root:Events"
#).arrays(
#    ["Jet_pt","MET_pt"],
#)

#table = ak.to_backend(ak.from_arrow(pq.read_table(filepath, columns=["Jet_pt", "MET_pt"])), "cuda")
#Jet_pt = table.Jet_pt
#MET_pt = table.MET_pt

table = cudf.read_parquet(filepath, columns = ["Jet_pt", "MET_pt"])
Jet_pt = cudf_to_awkward(table["Jet_pt"])
MET_pt = cudf_to_awkward(table["MET_pt"])

q4_hist = hist.Hist(
    "Counts",
    hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
)
has2jets = ak.sum(Jet_pt > 40, axis=1) >= 2
q4_hist.fill(met=MET_pt[has2jets])

q4_hist.to_hist().plot1d(flow="none");

###########################

# Q5

from coffea.nanoevents.methods import candidate

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

q5_hist = hist.Hist(
    "Counts",
    hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
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
#)[0:10]
)[0:1000]


mupair = ak.combinations(Muon, 2, fields=["mu1", "mu2"])
pairmass = (mupair.mu1 + mupair.mu2).mass
goodevent = ak.any(
    (pairmass > 60)
    & (pairmass < 120)
    & (mupair.mu1.charge == -mupair.mu2.charge),
    axis=1,
)

print("len(MET_pt)",len(MET_pt))
print("len(goodevent)",len(goodevent))
print("np.all(goodevent)",np.all(goodevent))
print("goodevent",goodevent)
print("MET_pt[goodevent]",MET_pt[goodevent])
print("MET_pt",MET_pt)

print("MET_pt[goodevent]",MET_pt[goodevent])
#q5_hist.fill(MET_pt[goodevent])
q5_hist.fill(met=MET_pt[goodevent])
print("done filling")


q5_hist.to_hist().plot1d(flow="none");
print("done plot1ding")

###########################

jetmet = uproot.open(
    "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu.root:Events"
).arrays(
    ["Jet_pt","MET_pt"],
)

print("HERE 0")
print(time.time())
#Jet_pt = ak.to_backend(jetmet.Jet_pt, "cuda")
Jet_pt = ak.to_backend(jetmet.Jet_pt, "cpu")
Jet_pt = Jet_pt[:1000000] # TMP

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
###########################

import fastparquet

###########################

from hepconvert import root_to_parquet

print("HERE 2 time.time root_to_parquet")
root_to_parquet(in_file = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu.root",
                out_file = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_00.parquet",
                tree="Events",
                compression = "zstd",
                compression_level = 3,
                extensionarray=False,
                parquet_version="2.6",
                parquet_page_version="2.0",
                parquet_extra_options = {"column_encoding": "PLAIN"}
               )

print("HERE 3")
###########################

import torch

###########################

import awkward as ak
import cupy as cp
import numba as nb

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

values2 = square_cuda_wrapped(values)

print(values2)

print("HERE 4")
###########################

import awkward as ak
import cupy as cp
import numpy as np

counts = cp.random.poisson(lam=3, size=5000000)
flat_values = cp.random.normal(size=int(counts.sum()))

values = ak.unflatten(flat_values, counts)

np_vals = np.abs(values)
print(np_vals, ak.backend(np_vals))

cp_vals = cp.abs(values)
print(cp_vals, ak.backend(cp_vals))

print("HERE 5")
###########################

values

###########################

dir(nb.cuda)

print("HERE 6")
###########################

dir(values2)

print("HERE 7")
###########################

import cupy as cp
import numpy as np

cp.float32 == np.float32

print("HERE 8")
###########################
