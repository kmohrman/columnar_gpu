import time
import awkward as ak
import cupy as cp
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

import cudf
from ak_from_cudf import cudf_to_awkward

import pyarrow.parquet as pq

import uproot
from coffea.jitters import hist as gpu_hist
import hist
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import vector

import pandas as df

print("cudf version", cudf.__version__)


####################################################################################################
### Helper functions ###

# Make a plot comparing the hists from CPU and GPU query
def make_comp_plot(h1,h2=None, h1_tag="CPU",h2_tag="GPU", h1_clr="orange",h2_clr="blue", name="test"):

    fig, ax = plt.subplots(1, 1, figsize=(7,7))

    # Assumes this is the CPU one, so directly call plot1d
    h1.plot1d(linewidth=4,flow="none",color=h1_clr,label=h1_tag);

    # Assumes this is gpu, so call to_hist before plot1d
    if h2 is not None:
        h2.to_hist().plot1d(linewidth=1.5,flow="none",color=h2_clr,label=h2_tag);

    ax.legend(fontsize="21",framealpha=1,frameon=False)
    plt.title(name)

    fig.savefig(f"plots/fig_{name}.png")
    fig.savefig(f"plots/fig_{name}.pdf")


# Workaround for argmin since not implemented on GPU
# Only tested for axis=1 (i.e., innermost, for an array with 2 axes)
# Use at your own risk
def argmin_workaround_axis1(in_arr,axis,keepdims=False):
    if axis != 1:
        raise Exception("Not tested for axis other than 1")
    min_mask = in_arr == ak.min(in_arr,axis=axis)
    min_idx = ak.firsts(ak.local_index(in_arr)[min_mask])
    if keepdims:
        return(ak.singletons(min_idx))
    else:
        return min_idx

# Check if arrays agree
def arrays_agree(inarr1,inarr2):
    arr1 = ak.to_backend(inarr1,"cpu")
    arr2 = ak.to_backend(inarr2,"cpu")

    # Check for exact agreement
    arr_agree = arr1 == arr2

    # Check for largest difference
    diff_arr = abs(arr1 - arr2)
    largest_diff = max(diff_arr)

    threshold = 0
    large_differences = diff_arr[diff_arr>threshold]

    idxmax = ak.argmax(diff_arr)
    print("arr1:",arr1)
    print("arr2:",arr2)
    print("idx of the max:", idxmax)
    print("val in arr1 of the max different:", arr1[idxmax])
    print("val in arr2 of the max different:", arr2[idxmax])
    print("large_differences:",large_differences)
    print("len large_differences:",len(large_differences))

    return(largest_diff)



####################################################################################################
### ADL queries ###

# Q1 query GPU
# Fill hist with met for all events
def query1_gpu(filepath,makeplot=False):

    print("\nStarting Q1 code on gpu..")

    # Get met pt and fill hist

    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["MET_pt"])

    cp.cuda.Device(0).synchronize()
    t_after_read = time.time() # Time

    MET_pt = cudf_to_awkward(table["MET_pt"])

    cp.cuda.Device(0).synchronize()
    t_after_load = time.time() # Time

    q1_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
    )
    q1_hist.fill(met=MET_pt)

    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q1_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q1_gpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q1: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q1_hist,MET_pt,[dt_after_read,dt_after_load,dt_after_fill,dt_tot])



# Q1 query CPU
# Fill hist with met for all events
def query1_cpu(filepath,makeplot=False):

    # Fill hist with met for all events
    print("\nStarting Q1 code on cpu..")

    # Get met pt and fill hist

    t0 = time.time()

    table = pq.read_table(filepath, columns = ["MET_pt"])
    t_after_read = time.time() # Time

    MET_pt = ak.Array(table["MET_pt"])
    t_after_load = time.time() # Time

    q1_hist = hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]").Double()
    q1_hist.fill(met=MET_pt)
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q1_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q1_cpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q1: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q1_hist,MET_pt,[dt_after_read,dt_after_load,dt_after_fill,dt_tot])



# Q2 query GPU
# Fill hist with pt for all jets
def query2_gpu(filepath,makeplot=False):

    print("\nStarting Q2 code on gpu..")

    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["Jet_pt"])

    cp.cuda.Device(0).synchronize()
    t_after_read = time.time() # Time

    Jet_pt = cudf_to_awkward(table["Jet_pt"])

    cp.cuda.Device(0).synchronize()
    t_after_load = time.time() # Time

    q2_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("ptj", "Jet $p_{T}$ [GeV]", 100, 0, 200),
    )
    fillarr = ak.flatten(Jet_pt)
    q2_hist.fill(ptj=fillarr)

    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q2_hist.to_hist().plot1d(flow="none");
        fig.savefig("plots/fig_q2_gpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q2: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q2_hist,fillarr, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])



# Q2 query CPU
# Fill hist with pt for all jets
def query2_cpu(filepath,makeplot=False):

    print("\nStarting Q2 code on cpu..")

    t0 = time.time()

    table = pq.read_table(filepath, columns = ["Jet_pt"])
    t_after_read = time.time() # Time

    Jet_pt = ak.Array(table["Jet_pt"])
    t_after_load = time.time() # Time

    q2_hist = hist.new.Reg(100, 0, 200, name="ptj", label="Jet $p_{T}$ [GeV]").Double()
    fillarr = ak.flatten(Jet_pt)
    q2_hist.fill(ptj=fillarr)
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q2_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q2_cpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q2: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q2_hist,fillarr, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])



# Q3 query GPU
# Fill a hist with pt of jets with eta less than 1
def query3_gpu(filepath,makeplot=False):

    print("\nStarting Q3 code on gpu..")

    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["Jet_pt", "Jet_eta"])

    cp.cuda.Device(0).synchronize()
    t_after_read = time.time() # Time

    Jet_pt = cudf_to_awkward(table["Jet_pt"])
    Jet_eta = cudf_to_awkward(table["Jet_eta"])

    cp.cuda.Device(0).synchronize()
    t_after_load = time.time() # Time

    q3_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("ptj", "Jet $p_{T}$ [GeV]", 100, 0, 200),
    )
    fillarr = ak.flatten(Jet_pt[abs(Jet_eta) < 1.0])
    q3_hist.fill(ptj=fillarr)

    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q3_hist.to_hist().plot1d(flow="none");
        fig.savefig("plots/fig_q3_gpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q3: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q3_hist,fillarr, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])



# Q3 query CPU
# Fill a hist with pt of jets with eta less than 1
def query3_cpu(filepath,makeplot=False):

    print("\nStarting Q3 code on cpu..")

    t0 = time.time()

    table = pq.read_table(filepath, columns = ["Jet_pt", "Jet_eta"])
    t_after_read = time.time() # Time

    Jet_pt = ak.Array(table["Jet_pt"])
    Jet_eta = ak.Array(table["Jet_eta"])
    t_after_load = time.time() # Time

    q3_hist = hist.new.Reg(100, 0, 200, name="ptj", label="Jet $p_{T}$ [GeV]").Double()
    fillarr = ak.flatten(Jet_pt[abs(Jet_eta) < 1.0])
    q3_hist.fill(ptj=fillarr)

    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q3_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q3_cpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q3: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q3_hist,fillarr, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])



# Q4 query GPU
# Fill a hist with MET of events that have at least two jets with pt>40
def query4_gpu(filepath,makeplot=False):

    print("\nStarting Q4 code on gpu..")

    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["Jet_pt", "MET_pt"])

    cp.cuda.Device(0).synchronize()
    t_after_read = time.time() # Time

    Jet_pt = cudf_to_awkward(table["Jet_pt"])
    MET_pt = cudf_to_awkward(table["MET_pt"])

    cp.cuda.Device(0).synchronize()
    t_after_load = time.time() # Time

    q4_hist = gpu_hist.Hist(
        "Counts",
        gpu_hist.Bin("met", "$E_{T}^{miss}$ [GeV]", 100, 0, 200),
    )
    has2jets = ak.sum(Jet_pt > 40, axis=1) >= 2
    fillarr = MET_pt[has2jets]
    q4_hist.fill(met=fillarr)

    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q4_hist.to_hist().plot1d(flow="none");
        fig.savefig("plots/fig_q4_gpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q4: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q4_hist,fillarr, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])


# Q4 query CPU
# Fill a hist with MET of events that have at least two jets with pt>40
def query4_cpu(filepath,makeplot=False):

    print("\nStarting Q4 code on cpu..")

    t0 = time.time()

    table = pq.read_table(filepath, columns = ["Jet_pt", "MET_pt"])
    t_after_read = time.time() # Time

    Jet_pt = ak.Array(table["Jet_pt"])
    MET_pt = ak.Array(table["MET_pt"])
    t_after_load = time.time() # Time

    q4_hist = hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]").Double()
    has2jets = ak.sum(Jet_pt > 40, axis=1) >= 2
    fillarr = MET_pt[has2jets]
    q4_hist.fill(met=fillarr)

    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q4_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q4_cpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q4: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q4_hist,fillarr, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])



# Q5 query GPU
# Fill a hist with MET For events that have an OS muon pair with an invariant mass between 60 and 120 GeV
def query5_gpu(filepath,makeplot=False):

    print("\nStarting Q5 code on gpu..")

    cp.cuda.Device(0).synchronize()
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

    cp.cuda.Device(0).synchronize()
    t_after_read = time.time() # Time

    MET_pt = cudf_to_awkward(table["MET_pt"])
    Muon_pt = cudf_to_awkward(table["Muon_pt"])
    Muon_eta = cudf_to_awkward(table["Muon_eta"])
    Muon_phi = cudf_to_awkward(table["Muon_phi"])
    Muon_mass = cudf_to_awkward(table["Muon_mass"])
    Muon_charge = cudf_to_awkward(table["Muon_charge"])

    cp.cuda.Device(0).synchronize()
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


    mupair = ak.combinations(Muon, 2, fields=["mu1", "mu2"])
    pairmass = (mupair.mu1 + mupair.mu2).mass
    goodevent = ak.any(
        (pairmass > 60)
        & (pairmass < 120)
        & (mupair.mu1.charge == -mupair.mu2.charge),
        axis=1,
    )

    fillarr = MET_pt[goodevent]
    q5_hist.fill(met=fillarr)

    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()


    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q5_hist.to_hist().plot1d(flow="none");
        fig.savefig("plots/fig_q5_gpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q5: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q5_hist,fillarr, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])



# Q5 query CPU
# Fill a hist with MET For events that have an OS muon pair with an invariant mass between 60 and 120 GeV
def query5_cpu(filepath,makeplot=False):

    print("\nStarting Q5 code on cpu..")

    t0 = time.time()
    table = pq.read_table(
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

    mupair = ak.combinations(Muon, 2, fields=["mu1", "mu2"])
    pairmass = (mupair.mu1 + mupair.mu2).mass
    goodevent = ak.any(
        (pairmass > 60)
        & (pairmass < 120)
        & (mupair.mu1.charge == -mupair.mu2.charge),
        axis=1,
    )

    fillarr = MET_pt[goodevent]
    q5_hist.fill(met=fillarr)
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q5_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q5_cpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q5: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q5_hist,fillarr, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])


# Q6 query GPU
# Select events at least 3 jets
#   - Fill hist with pt of tri-jet system closest to top mass
#   - Fill hist with max b-tag score of the jets in the system
def query6_gpu(filepath,makeplot=False):

    print("\nStarting Q6 code on gpu..")

    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_btag",])

    cp.cuda.Device(0).synchronize()
    t_after_read = time.time() # Time

    Jet_pt = cudf_to_awkward(table["Jet_pt"])
    Jet_eta = cudf_to_awkward(table["Jet_eta"])
    Jet_phi = cudf_to_awkward(table["Jet_phi"])
    Jet_mass = cudf_to_awkward(table["Jet_mass"])
    Jet_btag = cudf_to_awkward(table["Jet_btag"])

    cp.cuda.Device(0).synchronize()
    t_after_load = time.time() # Time

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

    trijet_t = ak.flatten(
        trijet[ak.singletons(ak.argmin(abs(trijet.p4.mass - 172.5), axis=1))]
    )

    # Get max btag of the trijet system
    maxBtag = np.maximum(
        trijet_t.j1.btag,
        np.maximum(
            trijet_t.j2.btag,
            trijet_t.j3.btag,
        ),
    )

    q6_hist_1 = gpu_hist.Hist("Counts", gpu_hist.Bin("pt3j", "Trijet $p_{T}$ [GeV]", 100, 0, 200))
    q6_hist_1.fill(pt3j=trijet_t.p4.pt)

    q6_hist_2 = gpu_hist.Hist("Counts", gpu_hist.Bin("btag", "Max jet b-tag score", 100, -10, 1))
    q6_hist_2.fill(btag=maxBtag)

    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        # First hist
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q6_hist_1.to_hist().plot1d(flow="none");
        fig.savefig("plots/fig_q6p1_gpu.png")
        # Second hist
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q6_hist_2.to_hist().plot1d(flow="none");
        fig.savefig("plots/fig_q6p2_gpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q6: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q6_hist_1, q6_hist_2, trijet_t.p4.pt, maxBtag, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])


# Q6 query CPU
# Select events at least 3 jets
#   - Fill hist with pt of tri-jet system closest to top mass
#   - Fill hist with max b-tag score of the jets in the system
def query6_cpu(filepath,makeplot=False):

    print("\nStarting Q6 code on cpu..")

    t0 = time.time()

    table = pq.read_table(filepath, columns = ["Jet_pt","Jet_eta","Jet_phi","Jet_mass","Jet_btag"])
    t_after_read = time.time() # Time

    Jet_pt = ak.Array(table["Jet_pt"])
    Jet_eta = ak.Array(table["Jet_eta"])
    Jet_phi = ak.Array(table["Jet_phi"])
    Jet_mass = ak.Array(table["Jet_mass"])
    Jet_btag = ak.Array(table["Jet_btag"])
    t_after_load = time.time() # Time

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

    trijet_t = ak.flatten(
        trijet[ak.singletons(ak.argmin(abs(trijet.p4.mass - 172.5), axis=1))]
    )

    # Get max btag of the trijet system
    maxBtag = np.maximum(
        trijet_t.j1.btag,
        np.maximum(
            trijet_t.j2.btag,
            trijet_t.j3.btag,
        ),
    )

    q6_hist_1 = hist.new.Reg(100, 0, 200, name="pt3j", label="Trijet $p_{T}$ [GeV]").Double()
    q6_hist_1.fill(pt3j=trijet_t.p4.pt)

    q6_hist_2 = hist.new.Reg(100, -10, 1, name="btag", label="Max jet b-tag score").Double()
    q6_hist_2.fill(btag=maxBtag)

    t_after_fill = time.time()

    # Plotting
    if makeplot:
        # First hist
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q6_hist_1.plot1d(flow="none");
        fig.savefig("plots/fig_q6p1_cpu.png")
        # Second hist
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q6_hist_2.plot1d(flow="none");
        fig.savefig("plots/fig_q6p2_cpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q6: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q6_hist_1, q6_hist_2, trijet_t.p4.pt, maxBtag, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])


# Q7 query GPU
# Fill hist with HT of jets
#   - Jets have pt>30 and far (dR>0.4) from leptons
#   - Leptons have pt>10
def query7_gpu(filepath,makeplot=False):

    print("\nStarting Q7 code on gpu..")

    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = [
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_charge",
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass"
    ])

    cp.cuda.Device(0).synchronize()
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

    cp.cuda.Device(0).synchronize()
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

    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q7_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q7_cpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q7: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q7_hist, ht, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])


# Q7 query CPU
# Fill hist with HT of jets
#   - Jets have pt>30 and far (dR>0.4) from leptons
#   - Leptons have pt>10
def query7_cpu(filepath,makeplot=False):

    print("\nStarting Q7 code on cpu..")

    t0 = time.time()

    table = pq.read_table(filepath, columns = [
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

    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q7_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q7_cpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q7: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q7_hist, ht, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])


# Q8 query GPU
# Select events with at least 3 leptons, that inlude a SFOS pair
# Plot MT of the system of the leading non-Z lepton and MET
def query8_gpu(filepath,makeplot=False):

    print("\nStarting Q8 code on gpu..")

    cp.cuda.Device(0).synchronize()
    t0 = time.time()

    table = cudf.read_parquet(filepath, columns = [
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_charge",
        "MET_pt", "MET_phi",
    ])

    cp.cuda.Device(0).synchronize()
    t_after_read = time.time() # Time

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

    MET_pt  = cudf_to_awkward(table["MET_pt"])
    MET_phi = cudf_to_awkward(table["MET_phi"])

    cp.cuda.Device(0).synchronize()
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
    print("this!",leps_not_from_z_candidate)
    lead_lep_not_from_z_candidate = leps_not_from_z_candidate[ak.argmax(leps_not_from_z_candidate.pt, axis=1, keepdims=True)]
    print("made it here!")
    lead_lep_not_from_z_candidate = lead_lep_not_from_z_candidate[:,0] # Go from e.g. [None,[lepton object]] to [None,lepton object]

    # Get the MT
    mt = np.sqrt(2 * lead_lep_not_from_z_candidate.pt * MET_pt * (1 - np.cos(MET.delta_phi(lead_lep_not_from_z_candidate))))

    # Apply 3l SFOS selection
    has_3l = ak.num(leptons) >=3
    has_sfos = ak.any(sfos_mask,axis=1)
    mt = mt[has_3l & has_sfos]

    # Fill hist
    q8_hist = gpu_hist.Hist("Counts", gpu_hist.Bin("mt_lep_met", "lep-MET transverse mass [GeV]", 100, 0, 200))
    q8_hist.fill(mt_lep_met=mt)

    cp.cuda.Device(0).synchronize()
    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q8_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q8_gpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q8: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q8_hist, mt, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])


# Q8 query CPU
# Select events with at least 3 leptons, that inlude a SFOS pair
# Plot MT of the system of the leading non-Z lepton and MET
def query8_cpu(filepath,makeplot=False):

    print("\nStarting Q8 code on cpu..")

    t0 = time.time()

    table = pq.read_table(filepath, columns = [
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
    print("this!",leps_not_from_z_candidate)
    lead_lep_not_from_z_candidate = leps_not_from_z_candidate[ak.argmax(leps_not_from_z_candidate.pt, axis=1, keepdims=True)]
    print("made it!")
    lead_lep_not_from_z_candidate = lead_lep_not_from_z_candidate[:,0] # Go from e.g. [None,[lepton object]] to [None,lepton object]

    # Get the MT
    mt = np.sqrt(2 * lead_lep_not_from_z_candidate.pt * MET_pt * (1 - np.cos(MET.delta_phi(lead_lep_not_from_z_candidate))))

    # Apply 3l SFOS selection
    has_3l = ak.num(leptons) >=3
    has_sfos = ak.any(sfos_mask,axis=1)
    mt = mt[has_3l & has_sfos]

    # Fill hist
    q8_hist = hist.new.Reg(100, 0, 200, name="mt_lep_met", label="lep-MET transverse mass [GeV]").Double()
    q8_hist.fill(mt_lep_met=mt)

    t_after_fill = time.time()

    # Plotting
    if makeplot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        q8_hist.plot1d(flow="none");
        fig.savefig("plots/fig_q8_cpu.png")

    # Timing information
    dt_after_read = t_after_read-t0
    dt_after_load = t_after_load-t_after_read
    dt_after_fill = t_after_fill-t_after_load
    dt_tot        = t_after_fill-t0
    print(f"Time for q8: {dt_tot}")
    print(f"    Time for reading: {dt_after_read} ({np.round(100*(dt_after_read)/(dt_tot),1)}%)")
    print(f"    Time for loading: {dt_after_load} ({np.round(100*(dt_after_load)/(dt_tot),1)}%)")
    print(f"    Time for computing and histing: {dt_after_fill} ({np.round(100*(dt_after_fill)/(dt_tot),1)}%)")

    return(q8_hist, mt, [dt_after_read,dt_after_load,dt_after_fill,dt_tot])



####################################################################################################


def main():

    # File paths
    ## https://github.com/CoffeaTeam/coffea-benchmarks/blob/master/coffea-adl-benchmarks.ipynb
    ##root_filepath = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu.root:Events"
    ##filepath = "/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN.parquet"
    filepath = "/blue/p.chang/k.mohrman/coffea_rd/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets/pq_subset_100k.parquet"
    #filepath = "/blue/p.chang/k.mohrman/coffea_rd/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets/pq_subset_1M.parquet"
    #filepath = "/blue/p.chang/k.mohrman/coffea_rd/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets/pq_subset_10M.parquet"
    #filepath = "/blue/p.chang/k.mohrman/coffea_rd/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN_subsets/Run2012B_SingleMu_compressed_zstdlv3_PPv2-0_PLAIN.parquet"

    # Print the number of events we are running over
    nevents = len(df.read_parquet(filepath, columns = ["MET_pt"]))
    print(f"\nNumber of nevents to be processed: {nevents}")

    # Placeholder timing array for the not yet implemented queries
    zeros = [0,0,0,0]

    # Run the benchmark queries on GPU
    hist_q1_gpu, arr_q1_gpu, t_q1_gpu = query1_gpu(filepath)
    hist_q2_gpu, arr_q2_gpu, t_q2_gpu = query2_gpu(filepath)
    hist_q3_gpu, arr_q3_gpu, t_q3_gpu = query3_gpu(filepath)
    hist_q4_gpu, arr_q4_gpu, t_q4_gpu = query4_gpu(filepath)
    hist_q5_gpu, arr_q5_gpu, t_q5_gpu = query5_gpu(filepath)
    hist_q6p1_gpu, hist_q6p2_gpu, arr_q6p1_gpu, arr_q6p2_gpu, t_q6_gpu = query6_gpu(filepath)
    hist_q7_gpu, arr_q7_gpu, t_q7_gpu = query7_gpu(filepath)
    hist_q8_gpu, arr_q8_gpu, t_q8_gpu = None,None,zeros #query8_gpu(filepath)
    #hist_q8_gpu, arr_q8_gpu, t_q8_gpu = query8_gpu(filepath)

    # Run the benchmark queries on CPU
    hist_q1_cpu,   arr_q1_cpu, t_q1_cpu = query1_cpu(filepath)
    hist_q2_cpu,   arr_q2_cpu, t_q2_cpu = query2_cpu(filepath)
    hist_q3_cpu,   arr_q3_cpu, t_q3_cpu = query3_cpu(filepath)
    hist_q4_cpu,   arr_q4_cpu, t_q4_cpu = query4_cpu(filepath)
    hist_q5_cpu,   arr_q5_cpu, t_q5_cpu = query5_cpu(filepath)
    hist_q6p1_cpu, hist_q6p2_cpu, arr_q6p1_cpu, arr_q6p2_cpu, t_q6_cpu = query6_cpu(filepath)
    hist_q7_cpu,   arr_q7_cpu, t_q7_cpu = query7_cpu(filepath)
    hist_q8_cpu,   arr_q8_cpu, t_q8_cpu = query8_cpu(filepath)


    #t = (ak.to_backend(arr_q1_gpu,"cpu") == arr_q1_cpu)
    print("q1:",arrays_agree(arr_q1_gpu,arr_q1_cpu),"\n")
    print("q2:",arrays_agree(arr_q2_gpu,arr_q2_cpu),"\n")
    print("q3:",arrays_agree(arr_q3_gpu,arr_q3_cpu),"\n")
    print("q4:",arrays_agree(arr_q4_gpu,arr_q4_cpu),"\n")
    print("q5:",arrays_agree(arr_q5_gpu,arr_q5_cpu),"\n")
    print("q6:",arrays_agree(arr_q6p1_gpu,arr_q6p1_cpu),"\n")
    print("q6:",arrays_agree(arr_q6p2_gpu,arr_q6p2_cpu),"\n")
    print("q7:",arrays_agree(arr_q7_gpu,arr_q7_cpu),"\n")
    #print("q8:",arrays_agree(arr_q8_gpu,arr_q8_cpu),"\n")
    exit()

    # Print the times in a way we can easily paste as the plotting inputs
    print(f"\nTiming info for this run over {nevents} events:")
    print(f"gpu:\n{[t_q1_gpu,t_q2_gpu,t_q3_gpu,t_q4_gpu,t_q5_gpu,t_q6_cpu,                 ]},")
    print(f"cpu:\n{[t_q1_cpu,t_q2_cpu,t_q3_cpu,t_q4_cpu,t_q5_cpu,t_q6_cpu,t_q7_cpu,t_q8_cpu]},")

    # Plotting the query output histos
    print("Making plots...")
    make_comp_plot(h1=hist_q1_cpu,   h2=hist_q1_gpu,   name=f"query1_nevents{nevents}")
    make_comp_plot(h1=hist_q2_cpu,   h2=hist_q2_gpu,   name=f"query2_nevents{nevents}")
    make_comp_plot(h1=hist_q3_cpu,   h2=hist_q3_gpu,   name=f"query3_nevents{nevents}")
    make_comp_plot(h1=hist_q4_cpu,   h2=hist_q4_gpu,   name=f"query4_nevents{nevents}")
    make_comp_plot(h1=hist_q5_cpu,   h2=hist_q5_gpu,   name=f"query5_nevents{nevents}")
    make_comp_plot(h1=hist_q6p1_cpu, h2=hist_q6p1_gpu, name=f"query6_part1_nevents{nevents}")
    make_comp_plot(h1=hist_q6p2_cpu, h2=hist_q6p2_gpu, name=f"query6_part2_nevents{nevents}")
    make_comp_plot(h1=hist_q7_cpu,   h2=hist_q7_gpu,   name=f"query7_nevents{nevents}")
    make_comp_plot(h1=hist_q8_cpu,   h2=hist_q8_gpu,   name=f"query8_nevents{nevents}")
    print("Done")



main()




