import time
import uproot
import awkward as ak

jetmet = uproot.open(
    #"/blue/p.chang/k.mohrman/fromLindsey/Run2012B_SingleMu.root:Events"
    #"root://cmsio2.rc.ufl.edu//cmsuf/data/store/user/t2/users/k.mohrman/test/Run2012B_SingleMu.root:Events"
    "DA1C387C-8EE5-0D48-9A16-79C0304EC3BC.root:Events"
    ).arrays(
        ["Jet_pt","MET_pt"],
    )

# Run with cpu
Jet_pt = ak.to_backend(jetmet.Jet_pt, "cpu")
#Jet_pt = Jet_pt[:1000000]
t1 = time.time()
jet_comb_out = ak.combinations(Jet_pt, 2)
t2 = time.time()
print(f"Time for ak.combinations for array of len {len(Jet_pt)} with backend \"{ak.backend(Jet_pt)}\": {t2-t1}s")

# Run with cuda
Jet_pt = ak.to_backend(jetmet.Jet_pt, "cuda")
#Jet_pt = Jet_pt[:1000000]
t3 = time.time()
jet_comb_out = ak.combinations(Jet_pt, 2)
t4 = time.time()
print(f"Time for ak.combinations for array of len {len(Jet_pt)} with backend \"{ak.backend(Jet_pt)}\": {t4-t3}s\n")

