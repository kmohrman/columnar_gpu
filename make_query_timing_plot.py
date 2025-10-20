import os
import numpy as np
import matplotlib.pyplot as plt

# Scatter plot to compare Q times
def make_scatter_plot(xarr_1,xarr_2,yarr_1_lst,yarr_2_lst,log=False,xaxis_name="x",yaxis_name="y",tag1="set1",tag2="set2",save_name="test",nevents=None):

    #fig, axs = plt.subplots(nrows=1, ncols=1)

    # Create the figure
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(7,7),
        gridspec_kw={"height_ratios": (1, 1, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.09)

    # Loop over the sets of events and plot them
    if len(yarr_1_lst) != len(yarr_2_lst): raise Exception("Number of sets of points to plot do not agree between cpu and gpu")
    for i in range(len(yarr_1_lst)):
        print(f"Plotting for set {i}...")
        # Plot the data on main plot
        if i==0:
            ax1.scatter(xarr_1,yarr_1_lst[i],color="orange",edgecolors='orange',facecolors="none",label=tag1,zorder=100)
            ax1.scatter(xarr_2,yarr_2_lst[i],color="blue",edgecolors='blue',facecolors="none",label=tag2,zorder=100)
        else:
            ax1.scatter(xarr_1,yarr_1_lst[i],color="orange",edgecolors='orange',facecolors="none",zorder=100)
            ax1.scatter(xarr_2,yarr_2_lst[i],color="blue",edgecolors='blue',facecolors="none",zorder=100)

        # Plot events/s in kHz
        ax2.scatter(xarr_1,nevents/(1000*yarr_1_lst[i]),color="orange",edgecolors='orange',facecolors="none",zorder=100)
        ax2.scatter(xarr_2,nevents/(1000*yarr_2_lst[i]),color="blue",edgecolors='blue',facecolors="none",zorder=100)

        # Plot the ratio on the ratio plot
        min_len_x = min(len(xarr_1),len(xarr_2))
        r_arr = yarr_1_lst[i][:min_len_x]/yarr_2_lst[i][:min_len_x]
        ax3.scatter(xarr_1[:min_len_x],r_arr,color="orange",edgecolors='orange',facecolors="none",zorder=100)

    # Set log scale
    if log:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        save_name = save_name+"_log"

    # Set titles and such
    ax1.set_ylabel(yaxis_name)
    ax2.set_ylabel(f"Events/s [kHz]")
    ax1.legend(fontsize="12",framealpha=1)
    ax1.set_title(save_name)
    ax1.grid(zorder=-99)
    ax2.grid(zorder=-99)
    ax3.grid(zorder=-99)
    ax3.axhline(1.0,linestyle="-",color="k",linewidth=1)
    ax3.set_ylabel(f"{tag1}/{tag2}")
    ax3.set_xlabel(xaxis_name)

    #ax1.set_ylim(0.1,10e4)
    #ax2.set_ylim(1,10e5)
    #ax3.set_ylim(-100,1050)

    #ax3.yaxis.set_major_locator(5)
    #plt.yticks(5)
    plt.locator_params(axis="y", nbins=5) 


    plt.savefig(os.path.join(f"plots/{save_name}.png"),format="png")
    plt.savefig(os.path.join(f"plots/{save_name}.pdf"),format="pdf")
    #plt.show()
    return plt



def main():


    # Dict to keep track of the order of the dt time values printed in the timing array
    # The structure of the np array is like:
    # np.array([
    #    [read,load,fill,total times for query 1],
    #    [read,load,fill,total times for query 2], 
    #    ...
    # ]
    time_lables_dict = {
        0 : "read",
        1 : "load",
        2 : "compute",
        3 : "fill",
        4 : "total",
    }

    # Queries (for x axis)
    x_gpu = [1,2,3,4,5,6,7]
    x_cpu = [1,2,3,4,5,6,7,8]

    # Timing numbers
    timing_dict = {

        # Run 100k, about 3m
        "100k" : {
            "nevents" : 100000,
            "y_gpu_arr" : np.array([
                # Example
                [[2.144270420074463, 0.0052378177642822266, 1.9073486328125e-06, 0.09120798110961914, 2.240718126296997], [0.02709674835205078, 0.0020318031311035156, 1.430511474609375e-06, 0.0037124156951904297, 0.032842397689819336], [0.023229360580444336, 0.0006740093231201172, 3.509768486022949, 0.001636505126953125, 3.535308361053467], [0.022994518280029297, 0.0005764961242675781, 0.008098840713500977, 0.0008261203765869141, 0.032495975494384766], [0.013553380966186523, 0.0013649463653564453, 2.219862461090088, 0.0008931159973144531, 2.2356739044189453], [0.020303726196289062, 0.0013508796691894531, 0.33686280250549316, 0.0008966922760009766, 0.35941410064697266], [0.025888442993164062, 0.003199338912963867, 0.1412034034729004, 0.0012288093566894531, 0.17151999473571777]],
            ]),
            "y_cpu_arr" : np.array([
                # Example
                [[0.00448298454284668, 0.0005655288696289062, 0.0, 0.0009770393371582031, 0.006025552749633789], [0.007097482681274414, 0.0003719329833984375, 4.76837158203125e-07, 0.0015561580657958984, 0.009026050567626953], [0.011201858520507812, 0.0005850791931152344, 0.00941157341003418, 0.0005500316619873047, 0.02174854278564453], [0.006860971450805664, 0.0004546642303466797, 0.003831148147583008, 0.00032639503479003906, 0.01147317886352539], [0.009485483169555664, 0.001073598861694336, 0.02265453338623047, 0.0003077983856201172, 0.033521413803100586], [0.02029132843017578, 0.0012133121490478516, 0.3332183361053467, 0.0009164810180664062, 0.3556394577026367], [0.02793097496032715, 0.0024673938751220703, 0.0631263256072998, 0.0006544589996337891, 0.09417915344238281], [0.014684438705444336, 0.0017819404602050781, 0.06895589828491211, 0.00032782554626464844, 0.08575010299682617]]
            ]),
        },

        "1M" : {
            "nevents" : 1e6,
            "y_gpu_arr" : np.array([
            ]),
            "y_cpu_arr" : np.array([
            ]),
        },

        # Takes about 40m to run
        "10M" : {
            "nevents" : 1e6,
            "y_gpu_arr" : np.array([
                # Jul 2025: Not first run in the session (had run on smaller sample first), sync time for GPU, pyarrow for cpu
                #[[1.5815753936767578, 0.00156402587890625, 0.07590317726135254, 1.6590425968170166], [0.07701683044433594, 0.0016074180603027344, 0.038376569747924805, 0.11700081825256348], [0.09454607963562012, 0.0021135807037353516, 3.0548038482666016, 3.151463508605957], [0.07436060905456543, 0.001234292984008789, 0.03155684471130371, 0.10715174674987793], [0.13285064697265625, 0.005905628204345703, 1.1622741222381592, 1.3010303974151611]],
                #[[1.6861388683319092, 0.0018656253814697266, 0.07868528366088867, 1.7666897773742676], [0.07832741737365723, 0.0018076896667480469, 0.03959178924560547, 0.11972689628601074], [0.09486174583435059, 0.0021278858184814453, 3.0948309898376465, 3.1918206214904785], [0.07574081420898438, 0.001272439956665039, 0.0317685604095459, 0.10878181457519531], [0.12839722633361816, 0.005967378616333008, 1.1795237064361572, 1.3138883113861084]],
                #[[2.076409101486206, 0.00516200065612793, 4.802795648574829, 6.884366750717163], [0.5489840507507324, 0.002950906753540039, 0.0398101806640625, 0.591745138168335], [0.09500670433044434, 0.0021047592163085938, 5.1707375049591064, 5.267848968505859], [0.07592463493347168, 0.0012633800506591797, 0.0326848030090332, 0.10987281799316406], [0.6212971210479736, 0.005972623825073242, 0.7522389888763428, 1.3795087337493896]],
            ]),
            "y_cpu_arr" : np.array([
                # Jul 2025: Not first run in the session (had run on smaller sample first), sync time for GPU, pyarrow for cpu
                #[[0.04200387001037598, 0.09935307502746582, 0.056989431381225586, 0.19834637641906738], [0.3683950901031494, 4.350168228149414, 0.19568085670471191, 4.914244174957275], [0.6780776977539062, 8.776479244232178, 1.2706658840179443, 10.725222826004028], [0.3434610366821289, 4.401387453079224, 0.39362525939941406, 5.138473749160767], [0.6263217926025391, 8.871382713317871, 1.6156229972839355, 11.113327503204346], [1.7731642723083496, 21.914700508117676, 72.51578760147095, 96.20365238189697], [2.593351364135742, 27.898444890975952, 9.290273666381836, 39.78206992149353], [1.289369821548462, 10.471790552139282, 6.647113084793091, 18.408273458480835]],
                #[[0.041960954666137695, 0.09601473808288574, 0.056436777114868164, 0.1944124698638916], [0.3701472282409668, 4.3522789478302, 0.19462251663208008, 4.917048692703247], [0.6835751533508301, 8.78408932685852, 1.258218765258789, 10.72588324546814], [0.3445723056793213, 4.429948806762695, 0.39241695404052734, 5.166938066482544], [0.6315445899963379, 8.815284013748169, 1.615509033203125, 11.062337636947632], [1.740649700164795, 22.061144590377808, 72.15735483169556, 95.95914912223816], [2.342261552810669, 27.87834119796753, 9.386913537979126, 39.607516288757324], [1.0867652893066406, 10.180213451385498, 6.834189176559448, 18.101167917251587]],
                #[[0.07789087295532227, 0.09351325035095215, 0.057554006576538086, 0.2289581298828125], [0.3616511821746826, 4.343311071395874, 0.18419861793518066, 4.889160871505737], [0.6643061637878418, 8.726859331130981, 1.2653675079345703, 10.656533002853394], [0.3326690196990967, 4.400164842605591, 0.3871748447418213, 5.120008707046509], [0.6314654350280762, 8.807303428649902, 1.657123327255249, 11.095892190933228], [1.7669188976287842, 21.67437195777893, 72.2206289768219, 95.66191983222961], [2.3154752254486084, 27.49279236793518, 9.342978954315186, 39.151246547698975], [1.078124761581421, 10.38680124282837, 6.869341135025024, 18.334267139434814]],
            ]),
        },

        # Never run to full completion on CPU, many hours
        "53M" : {
            "nevents" : 53446198,
            "y_gpu_arr" : np.array([
            ]),
            "y_cpu_arr" : np.array([
            ]),
        },
    }

    # Make the plots
    #cat_to_plot = "10M"
    cat_to_plot = "100k"
    for dt_category_idx,dt_category_label in time_lables_dict.items():
        nevts = timing_dict[cat_to_plot]["nevents"]
        y_gpu = timing_dict[cat_to_plot]["y_gpu_arr"][:,:,dt_category_idx]
        y_cpu = timing_dict[cat_to_plot]["y_cpu_arr"][:,:,dt_category_idx]
        make_scatter_plot(x_cpu,x_gpu,y_cpu,y_gpu,xaxis_name=f"Benchmark Queries ({dt_category_label})",yaxis_name="Runtime [s]", tag1="CPU", tag2="GPU",save_name=f"adl_benchmarks_nEvents{cat_to_plot}_{dt_category_label}",log=False,nevents=nevts)
        make_scatter_plot(x_cpu,x_gpu,y_cpu,y_gpu,xaxis_name=f"Benchmark Queries ({dt_category_label})",yaxis_name="Runtime [s]", tag1="CPU", tag2="GPU",save_name=f"adl_benchmarks_nEvents{cat_to_plot}_{dt_category_label}",log=True,nevents=nevts)


main()




