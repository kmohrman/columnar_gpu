import numpy as np
import matplotlib.pyplot as plt

# Scatter plot to compare Q times
def make_scatter_plot(x_arr,y_arr_1_lst,y_arr_2_lst,log=False,xaxis_name="x",yaxis_name="y",tag1="set1",tag2="set2",save_name="test",nevents=None):

    #fig, axs = plt.subplots(nrows=1, ncols=1)

    # Create the figure
    fig, (ax, rax, rrax) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(7,7),
        gridspec_kw={"height_ratios": (1, 1, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.09)

    # Loop over the sets of events and plot them
    for i in range(len(y_arr_1_lst)):
        print(f"Plotting for set {i}...")
        # Plot the data on main plot
        if i==0:
            ax.scatter(x_arr,y_arr_1_lst[i],color="orange",edgecolors='none',label=tag1,zorder=100)
            ax.scatter(x_arr,y_arr_2_lst[i],color="blue",edgecolors='none',label=tag2,zorder=100)
        else:
            ax.scatter(x_arr,y_arr_1_lst[i],color="orange",edgecolors='none',zorder=100)
            ax.scatter(x_arr,y_arr_2_lst[i],color="blue",edgecolors='none',zorder=100)

        # Plot events/s in kHz
        rax.scatter(x_arr,nevents/(1000*np.array(y_arr_1_lst[i])),color="orange",edgecolors='none',zorder=100)
        rax.scatter(x_arr,nevents/(1000*np.array(y_arr_2_lst[i])),color="blue",edgecolors='none',zorder=100)

        # Plot the ratio on the ratio plot
        r_arr = np.array(y_arr_1_lst[i])/np.array(y_arr_2_lst[i])
        rrax.scatter(x_arr,r_arr,color="orange",edgecolors='none',zorder=100)

    # Set log scale
    if log:
        ax.set_yscale('log')
        rax.set_yscale('log')
        save_name = save_name+"_log"

    # Set titles and such
    ax.set_ylabel(yaxis_name)
    rax.set_ylabel(f"Events/s [kHz]")
    ax.legend(fontsize="12",framealpha=1)
    ax.set_title(save_name)
    ax.grid(zorder=-99)
    rax.grid(zorder=-99)
    rrax.grid(zorder=-99)
    rrax.axhline(1.0,linestyle="-",color="k",linewidth=1)
    #rrax.set_ylim(0.0,2.0)
    rrax.set_ylabel(f"{tag1}/{tag2}")
    rrax.set_xlabel(xaxis_name)

    ax.set_ylim(0.1,10e4)
    rax.set_ylim(1,10e5)
    rrax.set_ylim(-100,1050)

    #rrax.yaxis.set_major_locator(5)
    #plt.yticks(5)
    plt.locator_params(axis="y", nbins=5) 


    plt.savefig(save_name+".png",format="png")
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
        2 : "fill",
        3 : "total",
    }

    # Queries (for x axis)
    x = [1,2,3,4,5] #,6,7,8]

    # Timing numbers
    timing_dict = {
        "100k" : {
            "nevents" : 100000,
            "y_gpu_arr" : np.array([
                [[3.3297441005706787, 0.0032701492309570312, 0.07311630249023438, 3.40613055229187], [0.027616500854492188, 0.0021648406982421875, 0.00394892692565918, 0.033730268478393555], [0.02391195297241211, 0.001107931137084961, 3.2167067527770996, 3.2417266368865967], [0.023516416549682617, 0.0008764266967773438, 0.00836944580078125, 0.03276228904724121], [0.018593311309814453, 0.0017905235290527344, 0.1057736873626709, 0.12615752220153809]],
            ]),
            "y_cpu_arr" : np.array([
                [[0.0048258304595947266, 0.005120038986206055, 0.00060272216796875, 0.010548591613769531], [0.020740509033203125, 0.7300634384155273, 0.0010578632354736328, 0.7518618106842041], [0.03613471984863281, 1.4210612773895264, 0.005944490432739258, 1.4631404876708984], [0.019855260848999023, 0.7361855506896973, 0.0033833980560302734, 0.7594242095947266], [0.07020807266235352, 2.184292793273926, 0.022810697555541992, 2.2773115634918213]],
            ]),
        },
        "1M" : {
            "nevents" : 1e6,
            "y_gpu_arr" : np.array([
            ]),
            "y_cpu_arr" : np.array([
            ]),
        },
        "53M" : {
            "nevents" : 53446198,
            "y_gpu_arr" : np.array([
            ]),
            "y_cpu_arr" : np.array([
            ]),
        },
    }
    cat_to_plot = "100k"
    for dt_category_idx,dt_category_label in time_lables_dict.items():
        nevts = timing_dict[cat_to_plot]["nevents"]
        y_gpu = timing_dict[cat_to_plot]["y_gpu_arr"][:,:,dt_category_idx]
        y_cpu = timing_dict[cat_to_plot]["y_cpu_arr"][:,:,dt_category_idx]
        make_scatter_plot(x,y_cpu,y_gpu,xaxis_name=f"Benchmark Queries ({dt_category_label})",yaxis_name="Runtime [s]", tag1="CPU", tag2="GPU",save_name=f"adl_benchmarks_nEvents{cat_to_plot}_{dt_category_label}",log=False,nevents=nevts)


main()




