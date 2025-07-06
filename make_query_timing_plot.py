import numpy as np
import matplotlib.pyplot as plt

# Scatter plot to compare Q times
def make_scatter_plot(xarr,yarr_1_lst,yarr_2_lst,log=False,xaxis_name="x",yaxis_name="y",tag1="set1",tag2="set2",save_name="test",nevents=None,sub_len_x=None):

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
    for i in range(len(yarr_1_lst)):
        print(f"Plotting for set {i}...")
        # Plot the data on main plot
        if i==0:
            ax1.scatter(xarr,yarr_1_lst[i],color="orange",edgecolors='none',label=tag1,zorder=100)
            ax1.scatter(xarr,yarr_2_lst[i],color="blue",edgecolors='none',label=tag2,zorder=100)
        else:
            ax1.scatter(xarr,yarr_1_lst[i],color="orange",edgecolors='none',zorder=100)
            ax1.scatter(xarr,yarr_2_lst[i],color="blue",edgecolors='none',zorder=100)

        # Plot events/s in kHz
        ax2.scatter(xarr,nevents/(1000*np.array(yarr_1_lst[i])),color="orange",edgecolors='none',zorder=100)
        ax2.scatter(xarr,nevents/(1000*np.array(yarr_2_lst[i])),color="blue",edgecolors='none',zorder=100)

        # Plot the ratio on the ratio plot
        r_arr = np.array(yarr_1_lst[i][:sub_len_x])/np.array(yarr_2_lst[i][:sub_len_x])
        ax3.scatter(xarr[:sub_len_x],r_arr,color="orange",edgecolors='none',zorder=100)

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
    x_gpu = [1,2,3,4,5]
    x_cpu = [1,2,3,4,5,6,7,8]
    x     = [1,2,3,4,5,6,7,8]

    # Timing numbers
    timing_dict = {
        "100k" : {
            "nevents" : 100000,
            "y_gpu_arr" : np.array([
                [[1.6387996673583984, 0.003034830093383789, 0.0665884017944336, 1.7084228992462158], [0.026049137115478516, 0.0014963150024414062, 0.0031707286834716797, 0.0307161808013916], [0.022825002670288086, 0.0007688999176025391, 3.0250017642974854, 3.048595666885376], [0.02307581901550293, 0.0006453990936279297, 0.0075185298919677734, 0.031239748001098633], [0.013154983520507812, 0.0015950202941894531, 0.09278297424316406, 0.10753297805786133], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ]),
            "y_cpu_arr" : np.array([
                [[0.004276752471923828, 0.004875898361206055, 0.0005404949188232422, 0.009693145751953125], [0.01968693733215332, 0.7190582752227783, 0.0009844303131103516, 0.739729642868042], [0.03493309020996094, 1.496469259262085, 0.00546574592590332, 1.5368680953979492], [0.019341707229614258, 0.741987943649292, 0.0030748844146728516, 0.7644045352935791], [0.06572484970092773, 2.2517147064208984, 0.017827987670898438, 2.3352675437927246], [0.07760357856750488, 3.7023203372955322, 0.4688701629638672, 4.248794078826904], [0.19384026527404785, 6.081218481063843, 0.061873435974121094, 6.336932182312012], [0.13547539710998535, 3.0686004161834717, 0.07102584838867188, 3.275101661682129]],
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
        make_scatter_plot(x,y_cpu,y_gpu,xaxis_name=f"Benchmark Queries ({dt_category_label})",yaxis_name="Runtime [s]", tag1="CPU", tag2="GPU",save_name=f"adl_benchmarks_nEvents{cat_to_plot}_{dt_category_label}",log=False,nevents=nevts,sub_len_x=5)


main()




