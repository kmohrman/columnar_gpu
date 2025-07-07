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
            ax1.scatter(xarr_1,yarr_1_lst[i],color="orange",edgecolors='none',label=tag1,zorder=100)
            ax1.scatter(xarr_2,yarr_2_lst[i],color="blue",edgecolors='none',label=tag2,zorder=100)
        else:
            ax1.scatter(xarr_1,yarr_1_lst[i],color="orange",edgecolors='none',zorder=100)
            ax1.scatter(xarr_2,yarr_2_lst[i],color="blue",edgecolors='none',zorder=100)

        # Plot events/s in kHz
        ax2.scatter(xarr_1,nevents/(1000*np.array(yarr_1_lst[i])),color="orange",edgecolors='none',zorder=100)
        ax2.scatter(xarr_2,nevents/(1000*np.array(yarr_2_lst[i])),color="blue",edgecolors='none',zorder=100)

        # Plot the ratio on the ratio plot
        min_len_x = min(len(xarr_1),len(xarr_2))
        r_arr = np.array(yarr_1_lst[i][:min_len_x])/np.array(yarr_2_lst[i][:min_len_x])
        ax3.scatter(xarr_1[:min_len_x],r_arr,color="orange",edgecolors='none',zorder=100)

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

    # Timing numbers
    timing_dict = {

        # Run 100k, about 3m
        "100k" : {
            "nevents" : 100000,
            "y_gpu_arr" : np.array([
                # Not first run
                [[5.945056676864624, 0.004033088684082031, 0.1805863380432129, 6.129676103591919], [0.047772884368896484, 0.0027849674224853516, 0.0065402984619140625, 0.0570981502532959], [0.032845258712768555, 0.001909017562866211, 10.34265422821045, 10.377408504486084], [0.032349348068237305, 0.0013751983642578125, 0.015073537826538086, 0.0487980842590332], [0.02304673194885254, 0.00416874885559082, 0.2514491081237793, 0.27866458892822266]],
            ]),
            "y_cpu_arr" : np.array([
                # Not first run
                [[0.015338420867919922, 0.037996768951416016, 0.0020720958709716797, 0.05540728569030762], [0.10113024711608887, 4.460193634033203, 0.0042035579681396484, 4.565527439117432], [0.1836705207824707, 8.920819997787476, 0.01326608657836914, 9.117756605148315], [0.10673952102661133, 4.499321937561035, 0.006756782531738281, 4.612818241119385], [0.4088881015777588, 13.262264728546143, 0.07317900657653809, 13.74433183670044], [0.4390749931335449, 22.393534421920776, 1.777578353881836, 24.610187768936157], [1.1567962169647217, 35.6050009727478, 0.21976232528686523, 36.98155951499939], [0.7942583560943604, 17.86157202720642, 0.23632144927978516, 18.892151832580566]],
            ]),
        },

        "1M" : {
            "nevents" : 1e6,
            "y_gpu_arr" : np.array([
            ]),
            "y_cpu_arr" : np.array([
            ]),
        },

        "10M" : {
            "nevents" : 1e6,
            "y_gpu_arr" : np.array([
            ]),
            "y_cpu_arr" : np.array([
            ]),
        },

        # Never run to full completion on CPU, many hours
        "53M" : {
            "nevents" : 53446198,
            "y_gpu_arr" : np.array([
                ## Not sure if first run or not
                ##[[5.880941867828369, 0.0177614688873291, 10.41261911392212, 16.311322450637817],[2.496755599975586, 0.005660295486450195, 0.17136716842651367, 2.67378306388855],[3.2397615909576416, 0.0070765018463134766, 7.817948341369629, 11.064786434173584],[0.4555375576019287, 0.0029714107513427734, 0.09490132331848145, 0.5534102916717529],[14.84013319015503, 0.0187835693359375, 2.7588915824890137, 17.61780834197998]]
            ]),
            "y_cpu_arr" : np.array([
                ## Not sure if first run or not
                ## Note: Needed 512000 mem for q6, q8 did not finish (and a retry of q7 ran for way longer than this number)
                ##[[0.4074842929840088, 2.301905393600464, 0.2707850933074951, 2.9801747798919678],[8.869683980941772, 461.466509103775, 0.40516066551208496, 470.7413537502289],[18.56472420692444, 912.3368966579437, 3.023542642593384, 933.9251635074615],[9.68858814239502, 451.86974906921387, 0.9235742092132568, 462.48191142082214],[36.50802683830261, 1172.8312983512878, 6.587008476257324, 1215.9263336658478],[57.896148920059204, 2327.27223277092, 26049.023535728455, 28434.191917419434],[129.4102804660797, 3296.884009361267, 754.1636970043182, 4180.457986831665], []]
            ]),
        },
    }

    # Make the plots
    cat_to_plot = "100k"
    for dt_category_idx,dt_category_label in time_lables_dict.items():
        nevts = timing_dict[cat_to_plot]["nevents"]
        y_gpu = timing_dict[cat_to_plot]["y_gpu_arr"][:,:,dt_category_idx]
        y_cpu = timing_dict[cat_to_plot]["y_cpu_arr"][:,:,dt_category_idx]
        make_scatter_plot(x_cpu,x_gpu,y_cpu,y_gpu,xaxis_name=f"Benchmark Queries ({dt_category_label})",yaxis_name="Runtime [s]", tag1="CPU", tag2="GPU",save_name=f"adl_benchmarks_nEvents{cat_to_plot}_{dt_category_label}",log=False,nevents=nevts)


main()




