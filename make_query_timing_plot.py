import numpy as np
import matplotlib.pyplot as plt

# Scatter plot to compare Q times
def make_scatter_plot(x_arr,y_arr_1_lst,y_arr_2_lst,log=False,xaxis_name="x",yaxis_name="y",tag1="set1",tag2="set2",save_name="test"):

    #fig, axs = plt.subplots(nrows=1, ncols=1)

    # Create the figure
    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7,7),
        gridspec_kw={"height_ratios": (3, 1)},
        sharex=True
    )
    fig.subplots_adjust(hspace=.07)

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

        # Plot the ratio on the ratio plot
        r_arr = np.array(y_arr_1_lst[i])/np.array(y_arr_2_lst[i])
        rax.scatter(x_arr,r_arr,color="orange",edgecolors='none',zorder=100)

    # Set log scale
    if log:
        ax.set_yscale('log')
        save_name = save_name+"_log"

    # Set titles and such
    ax.set_ylabel(yaxis_name)
    ax.legend(fontsize="12",framealpha=1)
    ax.set_title(save_name)
    ax.grid(zorder=-99)
    rax.grid(zorder=-99)
    rax.axhline(1.0,linestyle="-",color="k",linewidth=1)
    #rax.set_ylim(0.0,2.0)
    rax.set_ylabel(f"{tag1}/{tag2}")
    rax.set_xlabel(xaxis_name)

    #rax.yaxis.set_major_locator(5)
    #plt.yticks(5)
    plt.locator_params(axis="y", nbins=5) 


    plt.savefig(save_name+".png",format="png")
    #plt.show()
    return plt



def main():

    x = [1,2,3,4,5]

    # Full run
    y_gpu = [[31.389608144760132, 3.022132158279419, 17.002512454986572, 0.9368855953216553, 2991.8851997852325]]
    y_cpu = [[16.73035740852356, 915.8247015476227, 1816.7914366722107, 886.7176096439362, 2366.1546704769135]]
    make_scatter_plot(x,y_cpu,y_gpu,xaxis_name="Benchmark Queries",yaxis_name="Runtime [s]", tag1="CPU", tag2="GPU",save_name="coffea_adl_benchmarks_nEvents53446198",log=False)
    make_scatter_plot(x,y_cpu,y_gpu,xaxis_name="Benchmark Queries",yaxis_name="Runtime [s]", tag1="CPU", tag2="GPU",save_name="coffea_adl_benchmarks_nEvents53446198",log=True)

    # 100k events
    y_gpu = [
        [24.451767206192017, 0.5169780254364014, 10.540595531463623, 0.10236239433288574, 11.251887321472168],
        [22.06326937675476, 0.39934349060058594, 10.627521514892578, 0.10174894332885742, 5.711676120758057],
        [23.687761545181274, 4.677731513977051, 9.980894327163696, 0.15544724464416504, 6.260521173477173],
    ]
    y_cpu = [
        [0.5371420383453369, 1.417785882949829, 2.8521766662597656, 1.4171738624572754, 4.656379222869873],
        [0.4499659538269043, 1.4361202716827393, 2.7690248489379883, 1.3557918071746826, 15.998578786849976],
        [0.611318826675415, 1.4838826656341553, 2.8346333503723145, 1.4293062686920166, 4.631955862045288],
    ]
    make_scatter_plot(x,y_cpu,y_gpu,xaxis_name="Benchmark Queries",yaxis_name="Runtime [s]", tag1="CPU", tag2="GPU",save_name="coffea_adl_benchmarks_nEvents100k",log=False)
    make_scatter_plot(x,y_cpu,y_gpu,xaxis_name="Benchmark Queries",yaxis_name="Runtime [s]", tag1="CPU", tag2="GPU",save_name="coffea_adl_benchmarks_nEvents100k",log=True)

    # 1M events
    y_gpu = [
        [24.593621730804443, 0.4054069519042969, 11.33007550239563, 0.32595205307006836, 56.92887783050537],
        [23.62338161468506, 0.6453971862792969, 10.756298303604126, 0.21704339981079102, 57.37603712081909],
        [19.454494953155518, 0.6683943271636963, 9.630228042602539, 0.25890207290649414, 59.6738817691803],
    ]

    y_cpu = [
        [1.2984192371368408, 17.31274437904358, 34.383469104766846, 17.065198183059692, 43.43406653404236],
        [1.4182829856872559, 16.802221298217773, 34.37035870552063, 17.264073371887207, 43.65031099319458],
        [1.095402717590332, 17.175547122955322, 34.54497003555298, 17.08597159385681, 43.89459705352783],
    ]
    make_scatter_plot(x,y_cpu,y_gpu,xaxis_name="Benchmark Queries",yaxis_name="Runtime [s]", tag1="CPU", tag2="GPU",save_name="coffea_adl_benchmarks_nEvents1M",log=False)
    make_scatter_plot(x,y_cpu,y_gpu,xaxis_name="Benchmark Queries",yaxis_name="Runtime [s]", tag1="CPU", tag2="GPU",save_name="coffea_adl_benchmarks_nEvents1M",log=True)







main()




