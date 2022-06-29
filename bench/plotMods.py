##PLOTTING MODULES##
import matplotlib.pyplot as plt
import pandas as pd

def make_cdf( data, label=""):

    sorted_data = np.sort(data)
    yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)

    plt.plot(sorted_data,yvals, label=label,alpha=0.8)
    return yvals

def plotDiscovery(DATAPATH,frac_services, frac_ips):

    df = pd.read_csv(DATAPATH+"lzr01_host_discovery.csv")
    plt.plot([0]+df["rolling_sum"],label="ips - mpp (lzr 65k)")

    df = pd.read_csv(DATAPATH+"lzr01_service_discovery.csv")
    plt.plot([0]+df["rolling_sum"],label="services - mpp (lzr 65k)")

    plt.plot([0]+frac_services,label="services - ML")
    plt.plot([0]+frac_ips,label="ips -ML")
    plt.ylabel("Fraction of IPs/Services Discovered")
    plt.xlabel("Number of Scans")
    plt.xlim((1,len(df["rolling_sum"])))
    plt.title("IPv4 Discovery-Dynamic Choice")
    plt.xscale('log')
    #plt.xlim((-1,20))
    plt.legend()
    plt.show()
    
def plotHitrate(hitrate,LIM):

    mpp_hitrate =  getMppHitRate(num_ips_per_port,LIM)
    #df = pd.read_csv(DATAPATH+"lzr001_hitrate.csv")
    plt.plot([0]+mpp_hitrate,label="hitrate- mpp, (lzr 65k)")
    plt.plot([0]+hitrate,label="hitrate- ML")
    plt.legend()
    plt.xlim((1,len(mpp_hitrate)))
    plt.xscale('log')
    plt.xlabel("Number of Scans")
    plt.ylabel("Hitrate %")
    plt.title("Bandwidth Saved")

    plt.show()

def plotNormServiceDiscovery(scanned_num,normed_services,LIM):
    plt.plot(np.cumsum([1]*int(max(np.cumsum(np.array(scanned_num)/LIM)))),label="mpp, (lzr 65k)")
    plt.plot(np.cumsum(np.array(scanned_num)/LIM),normed_services,label="ML")
    plt.legend()
    plt.xlabel("Number of Scans")
    plt.ylabel("# Normalized")
    plt.title("Normalized Services Discovered")
    plt.show()
    
def plotGradientNormServiceDiscovery(scanned_num,normed_services,LIM):

    plt.plot([1]*int(max(np.cumsum(np.array(scanned_num)/LIM))),label="mpp, (lzr 65k)")
    plt.plot(np.cumsum(np.array(scanned_num)/LIM),np.gradient(normed_services,np.cumsum(np.array(scanned_num)/LIM)),label="ML")
    plt.legend()
    plt.xlabel("Number of Scans")
    plt.ylabel("# Normalized Per Scan")
    plt.title("Derivative of Normalized Services Discovered")
    plt.show()
