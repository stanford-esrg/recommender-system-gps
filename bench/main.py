from bench.calcCoverage import calcCoverage_optimized
from bench.plotMods import plotDiscovery, plotHitrate, plotNormServiceDiscovery, plotGradientNormServiceDiscovery
import numpy as np

def bench2_Improvements( model,predictions,num_services,num_test_ips,\
                        test,testw,test_rows,train,user_features,\
                        item_features, inv_port_map, dmap, num_ips_per_port,\
                        NUM_THREADS, NUM_PREDS, RETRAIN,RETRAIN_GAP, \
                        RETRAIN_ALL,RETRAIN_EPOCHS, BIASED,LIM,HIST_BINS, \
                        CYCLES,RETRAIN_WEIGHTS, CALC_WEIGHTS, WITH_ITEM_FEATS,EF_CONSTRUCTION,\
                        M,model_description, DATAPATH):
        

    print("-------")
    print("...Beginning Bench2: Compare Different Metrics of Coverage and Hitrate")
    
    bench2_results,predictions,all_correctly_predicted = calcCoverage_optimized(model,\
                                  predictions,num_services,num_test_ips,\
                                  test,testw,test_rows,train,user_features,item_features,inv_port_map, 
                                  dmap, num_ips_per_port, NUM_THREADS, NUM_PREDS, 
                                  RETRAIN,RETRAIN_GAP, RETRAIN_ALL,\
                                  RETRAIN_EPOCHS ,BIASED,LIM,HIST_BINS, CYCLES,RETRAIN_WEIGHTS,\
                                 CALC_WEIGHTS,WITH_ITEM_FEATS,EF_CONSTRUCTION,M)
    print("...calculations complete")
    
    np.save(DATAPATH + model_description+ "_coverageB2.npy",bench2_results)
    np.save(DATAPATH + model_description+ "_correctpredictions.npy",all_correctly_predicted)
    
    plotDiscovery(DATAPATH,bench2_results["frac_services"], bench2_results["frac_ips"])

    plotHitrate(bench2_results["hitrate"],LIM)

    plotNormServiceDiscovery(bench2_results["scanned_num"],bench2_results["normed_services"],LIM)
    
    plotGradientNormServiceDiscovery(bench2_results["scanned_num"],bench2_results["normed_services"],LIM)
 
    print("...Finished Bench2")
    print("-------")
    return bench2_results,predictions,all_correctly_predicted
