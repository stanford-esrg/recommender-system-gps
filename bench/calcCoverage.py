import sys
sys.path.append('../')

from model.getMetaData import getNewPredictions
from bench.getRek import getRekd_v3Optimized
from bench.helpers import evalSuccess

def calcCoverage_optimized(model,predictions,num_services,num_test_ips,\
                 test, testw,test_rows,train,user_features, item_features, inv_port_map, \
                 dmap, num_ips_per_port,NUM_THREADS,NUM_PREDS,RETRAIN,
                 RETRAIN_GAP, RETRAIN_ALL,RETRAIN_EPOCHS, BIASED,LIM,HIST_BINS,\
                 CYCLES,RETRAIN_WEIGHTS,CALC_WEIGHTS,WITH_ITEM_FEATS,EF_CONSTRUCTION,M):

    
    bench2_results = {}
    portsScanned = {}
    normed_services = []
    frac_services = []
    frac_ips = []
    hitrate = []
    scanned_num =  []
    num_unique_rec = []
    pred_all_past = None
    start_choices = 1
    newtrain = train.tocoo()
    total_services_correct = 0
    
    i = 0                 
    while i < CYCLES: 
        print("cycle: ",i)

        pred_all_past,  pred_correct, newtrain,minitrain,cur_hitrate,correct, \
        portsScanned,start_choices,scanned_num,predictions = \
        getRekd_v3Optimized(model,portsScanned, predictions, test,test_rows,\
                   newtrain,start_choices,inv_port_map,scanned_num, RETRAIN_GAP=RETRAIN_GAP,n=i,LIM=LIM,\
                            HIST_BINS=HIST_BINS,pred_past=pred_all_past)

             

        print("At Bin: ", i)
        print("Hitrate: ", cur_hitrate)
        i = i + RETRAIN_GAP

        all_correctly_predicted, norm_services, frac_service, frac_ip, total_ips_correct \
        = evalSuccess(newtrain,test,dmap,num_ips_per_port, num_services,num_test_ips)


        normed_services.append(norm_services)
        frac_services.append(frac_service)
        frac_ips.append(frac_ip)
        hitrate.append(cur_hitrate)
        num_unique_rec.append(len(portsScanned))
        
        if RETRAIN:

            if RETRAIN_ALL:
                training = newtrain
            else:
                training = minitrain


            if RETRAIN_WEIGHTS:
                training.multiply(10)
                
            if CALC_WEIGHTS:
                trainingw = training.multiply(testw).tocoo()
            else:
                trainingw = training.tocoo()    

            if WITH_ITEM_FEATS:
                model = model.fit_partial(training, 
                    sample_weight = trainingw, #what to do about the weight???
                    user_features=user_features,
                    item_features=item_features,
                    epochs=RETRAIN_EPOCHS,
                    num_threads=NUM_THREADS, verbose=True)
            else:
                model = model.fit_partial(training,
                    sample_weight = trainingw,
                    user_features=user_features,
                    epochs=RETRAIN_EPOCHS,
                    num_threads=NUM_THREADS, verbose=True)       

            print("...Grabbed Representations")
            _,predictions = \
            getNewPredictions(model,user_features,test,test_rows,item_features,BIASED,\
                              WITH_ITEM_FEATS,NUM_PREDS,EF_CONSTRUCTION,M, NUM_THREADS)
            print("...Calculated Predictions")

    bench2_results["num_unique_rec"] = num_unique_rec
    bench2_results["frac_services"] = frac_services
    bench2_results["frac_ips"] = frac_ips
    bench2_results["hitrate"] = hitrate
    bench2_results["normed_services"] = normed_services
    bench2_results["scanned_num"] = scanned_num
    
    return bench2_results,predictions, all_correctly_predicted    
