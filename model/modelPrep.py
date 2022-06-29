import tensorflow as tf

from model.getMetaData import groundTruthServices, getNewPredictions

def prepTheModel(model,test,train,LIM,START,HIST_BINS, user_features,item_features,\
                 BIASED = True,WITH_ITEM_FEATS=False,NUM_PREDS=30,EF_CONSTRUCTION=200, M=20,\
                 NUM_THREADS=72):

    
    num_test_ips, num_services,num_ips_per_port, test_rows, testcoo, LIM = \
    groundTruthServices(test,LIM,START)
    testPorts = list(set(testcoo.col))
    
    print("...Grabbed Representations")
    predictions,  top_items = \
    getNewPredictions(model,user_features,test,test_rows,item_features,BIASED,\
                      WITH_ITEM_FEATS,NUM_PREDS,EF_CONSTRUCTION, M,\
                      NUM_THREADS)
    print("...Calculated Predictions")
    predictions = top_items
    return num_test_ips, num_services,num_ips_per_port, test_rows, testcoo,\
        predictions, testPorts, LIM


