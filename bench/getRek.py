import numpy as np
from bench.helpers import top_n_idx_sparse
from scipy.sparse import csr_matrix, coo_matrix

def getRekd_v3Optimized(model, res, predictions, test,test_rows,train,start_choices, \
               inv_port_map, scanned_num = [], RETRAIN_GAP=1, n=0,START=0,LIM=1000,\
            HIST_BINS=100,NUM_THREADS=72,pred_past = None):
    
    testcoo = test.tocoo()
    traincoo = train.tocoo()
    
    correct = 0
    
    # need this to make sure old predictions dont show up again
    # even when predictions have been updated
 
    if n > 0:
        
        pp = pred_past.multiply(10)
        
        #first subtract out previous pred_coo
        predictions = predictions + pp

    
    #get top per row
    chosen_indexes = np.concatenate(top_n_idx_sparse(predictions.tocsr().multiply(-1), RETRAIN_GAP))
        

    chosen_test_rows = np.repeat(test_rows,RETRAIN_GAP)
    
    scanned_num.append(len(chosen_indexes))
    #make matrix of predictions
    pred_all = coo_matrix(([1]*len(chosen_indexes),(chosen_test_rows,chosen_indexes)),shape=test.shape)
        
 

    # filter for only the correctly predicted this time
    pred_correct_coo = testcoo.multiply(pred_all)
    
    """ 
    if UNIQUE_IPS:
        pred_correct_ips = list(set(coo_matrix(pred_correct_coo).row))
        test_rows_l = list(test_rows)
        #find the corresponding prediction rows...
        pred_cor_rows = [test_rows_l.index(i) for i in pred_correct_ips]
        #"zero them out"
        predictions[:,1][pred_cor_rows,:] = 10
    """
        

    #find how many were correct
    correct = pred_correct_coo.count_nonzero()
        
    #log chosen port
    for i in chosen_indexes:
        chosen_port = inv_port_map[i]    
        if chosen_port not in res:
            res[chosen_port] = 0

        res[chosen_port] +=1 

    print("correct:",correct)
    print("Num Scanned: ",len(chosen_indexes) )
    hitrate = correct/len(chosen_indexes)
    #print("accuracy:", accuracy)
    print("Number unique ports Rec'd: ", len(res))
    #print(res)

    newtrain = pred_correct_coo + traincoo #+ pred_false_coo
    minitrain = pred_correct_coo

    
    if pred_past is not None:
        pred_all = pred_all + pred_past
    
    return pred_all,  pred_correct_coo, newtrain,minitrain,hitrate,correct,\
        res,start_choices, scanned_num,predictions
