import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, hstack,dok_matrix, lil_matrix, vstack
import nmslib
import gc

def groundTruthServices(test,LIM,START):
    testcoo = test.tocoo()
    test_rows = np.array(list(set(testcoo.row))[START:LIM])
    test_rows.sort()
    
    #adjust lim
    if LIM > len(test_rows) or LIM == 0:
        LIM = len(test_rows)
    
    num_test_ips = len(test_rows)
    num_services = sum(test[test_rows].tocoo().data)
    num_ips_per_port=  coo_matrix(test[test_rows].sum(axis=0))
    return num_test_ips, num_services,num_ips_per_port, test_rows, testcoo, LIM

def getNewPredictions(model,user_features,test, test_rows,item_features,\
                      BIASED,WITH_ITEM_FEATS,NUM_PREDS, \
                      EF_CONSTRUCTION,M, NUM_THREADS):
   
    if WITH_ITEM_FEATS:
        item_biases, item_embeddings = model.get_item_representations(item_features)
    else:
        item_biases, item_embeddings = model.get_item_representations()
        
    print("Got Item Representations")
    if user_features is not None:
        user_biases, user_embeddings = model.get_user_representations(user_features[test_rows])
    else:
        user_biases, user_embeddings = model.get_user_representations()
    print("Got User Representations")
    predictions = None
    
    norms = np.linalg.norm(item_embeddings, axis=1)
    max_norm = norms.max()
    extra_dimension = np.sqrt(max_norm ** 2 - norms ** 2)
    norm_data = np.append(item_embeddings, extra_dimension.reshape(norms.shape[0], 1), axis=1)

    #first an nmslib
    nms_member_idx = nmslib.init(method='hnsw', space='cosinesimil')
     
    nms_member_idx.addDataPointBatch(norm_data)
        
    #indexTimeParams = {'M': M, 'indexThreadQty': NUM_THREADS,\
    #                  'efConstruction': efC, 'post' : 0}
    #https://github.com/nmslib/nmslib/blob/master/manual/methods.md
    indexTimeParams = {'efConstruction':  EF_CONSTRUCTION, 'M':M }
        
        
    nms_member_idx.createIndex(indexTimeParams,print_progress=True)
    print("Made Index")
    
    #allUsers = np.c_[user_embeddings,[0]*len(test_rows)]
    allUsers = user_embeddings
    top_items = np.array(nms_member_idx.knnQueryBatch(allUsers, k=NUM_PREDS, num_threads=NUM_THREADS))
    print("Got Top Items")
    del  item_embeddings
    del user_embeddings
    gc.collect()
        
        
    num_r = [len(x) for x in top_items[:,1]]
    top_items_coo = coo_matrix((np.concatenate(top_items[:,1]),\
                                  (np.repeat(test_rows, num_r), np.concatenate(top_items[:,0]))),\
                                   shape=test.shape)
    if BIASED:
        norm = np.linalg.norm(item_biases)
        item_biases_norm = (item_biases/norm) #*4
        item_biases_coo = coo_matrix((item_biases_norm,\
                                     ([0]*len(item_biases_norm),np.arange(len(item_biases_norm)))))
            
        top_items_identity = coo_matrix(([1] * len(np.concatenate(top_items[:,1])),\
                                          (np.repeat(test_rows, num_r), np.concatenate(top_items[:,0]))),\
                                           shape=test.shape)
        item_biases_identity = top_items_identity.multiply(item_biases_coo)
        top_items_biased = top_items_coo - item_biases_identity
        top_items = top_items_biased

    else:
        top_items = top_items_coo
    print("Got Predictions/Top Items")
    
 
    return predictions, top_items
    
