import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def getMppHitRate(num_ips_per_port,LIM):
    temp_ipsAndPorts = num_ips_per_port.data
    temp_ipsAndPorts.sort()
    temp_ipsAndPorts = temp_ipsAndPorts[::-1]/LIM
    return temp_ipsAndPorts

def calcNormed(all_correctly_predicted,num_ips_per_port):
    num_correctly_pred_per_port = coo_matrix(all_correctly_predicted.sum(axis=0))

    #do some kind of sum of all of these fractions and divide by total number ports?
    norm_services = coo_matrix(num_correctly_pred_per_port/num_ips_per_port)
    norm_services.data = np.nan_to_num(norm_services.data, copy=False)
    norm_services = norm_services.sum()
    return norm_services

def evalSuccess(predictions,test,dmap,num_ips_per_port, num_services,num_test_ips):

    
    predictions_identity = coo_matrix(([1] * len(predictions.data),\
                                                  (predictions.tocoo().row, predictions.tocoo().col)),\
                                                   shape=test.shape)


    all_correctly_predicted = (test.multiply(predictions_identity)).tocoo()
    
    del predictions_identity
    corPred_identityish = coo_matrix(([1] * len(set(all_correctly_predicted.row)),\
                                                  (list(set(all_correctly_predicted.row)), list(set(all_correctly_predicted.row)))),\
                                                   shape=(test.shape[0],test.shape[0]))


    #extracting rows w/ at least 1 prediction, pretending we found all ports
    normalized_found = corPred_identityish * test
    del corPred_identityish

    total_services_correct = all_correctly_predicted.count_nonzero()
    total_ips_correct = len(set(all_correctly_predicted.row))
    print("total ips correct: ",total_ips_correct)    

    norm_services = calcNormed(all_correctly_predicted,num_ips_per_port)
    norm_IPs = calcNormed(normalized_found, num_ips_per_port)

    
    norm_services = calcNormed(all_correctly_predicted,num_ips_per_port)
    norm_IPs = calcNormed(normalized_found, num_ips_per_port)


    #Remove top 4 ports 80,443,7547,22
    popPorts = []
    for p in [80,443,7547,22]:
        popPorts.append(dmap[2][p])
    
    
    popCrew = all_correctly_predicted.tocsr()[:,popPorts]
    popCrew_services_correct = popCrew.count_nonzero()
    popCrew_ips_correct = len(set(popCrew.tocoo().row))


    num_test_ips_depop = len(set(test[:,popPorts].tocoo().row))
    num_test_services_depop = test[:,popPorts].count_nonzero()


    #if RETRAIN_WEIGHTS:
    #    norm_services = norm_services/10

    frac_services = total_services_correct/num_services
    frac_ips = total_ips_correct/num_test_ips

    frac_services_depop = (total_services_correct -popCrew_services_correct) / (num_services - num_test_services_depop)
    frac_ips_depop = (total_ips_correct -  popCrew_ips_correct)/(num_test_ips - num_test_ips_depop)


    print("Total Normalized Services: ", norm_services)
    print("Total Normalized IPs: ", norm_IPs)
    print("Total Fraction Services:", frac_services)
    print("Total Fraction IPs:", frac_ips )
    print("Total Fraction Depopularized Services:", frac_services_depop)
    print("Total Fraction Depopularized IPs:", frac_ips_depop )
    
    
    return all_correctly_predicted, norm_services, frac_services, frac_ips, total_ips_correct

def top_n_idx_sparse(matrix, n):
    '''Return index of top n values in each row of a sparse matrix'''
    top_n_idx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        top_n_idx.append(matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]])
    return top_n_idx


