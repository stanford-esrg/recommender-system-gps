import numpy as np
from scipy.sparse import dok_matrix



def cold_train_test_split(interactions,num_users,train_percentage=0.5,random_state=7):

    if train_percentage == 1.0:
        return interactions, None    

    #remove weird duplicates that are introduced by lightfm
    interactions_dok=dok_matrix((interactions.shape),dtype=interactions.dtype)
    interactions_dok._update(zip(zip(interactions.row,interactions.col),interactions.data))

    train_a = interactions_dok.tocsr()
    test_a = interactions_dok.tocsr()
    
    train_mask = np.array([True]*num_users)
    np.random.seed(random_state)
    indices = np.random.choice(np.arange(num_users), replace=False,
                           size=int(num_users * train_percentage))
    
    train_mask[indices] = False
    test_mask = ~np.array(train_mask)
    
    nnz_per_row = np.diff(train_a.indptr)
    
    #zero out rows
    
    test_a.data[np.repeat(test_mask, nnz_per_row)] = 0
    train_a.data[np.repeat(train_mask, nnz_per_row)] = 0
    train_a.eliminate_zeros()
    test_a.eliminate_zeros()
    

    #makes its own shape--need to have it retain old shape....
    return train_a,test_a
