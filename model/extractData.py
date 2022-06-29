import pandas as pd
from lightfm.data import Dataset
import numpy as np

#internal funcs
from model.coldSplit import cold_train_test_split

def extractData(f,user="ip", item="p", userFeat = None, itemFeat = None, train_percentage=0.8,\
                calcWeights = True,weightCap = 2000): 
    df = pd.read_csv(f)
    df = df.fillna(0)

    #prune the items, as they are expensive
    #if withItemFeats:
    #    df["minidata"] = df["minidata"].astype('category')
    #    df[["minidata"]] = df[["minidata"]].apply(lambda x: x.cat.codes)

    #    df['minidata'] = np.where(~df['minidata'].duplicated(keep=False), 0, df['minidata'])

    #introduce dataset
    dataset = Dataset()
    if itemFeat is not None and userFeat is not None:
        dataset.fit(list(df[user]),list(df[item]),\
        user_features=list(df[userFeat]),
        item_features = list(df[itemFeat]))
    elif userFeat is not None:
        dataset.fit(df[user],df[item],\
                user_features=df[userFeat])
    elif itemFeat is not None:
        dataset.fit(df[user],df[item],\
                item_features=df[itemFeat])
    else:
        dataset.fit(df[user],df[item])
    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))


    
    #get inverse propensity weights
    if calcWeights and item == "p": 
        normed_counts = (1/(df[item].value_counts(normalize=True)*100))
        df_weights = pd.DataFrame({item:normed_counts.index, 'w':normed_counts.values})
        df= pd.merge(df,df_weights,how='left',on=item)
        df['w'] = df['w'].apply(lambda x: weightCap if x > weightCap else x)
    
        #build interactions
        (interactions, weights) = \
        dataset.build_interactions(list(zip(df[user], df[item],df['w'])))

    else:
        (interactions, weights) = \
        dataset.build_interactions(list(zip(df[user], df[item])))

    print(repr(interactions))

    user_features = None
    item_features = None

    #build featureset
    if userFeat is not None:
        newFeats = list(zip(df[user], \
                        list(zip(list(df[userFeat])))))
        user_features = dataset.build_user_features(newFeats) 
        #user_features = normalize(user_features, norm='l2', axis=1)
    
    if itemFeat is not None:
        newFeats = list(zip(df[item], \
                        list(zip(list(df[itemFeat])))))
        item_features = dataset.build_user_features(newFeats) 
        #item_features = normalize(item_features, norm='l1', axis=1)
    
    #create train test
    train,test = cold_train_test_split(interactions,num_users,train_percentage=train_percentage)

    train.sort_indices()
    test.sort_indices()
    trainw = None
    testw = None
    if calcWeights:
        #handle weight stuff
        trainw = train.multiply(weights)
        testw = test.multiply(weights)
        trainw.sort_indices()
        testw.sort_indices()
        trainw = trainw.tocoo()
        testw = testw.tocoo()
    
    #zero out rows for training
    userf_train = None
    itemf_train = None
    if userFeat is not None:
        userf_train = zeroOutFeatures(train.tocoo(), num_users, user_features)    
    if itemFeat is not None:
        itemf_train = zeroOutFeatures(train.tocoo(), num_items, item_features)    

    #good to have mappings
    dmap= dataset.mapping()
    inv_ip_map = {v: k for k, v in dmap[0].items()}
    inv_port_map = {v: k for k, v in dmap[2].items()}
    
    return df, dataset, num_users, num_items, interactions, weights,user_features, userf_train, item_features, itemf_train, train,test,trainw,testw, dmap, inv_ip_map, inv_port_map
    
    
def zeroOutFeatures(train, num_users, user_features):

    train_rows = np.array(list(set(train.row)))
    userf_train_mask = np.array([True]*num_users)
    userf_train_mask[train_rows] = False
    nnz_per_row = np.diff(user_features.indptr)
    userf_train = user_features.copy()
    userf_train.data[np.repeat(userf_train_mask, nnz_per_row)] = 0
    userf_train.eliminate_zeros()
    return userf_train
