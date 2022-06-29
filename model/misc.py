def constructDescript(prefix,LR=0.01,N_COMP=4000,EPOCHS=10,MAX_SAMPLE=175,BIASED=False,\
                 RETRAIN=False,RETRAIN_WEIGHTS=False,WITH_ITEM_FEATS=False,\
                      CALC_WEIGHTS=False):
    
    return prefix + str(LR)+"LR_"+str(N_COMP)+"NCOMP_"+str(EPOCHS)+"EPOCHS_" + str(MAX_SAMPLE)+"MAXSAMPLE_"\
            +str(BIASED)+"BIASED_"\
            + str(RETRAIN)+"RETRAIN_"+str(RETRAIN_WEIGHTS)+"RETRAINWEIGHTS_"\
            + str(WITH_ITEM_FEATS)+"WITHITEMFEATS_" + str(CALC_WEIGHTS)+"CALCWEIGHTS"
