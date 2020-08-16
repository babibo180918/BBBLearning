import numpy as np

def update(w, b, optimizer = "GradientDecent", grads, vgrads, sgrads, learning_rate, epsilon, v_corrected, s_corrected):
    
    if optimizer == "GradientDecent":
        w = w - learning_rate*grads["dw"]
        b = b - learning_rate*grads["db"]
    else if optimizer == "momentum":
        w = w - learning_rate*vgrads["vdw"]
        b = b - learning_rate*vgrads["vdb"]
    else if optimizer == "LMSprop":
        w = w - learning_rate*grads["dw"]/np.sqrt(sgrads["sdw"])
        b = b - learning_rate*grads["db"]/np.sqrt(sgrads["sdb"])
    else if optimizer == "Adam":
        vdw_ = vgrads["vdw"]/v_corrected
        vdb_ = vgrads["vdb"]/v_corrected
        sdw_ = sgrads["sdw"]/s_corrected
        sdb_ = sgrads["sdb"]/s_corrected
        w = w - learning_rate*vdw_/(np.sqrt(sdw_) + epsilon)
        b = b - learning_rate*vdb_/(np.sqrt(sdb_) + epsilon)
        
    return w, b
