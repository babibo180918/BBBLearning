import numpy as np

def update(w, b, grads, vgrads, sgrads, learning_rate, epsilon, v_corrected, s_corrected, optimizer = "GradientDecent"):
    
    if optimizer == "GradientDecent":
        w = w - learning_rate*grads["dw"]
        b = b - learning_rate*grads["db"]
    elif optimizer == "momentum":
        w = w - learning_rate*vgrads["vdw"]
        b = b - learning_rate*vgrads["vdb"]
    elif optimizer == "LMSprop":
        w = w - learning_rate*grads["dw"]/np.sqrt(sgrads["sdw"])
        b = b - learning_rate*grads["db"]/np.sqrt(sgrads["sdb"])
    elif optimizer == "Adam":
        vdw_ = vgrads["vdw"]/v_corrected
        vdb_ = vgrads["vdb"]/v_corrected
        sdw_ = sgrads["sdw"]/s_corrected
        sdb_ = sgrads["sdb"]/s_corrected
        w = w - learning_rate*vdw_/(np.sqrt(sdw_) + epsilon)
        b = b - learning_rate*vdb_/(np.sqrt(sdb_) + epsilon)
        
    return w, b
