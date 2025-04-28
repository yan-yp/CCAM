import numpy as np

dump_dir="dump/loftr_ot_outdoor/LoFTR_pred_eval.npy"

results = np.load(dump_dir)
print(results)