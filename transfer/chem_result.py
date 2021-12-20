import pandas as pd
import numpy as np

result_log_path = 'transfer_exp/chem/chembl_filtered/fine_tune_result.txt'
result = pd.read_csv(result_log_path, sep=' ')
print(result)

datasets = ["bbbp", "tox21", "toxcast", "sider", "clintox", "muv", "hiv", "bace"]

for dataset in datasets:
    now_result = result.loc[result['dataset'] == dataset]
    mean = now_result['test_acc'].mean()
    std = now_result['test_acc'].std()
    print('{}\t{:.2f} ± {:.2f}'.format(dataset, mean*100, std*100) )
