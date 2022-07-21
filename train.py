import pandas as pd

from utils import preprocess_data, encode_items

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, fpgrowth

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--data_path', required=True, help='Input data path')
arg_parser.add_argument('--sample_size', required=False, default=0.02, help='Sample data size (to avoid out of memory issues)')
arg_parser.add_argument('--min_support', required=False, default=0.001, help='Min support threshold')
arg_parser.add_argument('--min_threshold', required=False, default=0.001, help='Min threshold for association rules')
arg_parser.add_argument('--metric_rule', required=False, default='support', help='Association rules when creating matrix')
arg_parser.add_argument('--save_path', required=True, help='Model output save path')
args = vars(arg_parser.parse_args())

RANDOM_SEED=42

# load & sample data
print('[INFO] Loading data...')
data = pd.read_csv(args['data_path'])
sample = sample = data.sample(frac=.02, random_state=RANDOM_SEED)
preprocessed_data = preprocess_data(sample)

# fit model
print('[INFO] Training model...')
model = encode_items(preprocessed_data)
freq_items = fpgrowth(model, args['min_support'], use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=args['min_threshold'])

rules.to_csv(args['path'])
print('[INFO] Model has finished training')