import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

def preprocess_data(df):
    """
    Preprocess data with several steps:
        1. remove unwanted columns
        2. convert id columns to string
        3. fill empty values
        4. split category_code
        5. create a list of item combinations

    Arguments:
        df: pd.DataFrame
            input data

    Returns:
        items: list
            list of item combinations
    """
    
    to_remove = ['event_time', 'event_type', 'user_session', 'price', 'brand']
    df = df.drop(columns=to_remove, axis=1)

    df['product_id'] = df['product_id'].astype(str)
    df['user_id'] = df['user_id'].astype(str)
    df['category_id'] = df['category_id'].astype(str)

    df['category_code'] = df['category_code'].fillna('none')
    df['category_code'] = df['category_code'].apply(lambda x: x.split('.'))

    items = []
    for _, row in df.groupby(['user_id']):
        items.append(row['product_id'].to_list())

    return items

def encode_items(preprocessed_data):
    """
    Fit a TransactionEncoder model

    Arguments:
        preprocessed_data: pd.DataFrame
            preprocessed input data

    Returns:
        df: pd.DataFrame
            fitted TransactionEncoder data
    """

    te = TransactionEncoder()
    te_fit = te.fit(preprocessed_data).transform(preprocessed_data)
    df = pd.DataFrame(te_fit, columns=te.columns_)

    return df