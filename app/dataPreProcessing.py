#pip install psycopg2-binary 

import os
import psycopg2
import pandas as pd
from collections import Counter

def dropNa (df):
    return df.dropna()

def dropId (df):
    columnNames = df.columns.tolist()
    listToDrop = []
    for i in range(df.shape[1]):
        if df[columnNames[i]].nunique() == df.shape[0]:
            listToDrop.append(columnNames[i])
    df = df.drop(columns=listToDrop)
    return df    

def binaryEncoding (df): 
    # binary encoding with "yes"/"no" strings
    mappings = {}
    columnNames = df.columns.tolist()
    for i in range(df.shape[1]): 
        if df[columnNames[i]].nunique() == 2 and df[columnNames[i]].dtype == "object":
            unique_vals = [str(x).lower() for x in df[columnNames[i]].unique()]
            if (("yes" in unique_vals and "no" in unique_vals) 
                or ("1" in unique_vals and "0" in unique_vals) 
                or ("true" in unique_vals and "false" in unique_vals)):
                mapping = {
                    'yes': 1, 'true': 1, '1': 1,
                    'no': 0, 'false': 0, '0': 0
                }
                df[columnNames[i]] = df[columnNames[i]].str.lower().map(mapping).astype(int)
            else: 
                print(f"Mapping '{unique_vals[0]}' as 1, and '{unique_vals[1]}' as 0")
                mapping = {
                    unique_vals[0]: 1, 
                    unique_vals[1]: 0}
                df[columnNames[i]] = df[columnNames[i]].str.lower().map(mapping).astype(int)
                mappings[columnNames[i]] = mapping
    return df, mappings


def oneHotEncoding (df): 
    columnNames = df.columns.tolist()
    columnsToEncode = []
    for i in range(df.shape[1]): 
        '''if (df[columnNames[i]].nunique() > 2 
            and df[columnNames[i]].nunique() <= 5 # target 5 categories only. 
            and df[columnNames[i]].dtype == "object"):
            columnsToEncode.append(columnNames[i])'''
        if (df[columnNames[i]].nunique() > 2
            and df[columnNames[i]].dtype == "object"):
            unique_vals = [str(x).lower() for x in df[columnNames[i]].unique()]
            val_counts = Counter(unique_vals)
            sorted_counts = dict(sorted(val_counts.items(), key=lambda item: item[1], reverse=True)) #declining order; highest to lowest
            total_count = sum(val_counts.values())
            # while loop to sum from the top till the sum is >90%, then remaining will be lumped as others.
            cumulative_sum = 0
            count = 0
            top_categories = []
            while cumulative_sum / total_count  <0.9 and count < len(sorted_counts):
                key = list(sorted_counts.keys())[count]
                cumulative_sum += sorted_counts[key]
                top_categories.append(key)
                count += 1
            
            # Map function to replace categories outside top_categories with 'others'
            def group_others(val):
                val = str(val).lower()
                if val in top_categories:
                    return val
                else:
                    return 'others'
                       
            df[columnNames[i]] = df[columnNames[i]].apply(group_others)
            columnsToEncode.append(columnNames[i]) 

    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=columnsToEncode, drop_first=False)
    # Convert all boolean columns to int (0/1)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df

def main():
    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"]
    )

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM churn_rate LIMIT 5;") #actual table
    rows = cursor.fetchall()

    print("First 5 rows from 'churn_rate':") #actual table
    for row in rows:
        print(row)
        print(type(row))
    
    cursor.execute("SELECT * FROM churn_rate;")
    allRows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description] #only available after cursor.execute
    df = pd.DataFrame(allRows, columns=colnames)

    print("First 5 rows from 'churn_rate' as DataFrame:")
    print(df.head())
    print(df.shape)

    dropNaDf = dropNa(df)
    print(dropNaDf.head())
    print(dropNaDf.shape)

    dropIdDf = dropId(dropNaDf)
    print(dropIdDf.head())
    print(dropIdDf.shape)

    binaryEncodedDf, mappings = binaryEncoding(dropIdDf)
    print(binaryEncodedDf.head())
    print(binaryEncodedDf.shape)
    print(mappings)

    oneHotEncodedDf = oneHotEncoding(binaryEncodedDf)
    # Temporarily show all columns
    with pd.option_context('display.max_columns', None):
        print(oneHotEncodedDf.head())
    print(oneHotEncodedDf.shape)
    print(oneHotEncodedDf.isnull().sum())

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
    # run with : docker-compose run --rm app python XGB.py
    # update with :  docker-compose up --build app



