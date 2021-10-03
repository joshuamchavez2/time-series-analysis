
import pandas as pd
import requests
import os


def get_items():
    base_url = 'https://python.zgulde.net'
    # first iteration
    response = requests.get(base_url + '/api/v1/items')
    data = response.json()
    df = pd.DataFrame(data['payload']['items'])
    
    # 2nd iteration
    response = requests.get(base_url + data['payload']['next_page'])
    data = response.json()
    df_item_page2 = pd.DataFrame(data['payload']['items'])
    df = pd.concat([df, df_item_page2]).reset_index()
    
    # 3rd iteration
    response = requests.get(base_url + data['payload']['next_page'])
    data = response.json()
    df = pd.concat([df, pd.DataFrame(data['payload']['items'])]).reset_index()
    
    return df

def get_stores():
    base_url = 'https://python.zgulde.net'
    response = requests.get(base_url + '/api/v1/stores')
    data = response.json()
    df = pd.DataFrame(data['payload']['stores'])
    return df

def get_sales():
    base_url = 'https://python.zgulde.net'
    # Before the loop start
    response = requests.get(base_url + '/api/v1/sales')
    data = response.json()
    maxpage = data['payload']['max_page']
    df_sales = pd.DataFrame(data['payload']['sales'])
    
    # Loop
    # Creating a loop from page 2 to maxpage + 1.  I did +1 because range's last value is not included
    for index in range(2, maxpage+1):
        
        # Get new reponse with next page 
        response = requests.get(base_url + data['payload']['next_page'])
        
        # save to data
        data = response.json()
        
        # concat new data to existing df
        df_sales = pd.concat([df_sales, pd.DataFrame(data['payload']['sales'])])
    
    return df_sales

def data():
    
    df_items = get_items()
    df_sales = get_sales()
    df_stores = get_stores()
    
    #Merge 1
    sales_plus_stores = pd.merge(
    df_sales,
    df_stores,
    how='left',
    left_on='store',
    right_on='store_id')
    
    #merge 2
    everything = pd.merge(
    sales_plus_stores,
    df_items,
    how='left',
    left_on='item',
    right_on='item_id')
    
    return everything

def acquire():
    '''
    This function reads in the time series data from https://python.zgulde.net, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('sales.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('sales.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = data()
        
        # Write DataFrame to a csv file.
        df.to_csv('sales.csv')
        
    return df