import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from vega_datasets import data
import numpy as np
from acquire import acquire
from datetime import timedelta, datetime



def prep(df):
    '''
        prep is taking in a dataframe dropping the index column, making date as 
        the index.  It is also making three new columns month, day_of_week, and sales_total
    
    '''
    
    # Removing column with nulls
    df = df.drop(columns=['index'])
    
    # Converting sale_date type object to datetime64
    df.sale_date = pd.to_datetime(df.sale_date)
    
    # Making the index equal to the sale_date
    df = df.set_index('sale_date').sort_index()
    
    # Creating a column named month and using sale_date.dt.month to extract just the month
    df['month'] = pd.DatetimeIndex(df.index).month
    
    # Creating a column named month and using sale_date.dt.month to extract just the month
    df['day_of_week'] = pd.DatetimeIndex(df.index).day_name()
    
    # Multiplying sales_amount by item price. Saving as sales_total
    df['sales_total'] = df['sale_amount'] * df['item_price']
    
    return df