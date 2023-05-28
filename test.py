from datetime import datetime, timedelta
import pickle
import time
import config
import pandas as pd
from binance.client import Client

from bot_funtions import *


def week_over_week(client,coin,acc_balance):
    try:
        week_over_week_df = pd.read_csv('week_over_week_df.csv')
    except Exception as e:
        week_over_week_df = pd.DataFrame(columns=['date', 'month', 'income', 'day', 'weekday', 'balance'])

    end_date=datetime.now()
    star_date=datetime.now()-timedelta(days=90)

    end_date=time.mktime(end_date.timetuple())
    star_date=time.mktime(star_date.timetuple())
    end_date=int(end_date)*1000
    star_date=int(star_date)*1000

    data=client.futures_income_history(symbol=f'{coin}USDT',startTime=star_date,endTime=end_date,limit=1000)
    data_BUSD=client.futures_income_history(symbol=f'{coin}BUSD',startTime=star_date,endTime=end_date,limit=1000)

    data.extend(data_BUSD)

    df=pd.DataFrame(data)
    df['time']=df['time'].apply(lambda x:datetime.fromtimestamp(x/1000))
    df['time']=pd.to_datetime(df['time'])
    df['month']=df['time'].dt.month
    df['date']=df['time'].dt.day
    df['income']=df['income'].astype('float')


    df_fin=df.groupby(['date','month']).agg({'income':'sum'}).reset_index()
    df_fin['day']=df_fin[['date','month']].apply(lambda x: datetime(datetime.now().year,x['month'], x['date'], 0, 0, 0, 0) ,axis=1)
    df_fin.sort_values(by=['day'],inplace=True)

    df_fin['day'] = pd.to_datetime(df_fin['day'])
    df_weekly = df_fin.resample('W', on='day').agg({'income': 'sum'}).reset_index()

    current_week_dict = {
    df_weekly['day'].iloc[-1] : acc_balance
}
    
    with open('data/week_over_week_dict.pkl', 'rb') as file:
        week_over_week_dict = pickle.load(file)

    week_over_week_dict = combine_dicts(week_over_week_dict,current_week_dict)

    with open('data/week_over_week_dict.pkl', 'wb') as file:
        pickle.dump(week_over_week_dict, file)

    for index, row in df_weekly.iterrows():
        day = row['day']
        
        # Check if the day matches with the keys in current_week_dict
        if day in current_week_dict:
            # Assign the value from current_week_dict to the 'balance' column
            df_weekly.loc[index, 'balance'] = current_week_dict[day]
        else:
            # Assign 88 to the 'balance' column if no match is found
            df_weekly.loc[index, 'balance'] = 50

    week_over_week_df = pd.concat([week_over_week_df,df_weekly],axis=0)

    week_over_week_df['day'] = pd.to_datetime(week_over_week_df['day'])

    week_over_week_df['day_duplicates'] = week_over_week_df['day'].dt.date

    week_over_week_df.drop_duplicates(subset='day_duplicates', keep='first', inplace=True)

    week_over_week_df.to_csv('week_over_week_df.csv',index=False,mode='w+')

    week_over_week_df = week_over_week_df[week_over_week_df['income']!=0]

    week_over_week_df['change'] = round((week_over_week_df['income']/week_over_week_df['balance'])*100,2)

    

    week_over_week_df[['day','change']].to_csv('change.csv',index=False,mode='w+')

    send_mail('change.csv')


client=Client(config.api_key,config.secret_key)
acc_balance = round(float(client.futures_account()['totalCrossWalletBalance']),2)
coin = 'ETH'
week_over_week(client,coin,acc_balance)