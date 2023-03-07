import config
import pandas as pd
from binance.client import Client
import math
from bot_funtions import send_mail
import time
from datetime import datetime,timedelta
import time



client=Client(config.api_key,config.secret_key)

end_date=datetime.now()
star_date=datetime(datetime.now().year,datetime.now().month, 1, 0, 0, 0, 239502)

end_date=time.mktime(end_date.timetuple())
star_date=time.mktime(star_date.timetuple())
end_date=int(end_date)*1000
star_date=int(star_date)*1000

data=client.futures_income_history(startTime=star_date,endTime=end_date,limit=1000)


df=pd.DataFrame(data)
df['time']=df['time'].apply(lambda x:datetime.fromtimestamp(x/1000))
df['time']=pd.to_datetime(df['time'])
df['month']=df['time'].dt.month
df['date']=df['time'].dt.day
df['income']=df['income'].astype('float')


df_fin=df.groupby(['date','month']).agg({'income':'sum'}).reset_index()
df_fin['day']=df_fin[['date','month']].apply(lambda x: datetime(datetime.now().year,x['month'], x['date'], 0, 0, 0, 0) ,axis=1)

df_fin.sort_values(by=['day'],inplace=True)

df_fin.to_csv('current_month.csv',index=False,mode='w+')

send_mail('current_month.csv')