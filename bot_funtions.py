from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import talib
import math
import requests
import time
import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.mime.text import MIMEText
import websocket
import pandas as pd
import json
from datetime import datetime
from numba import njit
import traceback
import sys


def candle_size(x,coin):
    return abs(((x['close']-x['open'])/x['open'])*100)

def supertrend(coin,df, period, atr_multiplier,pivot_period):
    
    
    pivot_period=pivot_period
    trend_atr=atr_multiplier
    trend_period=period
    
    
    df['OpenTime']=pd.to_datetime(df['OpenTime'])
    
    df['size']=df.apply(candle_size,axis=1,coin=coin)
    
    df['ma_7']=talib.MA(df['close'], timeperiod=7)
    df['ma_25']=talib.MA(df['close'], timeperiod=25)
    df['ma_40']=talib.MA(df['close'], timeperiod=40)
    df['ma_55']=talib.MA(df['close'], timeperiod=55)
    df['ma_99']=talib.MA(df['close'], timeperiod=99)
    df['ma_100']=talib.MA(df['close'], timeperiod=100)
    df['ma_200']=talib.MA(df['close'], timeperiod=200)

    
    
    df['ema_5']=talib.EMA(df['close'],5)
    df['ema_20']=talib.EMA(df['close'],20)
    df['ema_55']=talib.EMA(df['close'],55)
    df['ema_100']=talib.EMA(df['close'],100)
    df['ema_200']=talib.EMA(df['close'],200)
    
    df['ema_9']=talib.EMA(df['close'],9)
    
    df['prev_close']=df['close'].shift(1)
    df['prev_open']=df['open'].shift(1)
    
    df['color']=df.apply(lambda x: 1 if x['close']>x['open'] else -1,axis=1)
    

    df['ema_33']=talib.EMA(df['close'],33)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'],df['macdsignal'],df['macdhist']=talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)


    df['slowk'], df['slowd'] = talib.STOCH(df['high'],df['low'],df['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    
    
    
    
    df['pivot_high'] = pivot(df['high'], pivot_period, pivot_period, 'high')
    df['pivot_low'] = pivot(df['low'], pivot_period, pivot_period, 'low')
    df['atr']=talib.ATR(df['high'], df['low'], df['close'], timeperiod=trend_period)
        
    df['pivot_high']=df['pivot_high'].shift(pivot_period)
    df['pivot_low']=df['pivot_low'].shift(pivot_period)
    
    center = np.NaN
    lastpp=np.NaN
    centers=[np.NaN]
    for idx,row in df.iterrows():
        ph=row['pivot_high']
        pl=row['pivot_low']
        
        if ph:
            lastpp = ph
        elif pl:
            lastpp = pl
        else:
            lastpp=np.NaN
            
            
        if not math.isnan(lastpp):
            if math.isnan(centers[-1]): 
                centers.append(lastpp)
            else:         
                center = round(((centers[-1] * 2) + lastpp)/3,3)
                centers.append(center)
        df.at[idx,'center']=center
    
    df.ffill(axis=0,inplace=True) 
    df['up']=df['center']-(trend_atr*df['atr'])
    df['down']=df['center']+(trend_atr*df['atr'])
    
    Tup=[np.NaN]
    Tdown=[np.NaN]
    Trend=[0]
    df['prev_close']=df['close'].shift(1)
    for idx,row in df.iterrows():
        if row['prev_close'] > Tup[-1]:
            Tup.append(max(row['up'],Tup[-1]))
        else:
            Tup.append(row['up'])
            
        if row['prev_close'] < Tdown[-1]:
            Tdown.append(min(row['down'],Tdown[-1]))
        else:
            Tdown.append(row['down'])
            
        if row['close'] > Tdown[-1]:
            df.at[idx,'in_uptrend']=True
            Trend.append(True)
        elif row['close'] < Tup[-1]:
            df.at[idx,'in_uptrend']=False
            Trend.append(False)
        else:
            if math.isnan(Trend[-1]):
                df.at[idx,'in_uptrend']=True
                Trend.append(True)
            else:
                df.at[idx,'in_uptrend']=Trend[-1]
                Trend.append(Trend[-1])
                
    Tup.pop(0)
    Tdown.pop(0)
    df['lower_band']=Tup
    df['upper_band']=Tdown
    return df


def pivot(osc, LBL, LBR, highlow):
    left = []
    right = []
    pivots=[]
    for i in range(len(osc)):
        pivots.append(0.0)
        if i < LBL + 1:
            left.append(osc[i])
        if i > LBL:
            right.append(osc[i])
        if i > LBL + LBR:
            left.append(right[0])
            left.pop(0)
            right.pop(0)
            if checkhl(left, right, highlow):
                pivots[i - LBR] = osc[i - LBR]
    return pivots


def checkhl(data_back, data_forward, hl):
    if hl == 'high' or hl == 'High':
        ref = data_back[len(data_back)-1]
        for i in range(len(data_back)-1):
            if ref < data_back[i]:
                return 0
        for i in range(len(data_forward)):
            if ref <= data_forward[i]:
                return 0
        return 1
    if hl == 'low' or hl == 'Low':
        ref = data_back[len(data_back)-1]
        for i in range(len(data_back)-1):
            if ref > data_back[i]:
                return 0
        for i in range(len(data_forward)):
            if ref >= data_forward[i]:
                return 0
        return 1
    
def ema_pos(x,col_name):
    if x['close'] > x[col_name]:
        return 1
    else:
        return -1
    
def close_position(client,coin,signal):
    if signal == 'Buy':
        client.futures_create_order(symbol=f'{coin}USDT', side='SELL', type='MARKET', quantity=1000,dualSidePosition=True,positionSide='LONG')
    else:
        client.futures_create_order(symbol=f'{coin}USDT', side='BUY', type='MARKET', quantity=1000,dualSidePosition=True,positionSide='SHORT')
        
def close_position_busd(client,coin,signal):
    if signal == 'Buy':
        client.futures_create_order(symbol=f'{coin}BUSD', side='SELL', type='MARKET', quantity=1000,dualSidePosition=True,positionSide='LONG')
    else:
        client.futures_create_order(symbol=f'{coin}BUSD', side='BUY', type='MARKET', quantity=1000,dualSidePosition=True,positionSide='SHORT')


@njit
def cal_numba(opens,highs,lows,closes,in_uptrends,profit_perc,sl_perc,upper_bands,lower_bands,colors,rsis,macdhists,slowks,slowds,volumes):
    entries=np.zeros(len(opens))
    signals=np.zeros(len(opens))  #characters  1--> buy  2--->sell
    tps=np.zeros(len(opens))
    trades=np.zeros(len(opens))  #characters   1--->w  0---->L
    close_prices=np.zeros(len(opens))
    time_index=np.zeros(len(opens))
    candle_count=np.zeros(len(opens))
    local_max=np.zeros(len(opens))
    local_min=np.zeros(len(opens))
    upper=np.zeros(len(opens))
    lower=np.zeros(len(opens))
    next_colors=np.zeros(len(opens))
    local_max_bar=np.zeros(len(opens))
    local_min_bar=np.zeros(len(opens))
    
    
    next_colors=np.zeros(len(opens))
    local_max_bar=np.zeros(len(opens))
    local_min_bar=np.zeros(len(opens))
    next_close=np.zeros(len(opens))
    indication = 0
    buy_search=0
    sell_search=1
    change_index=0
    local_max_bar_2=np.zeros(len(opens))
    local_min_bar_2=np.zeros(len(opens))
    local_max_2=np.zeros(len(opens))
    local_min_2=np.zeros(len(opens))
    
    prev_candle_0_color=np.zeros(len(opens),dtype=np.float64)
    prev_candle_1_color=np.zeros(len(opens),dtype=np.float64)
    prev_candle_2_color=np.zeros(len(opens),dtype=np.float64)
    prev_candle_3_color=np.zeros(len(opens),dtype=np.float64)
    prev_candle_4_color=np.zeros(len(opens),dtype=np.float64)
    
    prev_candle_0_rsi=np.zeros(len(opens))
    prev_candle_1_rsi=np.zeros(len(opens),dtype=np.float64)
    prev_candle_2_rsi=np.zeros(len(opens),dtype=np.float64)
    prev_candle_3_rsi=np.zeros(len(opens),dtype=np.float64)
    prev_candle_4_rsi=np.zeros(len(opens),dtype=np.float64)
    
    prev_candle_0_macd=np.zeros(len(opens),dtype=np.float64)
    prev_candle_1_macd=np.zeros(len(opens),dtype=np.float64)
    prev_candle_2_macd=np.zeros(len(opens),dtype=np.float64)
    prev_candle_3_macd=np.zeros(len(opens),dtype=np.float64)
    prev_candle_4_macd=np.zeros(len(opens),dtype=np.float64)
    
    prev_candle_0_slowk=np.zeros(len(opens),dtype=np.float64)
    prev_candle_1_slowk=np.zeros(len(opens),dtype=np.float64)
    prev_candle_2_slowk=np.zeros(len(opens),dtype=np.float64)
    prev_candle_3_slowk=np.zeros(len(opens),dtype=np.float64)
    prev_candle_4_slowk=np.zeros(len(opens),dtype=np.float64)
    
    prev_candle_0_slowd=np.zeros(len(opens),dtype=np.float64)
    prev_candle_1_slowd=np.zeros(len(opens),dtype=np.float64)
    prev_candle_2_slowd=np.zeros(len(opens),dtype=np.float64)
    prev_candle_3_slowd=np.zeros(len(opens),dtype=np.float64)
    prev_candle_4_slowd=np.zeros(len(opens),dtype=np.float64)
    
    
    prev_candle_0_volume=np.zeros(len(opens))
    prev_candle_1_volume=np.zeros(len(opens))
    prev_candle_2_volume=np.zeros(len(opens))
    prev_candle_3_volume=np.zeros(len(opens))
    prev_candle_4_volume=np.zeros(len(opens))
    
    indication = 0
    buy_search=0
    sell_search=1
    change_index=0
    i=-1
    while(i<len(opens)):
        i=i+1
        
        if (indication == 0) & (sell_search == 1) & (buy_search == 0) & (change_index == i):
            
            sell_search=0
            flag=0
            trade= 5
            while (indication == 0):
                
                entry = closes[i]
                tp = entry - (entry * profit_perc)
                sl = entry + (entry * sl_perc)
                
                upper[i]=upper_bands[i]
                lower[i]=lower_bands[i]
                
                
                entries[i]=entry
                tps[i]=tp
                signals[i]=2
                local_max[i]=highs[i+1]
                local_min[i]=lows[i+1]
                local_max_2[i]=highs[i+2]
                local_min_2[i]=lows[i+2]
                next_colors[i]=colors[i+1]
                next_close[i]=closes[i+1]
                
                prev_candle_0_color[i]=colors[i]    
                prev_candle_1_color[i]=colors[i-1] 
                prev_candle_2_color[i]=colors[i-2]
                prev_candle_3_color[i]=colors[i-3]
                prev_candle_4_color[i]=colors[i-4]
                
                prev_candle_0_rsi[i]=rsis[i]  
                prev_candle_1_rsi[i]=rsis[i-1]
                prev_candle_2_rsi[i]=rsis[i-2]
                prev_candle_3_rsi[i]=rsis[i-3]
                prev_candle_4_rsi[i]=rsis[i-4]
                
                prev_candle_0_macd[i]=macdhists[i]   
                prev_candle_1_macd[i]=macdhists[i-1]
                prev_candle_2_macd[i]=macdhists[i-2]
                prev_candle_3_macd[i]=macdhists[i-3]
                prev_candle_4_macd[i]=macdhists[i-4]
                
                prev_candle_0_slowk[i]=slowks[i]  
                prev_candle_1_slowk[i]=slowks[i-1]
                prev_candle_2_slowk[i]=slowks[i-2]
                prev_candle_3_slowk[i]=slowks[i-3]
                prev_candle_4_slowk[i]=slowks[i-4]
                
                prev_candle_0_slowd[i]=slowds[i]   
                prev_candle_1_slowd[i]=slowds[i-1]
                prev_candle_2_slowd[i]=slowds[i-2]
                prev_candle_3_slowd[i]=slowds[i-3]
                prev_candle_4_slowd[i]=slowds[i-4]
                
                 
                prev_candle_0_volume[i]=volumes[i]   
                prev_candle_1_volume[i]=volumes[i-1] 
                prev_candle_2_volume[i]=volumes[i-2]
                prev_candle_3_volume[i]=volumes[i-3]
                prev_candle_4_volume[i]=volumes[i-4]
                
                
                
                
            
                for j in range(i+1,len(opens)):
                    candle_count[i]=candle_count[i]+1
                    if candle_count[i] > 2:
                        if lows[j] < local_min_2[i]:
                            local_min_2[i]=lows[j]
                            local_min_bar_2[i]=candle_count[i]
                        if highs[j]>local_max_2[i]:
                            local_max_2[i]=highs[j]
                            local_max_bar_2[i]=candle_count[i]

                    if lows[j] < local_min[i]:
                        local_min[i]=lows[j]
                        local_min_bar[i]=candle_count[i]
                    if highs[j]>local_max[i]:
                        local_max[i]=highs[j]
                        local_max_bar[i]=candle_count[i]

                    if lows[j] < tp and flag==0:

                        trades[i] = 1
                        close_prices[i]=tp
                        time_index[i]=i
                        
                        indication=1
                        buy_search=1
                        flag=1
                        
                        
                    elif (highs[j] > sl and flag==0) or (in_uptrends[j] == 'True'):
                        if highs[j] > sl and flag==0:
                            trades[i] = 0
                            close_prices[i]=sl
                            time_index[i]=i

                            indication=1
                            buy_search=1
                            flag=1
                            
                        if in_uptrends[j] == 'True':
                            

                            if trades[i] ==1:
                                change_index=j
                            elif trades[i] == 0 and flag ==1:
                                change_index=j
                            else:
                                trades[i] = 0
                                close_prices[i]=closes[j]
                                time_index[i]=i
                                change_index=j
                            
                            indication=1
                            buy_search=1
                            break
                    else:
                        pass
                break
        elif (indication == 1 ) & (sell_search == 0) & (buy_search == 1) & (change_index==i):
            
            buy_search= 0
            flag=0

            while (indication == 1):


                entry = closes[i]
                tp = entry + (entry * profit_perc)
                sl = entry - (entry * sl_perc)
                
                upper[i]=upper_bands[i]
                lower[i]=lower_bands[i]
                
                entries[i]=entry
                tps[i]=tp
                signals[i]=1
                local_max[i]=highs[i+1]  
                local_min[i]=lows[i+1]
                next_colors[i]=colors[i+1]
                local_max_2[i]=highs[i+2]
                local_min_2[i]=lows[i+2]
                
                prev_candle_0_color[i]=colors[i]    
                prev_candle_1_color[i]=colors[i-1] 
                prev_candle_2_color[i]=colors[i-2]
                prev_candle_3_color[i]=colors[i-3]
                prev_candle_4_color[i]=colors[i-4]
                
                prev_candle_0_rsi[i]=rsis[i]  
                prev_candle_1_rsi[i]=rsis[i-1]
                prev_candle_2_rsi[i]=rsis[i-2]
                prev_candle_3_rsi[i]=rsis[i-3]
                prev_candle_4_rsi[i]=rsis[i-4]
                
                prev_candle_0_macd[i]=macdhists[i]   
                prev_candle_1_macd[i]=macdhists[i-1]
                prev_candle_2_macd[i]=macdhists[i-2]
                prev_candle_3_macd[i]=macdhists[i-3]
                prev_candle_4_macd[i]=macdhists[i-4]
                
                prev_candle_0_slowk[i]=slowks[i]  
                prev_candle_1_slowk[i]=slowks[i-1]
                prev_candle_2_slowk[i]=slowks[i-2]
                prev_candle_3_slowk[i]=slowks[i-3]
                prev_candle_4_slowk[i]=slowks[i-4]
                
                prev_candle_0_slowd[i]=slowds[i]   
                prev_candle_1_slowd[i]=slowds[i-1]
                prev_candle_2_slowd[i]=slowds[i-2]
                prev_candle_3_slowd[i]=slowds[i-3]
                prev_candle_4_slowd[i]=slowds[i-4]
                
                 
                prev_candle_0_volume[i]=volumes[i]   
                prev_candle_1_volume[i]=volumes[i-1] 
                prev_candle_2_volume[i]=volumes[i-2]
                prev_candle_3_volume[i]=volumes[i-3]
                prev_candle_4_volume[i]=volumes[i-4]
                
                next_close[i]=closes[i+1]
                
                
                
                
                for j in range(i+1,len(opens)):
                    candle_count[i]=candle_count[i]+1
                    if candle_count[i] > 2:
                        if lows[j] < local_min_2[i]:
                            local_min_2[i]=lows[j]
                            local_min_bar_2[i]=candle_count[i]
                        if highs[j]>local_max_2[i]:
                            local_max_2[i]=highs[j]
                            local_max_bar_2[i]=candle_count[i]
                    if lows[j] < local_min[i]:
                        local_min[i]=lows[j]
                        local_min_bar[i]=candle_count[i]
                    if highs[j]>local_max[i]:
                        local_max[i]=highs[j]
                        local_max_bar[i]=candle_count[i]
                        
                    
                    if highs[j] > tp and flag==0 :
                        trades[i]  = 1
                        sell_search=1
                        close_prices[i]=tp
                        time_index[i]=i
                        

                        flag=1
                        indication=0
                    elif (lows[j] < sl and flag==0) or (in_uptrends[j] == 'False'):
                        if lows[j] < sl and flag==0:

                            trades[i]= 0
                            close_prices[i]=sl
                            time_index[i]=i
                            indication=0
                            sell_search=1
                            flag=1
                            
                        if in_uptrends[j] == 'False':
                            
                            if trades[i] ==1:
                                change_index=j
                            elif trades[i] == 0 and flag ==1:
                                change_index=j
                            else:
                                trades[i] = 0
                                close_prices[i]=closes[j]
                                time_index[i]=i
                                change_index=j
                            
                            indication=0
                            sell_search=1
                            break

                    
                        
                    else:
                        pass
                break
        else:
            continue
        
    return entries,signals,tps,trades,close_prices,time_index,candle_count,local_max,local_min,local_max_bar,local_min_bar,upper,lower,next_colors,next_close, \
            prev_candle_0_color,prev_candle_1_color,prev_candle_2_color,prev_candle_3_color,prev_candle_4_color, \
                prev_candle_0_rsi,prev_candle_1_rsi,prev_candle_2_rsi,prev_candle_3_rsi,prev_candle_4_rsi, \
                    prev_candle_0_macd,prev_candle_1_macd,prev_candle_2_macd,prev_candle_3_macd,prev_candle_4_macd, \
                        prev_candle_0_slowk,prev_candle_1_slowk,prev_candle_2_slowk,prev_candle_3_slowk,prev_candle_4_slowk, \
                            prev_candle_0_slowd,prev_candle_1_slowd,prev_candle_2_slowd,prev_candle_3_slowd,prev_candle_4_slowd, \
                                prev_candle_0_volume,prev_candle_1_volume,prev_candle_2_volume,prev_candle_3_volume,prev_candle_4_volume, \
                                    local_max_bar_2,local_min_bar_2,local_max_2,local_min_2   


def create_signal_df(super_df,df,coin,timeframe,atr1,period,profit,sl):
    opens=super_df['open'].to_numpy(dtype='float64')
    highs=super_df['high'].to_numpy(dtype='float64')
    lows=super_df['low'].to_numpy(dtype='float64')
    closes=super_df['close'].to_numpy(dtype='float64')
    in_uptrends=super_df['in_uptrend'].to_numpy(dtype='U5')
    upper_bands=super_df['upper_band'].to_numpy(dtype='float64')
    lower_bands=super_df['lower_band'].to_numpy(dtype='float64')
    colors=super_df['color'].to_numpy(dtype='float64')
    
    super_df['rsi']=round(super_df['rsi'],2)
    rsis=super_df['rsi'].to_numpy(dtype='float64')
   
    macdhists=super_df['macdhist'].to_numpy(dtype=np.float64)
    slowks=super_df['slowk'].to_numpy(dtype=np.float64)
   
    slowds=super_df['slowd'].to_numpy(dtype=np.float64)
    volumes=super_df['volume'].to_numpy(dtype=np.float64)
    
    

    
  
    entries,signals,tps,trades,close_prices,time_index,candle_count,local_max,local_min,local_max_bar,local_min_bar,upper,lower,colors,next_close, \
        prev_candle_0_color,prev_candle_1_color,prev_candle_2_color,prev_candle_3_color,prev_candle_4_color, \
            prev_candle_0_rsi,prev_candle_1_rsi,prev_candle_2_rsi,prev_candle_3_rsi,prev_candle_4_rsi, \
                prev_candle_0_macd,prev_candle_1_macd,prev_candle_2_macd,prev_candle_3_macd,prev_candle_4_macd, \
                    prev_candle_0_slowk,prev_candle_1_slowk,prev_candle_2_slowk,prev_candle_3_slowk,prev_candle_4_slowk, \
                        prev_candle_0_slowd,prev_candle_1_slowd,prev_candle_2_slowd,prev_candle_3_slowd,prev_candle_4_slowd, \
                            prev_candle_0_volume,prev_candle_1_volume,prev_candle_2_volume,prev_candle_3_volume,prev_candle_4_volume, \
                                local_max_bar_2,local_min_bar_2,local_max_2,local_min_2 =cal_numba(opens,highs,lows,closes,in_uptrends,profit,sl,upper_bands,lower_bands,colors,rsis,macdhists,slowks,slowds,volumes)
    
    trade_df=pd.DataFrame({'signal':signals,'entry':entries,'tp':tps,'trade':trades,'close_price':close_prices,'candle_count':candle_count,
                           'local_max':local_max,'local_min':local_min,'local_max_bar':local_max_bar,'local_min_bar':local_min_bar,
                           'upper_band':upper,'lower_band':lower,'next_color':colors,'next_close':next_close,
                           'prev_candle_0_color':prev_candle_0_color,'prev_candle_1_color':prev_candle_1_color,'prev_candle_2_color':prev_candle_2_color,'prev_candle_3_color':prev_candle_3_color,'prev_candle_4_color':prev_candle_4_color,
                           'prev_candle_0_rsi':prev_candle_0_rsi,'prev_candle_1_rsi':prev_candle_1_rsi,'prev_candle_2_rsi':prev_candle_2_rsi,'prev_candle_3_rsi':prev_candle_3_rsi,'prev_candle_4_rsi':prev_candle_4_rsi,
                           'prev_candle_0_macd':prev_candle_0_macd,'prev_candle_1_macd':prev_candle_1_macd,'prev_candle_2_macd':prev_candle_2_macd,'prev_candle_3_macd':prev_candle_3_macd,'prev_candle_4_macd':prev_candle_4_macd,
                           'prev_candle_0_slowk':prev_candle_0_slowk,'prev_candle_1_slowk':prev_candle_1_slowk,'prev_candle_2_slowk':prev_candle_2_slowk,'prev_candle_3_slowk':prev_candle_3_slowk,'prev_candle_4_slowk':prev_candle_4_slowk,
                            'prev_candle_0_slowd':prev_candle_0_slowd,'prev_candle_1_slowd':prev_candle_1_slowd,'prev_candle_2_slowd':prev_candle_2_slowd,'prev_candle_3_slowd':prev_candle_3_slowd,'prev_candle_4_slowd':prev_candle_4_slowd,   
                            'prev_candle_0_volume':prev_candle_0_volume,'prev_candle_1_volume':prev_candle_1_volume,'prev_candle_2_volume':prev_candle_2_volume,'prev_candle_3_volume':prev_candle_3_volume,'prev_candle_4_volume':prev_candle_4_volume,
                                               'local_max_bar_2':local_max_bar_2,'local_min_bar_2':local_min_bar_2,'local_max_2':local_max_2,'local_min_2':local_min_2 
                           
                           
                           
                           
                           })
    # before_drop=trade_df.shape[0]
    # print(f'Number of columns before drop : {before_drop}')
    
    
    trade_df_index=trade_df[trade_df['entry']!=0]
    
    indexes=trade_df_index.index.to_list()
    
    df=super_df
    
    print(df.shape[0])
    print(trade_df.shape[0])
    print(super_df.shape[0])
    for i in indexes:
        try:
            trade_df.at[i,'TradeOpenTime']=df[df.index==i+1]['OpenTime'][(i+1)]
        except KeyError:
            trade_df.at[i,'TradeOpenTime']=(df[df.index==i]['OpenTime'][(i)]) 
    for i in indexes:
        try:
            trade_df.at[i,'signalTime']=df[df.index==i]['OpenTime'][(i)]
        except KeyError:
            trade_df.at[i,'signalTime']=(df[df.index==i]['OpenTime'][(i)])
            
    trade_df['signal']=trade_df['signal'].apply(signal_decoding)
    
    trade_df.dropna(inplace=True)
                        
    entries=trade_df['entry'].to_numpy(dtype='float64')
    closes=trade_df['close_price'].to_numpy(dtype='float64')
    # trades=trade_df['trade'].to_numpy(dtype='U1')
    signals=trade_df['signal'].to_numpy(dtype='U5')
    outputs=np.zeros(len(entries))
    
   
    
    percentages=df_perc_cal(entries,closes,signals,outputs)
    trade_df['percentage'] = percentages.tolist()
    trade_df['trade']=trade_df['percentage'].apply(trade_decoding)
    # after_drop=trade_df.shape[0]
    # print(f'Number of columns after drop : {after_drop}')
    trade_df=trade_df.reset_index(drop=True)
    if (trade_df['percentage'][trade_df.shape[0]-1]==-1) | (trade_df['percentage'][trade_df.shape[0]-1]==1):
        trade_df=trade_df[:-1]
    else:
        pass
    trade_df['signalTime']=pd.to_datetime(trade_df['signalTime'])
    super_df['OpenTime']=pd.to_datetime(super_df['OpenTime'])
    
    trade_df=pd.merge(trade_df, super_df, how='left', left_on=['signalTime'], right_on = ['OpenTime'])
    
    trade_df=trade_df[['signal',
    'entry',
    'tp',
    'trade',
    'close_price',
    'TradeOpenTime',
    'percentage',
    'OpenTime',
    'hour',
    'minute','day',
    'month',
    'size','ma_7','ma_25','ma_99',
    'ema_9',
    'ma_40','ma_55','ema_20','ema_5','ema_55','ma_100','ma_200','ema_100','ema_200',
    'ema_33',
    'rsi',
    'macd',
    'macdsignal',
    'macdhist',
    'slowk',
    'slowd',
    'candle_count',
    'local_max','local_min',
    'local_max_bar','local_min_bar',
    'upper_band','lower_band','next_color','next_close',
    'prev_candle_0_color','prev_candle_1_color','prev_candle_2_color','prev_candle_3_color','prev_candle_4_color', \
            'prev_candle_0_rsi','prev_candle_1_rsi','prev_candle_2_rsi','prev_candle_3_rsi','prev_candle_4_rsi', \
                'prev_candle_0_macd','prev_candle_1_macd','prev_candle_2_macd','prev_candle_3_macd','prev_candle_4_macd', \
                    'prev_candle_0_slowk','prev_candle_1_slowk','prev_candle_2_slowk','prev_candle_3_slowk','prev_candle_4_slowk', \
                        'prev_candle_0_slowd','prev_candle_1_slowd','prev_candle_2_slowd','prev_candle_3_slowd','prev_candle_4_slowd', \
                            'prev_candle_0_volume','prev_candle_1_volume','prev_candle_2_volume','prev_candle_3_volume','prev_candle_4_volume',
                            'local_max_bar_2','local_min_bar_2','local_max_2','local_min_2']]
    
    trade_df=trade_df.dropna()
    trade_df=trade_df[2:]
    trade_df.to_csv(f'data/file.csv',index=False,mode='w+')
    
    return trade_df        
telegram_auth_token='5515290544:AAG9T15VaY6BIxX2VYX8x2qr34aC-zVEYMo'
telegram_group_id='notifier2_scanner_bot_link' 


def signal_decoding(x):
    if x == 1:
        return 'Buy'
    else:
        return 'Sell'
    
def trade_decoding(x):
    if x > 0:
        return 'W'
    else:
        return 'L'
    
@njit
def df_perc_cal(entries,closes,signals,percentages):
    for i in range(0,len(entries)):
        if signals[i]=='Buy':
            percentages[i]=(closes[i]-entries[i])/entries[i]
        else:
            percentages[i]=-(closes[i]-entries[i])/entries[i]
    return percentages
    
        
def notifier(message,tries=0):
    telegram_api_url=f'https://api.telegram.org/bot{telegram_auth_token}/sendMessage?chat_id=@{telegram_group_id}&text={message}'
    #https://api.telegram.org/bot5515290544:AAG9T15VaY6BIxX2VYX8x2qr34aC-zVEYMo/sendMessage?chat_id=@notifier2_scanner_bot_link&text=hii
    tel_resp=requests.get(telegram_api_url)
    if tel_resp.status_code==200:
        pass
    else:
        while(tries < 25):
            print(f'Telegram notifier problem retrying {tries}')
            tries+=1
            time.sleep(0.5)
            notifier(message,tries)
            
        
def condition_usdt(timeframe,pivot_period,atr1,period,ma_condition,exchange,client,coin,sleep_time,in_trade_usdt,in_trade_busd,lock):
    print(f'timeframe : {timeframe}')
    notifier(f'Starting USDT function,SARAVANA BHAVA' )
    restart=0
    while(True):
        if restart==1:
            notifier('USDT Restarted succesfully')
            restart=0
        try:
            ws = websocket.WebSocket()
            ws.connect(f"wss://fstream.binance.com/ws/{str.lower(coin)}usdt@kline_{timeframe}")
            notifier(f'Started USDT function : {timeframe}')
            ws.settimeout(15)
            risk=0.005
            bars = exchange.fetch_ohlcv(f'{coin}/USDT', timeframe=timeframe, limit=1998)
            df = pd.DataFrame(bars[:-1], columns=['OpenTime', 'open', 'high', 'low', 'close', 'volume'])
            df.drop(['OpenTime'],axis=1,inplace=True)
            x_str = str(df['close'].iloc[-1])
            decimal_index = x_str.find('.')
            round_price = len(x_str) - decimal_index - 1
            exchange_info = client.futures_exchange_info()
            
            
            for symbol in exchange_info['symbols']:
                if symbol['symbol'] == f"{coin}USDT":                   
                    round_quantity=symbol['quantityPrecision']
                    break
            notifier(round_quantity)
            indicator=0
            weight_reduce=0
            while True:
                result = ws.recv()
                data = json.loads(result)
                if data['k']['x']==True:
                    candle=data['k']
                    candle_data=[candle['t'],candle['o'],candle['h'],candle['l'],candle['c'],candle['v']]
                    temp_df = pd.DataFrame([candle_data], columns=['OpenTime','open', 'high', 'low', 'close', 'volume'])
                    df=pd.concat([df,temp_df])
                    df=df[2:]
                    df=df.reset_index(drop=True)
                    df = df.astype(float)
                    super_df=supertrend(coin,df, period, atr1,pivot_period)
                    super_df[f'{ma_condition}_pos']=super_df[[ma_condition,'close']].apply(ema_pos,col_name=ma_condition,axis=1)
                    ma_pos=super_df.iloc[-1][f'{ma_condition}_pos']
                    if super_df.iloc[-1]['in_uptrend'] != super_df.iloc[-2]['in_uptrend']: 

                        trade_df=create_signal_df(super_df,df,coin,timeframe,atr1,period,100,100)
                        # Add 'Year' and 'Week' columns to the DataFrame
                        trade_df['Year'] = trade_df['TradeOpenTime'].dt.isocalendar().year
                        trade_df['Week'] = trade_df['TradeOpenTime'].dt.isocalendar().week

                        # Group by the 'Year' and 'Week' columns and sum the 'percentage' column
                        df_weekly = trade_df.groupby(['Year', 'Week'])['percentage'].sum().reset_index()
                        current_week = pd.to_datetime(datetime.now()).isocalendar().week
                        current_year =pd.to_datetime(datetime.now()).isocalendar().year
                        previousWeekPercentage = df_weekly[(df_weekly['Week']==(current_week-1)) & (df_weekly['Year']==current_year)]['percentage'].values[0]
                       
                        lastTradeOpenTime = trade_df.iloc[-1]['OpenTime']
                        lastTradePerc = trade_df.iloc[-1]['percentage']
                        lastTradeOutcome = trade_df.iloc[-1]['trade']
                        notifier(f'from USDT previous trade opentime : {lastTradeOpenTime} , perc : {lastTradePerc} , trade : {lastTradeOutcome}')
                        
                        if previousWeekPercentage < 0:
                            risk = 0.04
                        else:
                            risk = 0.02
                        
                        if trade_df['trade'].iloc[-1]=='W':
                            notifier('Last one was a win reducing the risk')
                            risk = risk/2



                        lock.acquire()
                        
                        try:
                            close_position(client,coin,'Sell') #close open position if any
                            in_trade_usdt.value=0
                            notifier(f'Position Closed {timeframe}')
                        except Exception as err:
                            try:
                                close_position(client,coin,'Buy')
                                notifier(f'Position Closed {timeframe}')
                                in_trade_usdt.value=0
                            except Exception as e:
                                notifier(f'No Open Position to Close {timeframe}')
                                
                            print(err)

                        # print(f'scanning USDT {super_df.iloc[-1][f"OpenTime"]} trade found, ma_pos :{super_df.iloc[-1][f"{ma_condition}_pos"]} and uptrend :{super_df.iloc[-1]["in_uptrend"]},bsud_poisiton :{in_trade_busd.value},usdt_position :{in_trade_usdt.value},sleeping for {sleep_time*60} seconds')
                        acc_balance = round(float(client.futures_account()['totalCrossWalletBalance']),2)
                        
                        stake=(acc_balance*0.88)
                        
                        
                            
                        
                        notifier(f'Allocated stake:{round(stake,2)}')
                        
                        signal = ['Buy' if super_df.iloc[-1]['in_uptrend'] == True else 'Sell'][0]
                        entry=super_df.iloc[-1]['close']
                        
                        if signal == 'Buy':
                            sl=super_df.iloc[-1]['lower_band']
                            sl_perc=(entry-sl)/entry
                        else:
                            sl=super_df.iloc[-1]['upper_band']
                            sl_perc=(sl-entry)/entry
                            
                        stake=(stake*risk)/sl_perc
                        quantity=round(stake/entry,round_quantity)

                    
                        
                        
                        rr=17
                        
                        if signal == 'Buy' and ma_pos == 1:
                            #buy order
                            client.futures_create_order(symbol=f'{coin}USDT', side='BUY', type='MARKET', quantity=quantity,dualSidePosition=True,positionSide='LONG')

                            take_profit=entry+((entry-sl)*rr)
                            client.futures_create_order(
                                    symbol=f'{coin}USDT',
                                    price=round(take_profit,round_price),
                                    side='SELL',
                                    positionSide='LONG',
                                    quantity=quantity,
                                    timeInForce='GTC',
                                    type='LIMIT',
                                    # reduceOnly=True,
                                    closePosition=False,
                                    # stopPrice=round(take_profit,2),
                                    workingType='MARK_PRICE',
                                    priceProtect=True  
                                    )
                            in_trade_usdt.value=1
                            notifier(f'Previous week percentage : {round(previousWeekPercentage,2)} Current risk : {risk}')
                            notifier(f'Risk adjusted stake:{round(stake,2)},entry:{entry},sl_perc: {round(sl_perc,3)}')
                            notifier(f'Trend Changed {signal} and ma condition {ma_condition} is {ma_pos}')
                            notifier(f'Bought @{entry}, Timeframe : {timeframe} , pivot_period: {pivot_period},atr:{atr1},period : {period},ma :{ma_condition}')
                            notifier(f'TP : {take_profit}')
                        elif signal == 'Sell' and ma_pos == -1:
                                
                            #sell order
                            client.futures_create_order(symbol=f'{coin}USDT', side='SELL', type='MARKET', quantity=quantity,dualSidePosition=True,positionSide='SHORT')

                            take_profit=entry-((sl-entry)*rr)
                            client.futures_create_order(
                                                    symbol=f'{coin}USDT',
                                                    price=round(take_profit,round_price),
                                                    side='BUY',
                                                    positionSide='SHORT',
                                                    quantity=quantity,
                                                    timeInForce='GTC',
                                                    type='LIMIT',
                                                    # reduceOnly=True,
                                                    closePosition=False,
                                                    # stopPrice=round(take_profit,2),
                                                    workingType='MARK_PRICE',
                                                    priceProtect=True  
                                                    )
                            in_trade_usdt.value=1
                            notifier(f'Previous week percentage : {round(previousWeekPercentage,2)} Current risk : {risk}')                          
                            notifier(f'Risk adjusted stake:{round(stake,2)},entry:{entry},sl_perc: {round(sl_perc,3)}')
                            notifier(f'Trend Changed {signal} and ma condition {ma_condition} is {ma_pos}')
                            notifier(f'Sold @{entry},Timeframe : {timeframe} , pivot_period: {pivot_period},atr:{atr1},period : {period},ma :{ma_condition}')
                            notifier(f'TP : {take_profit}')
                        else:
                            notifier(f'Not taking the trade')
                        lock.release()
                        
                    else:
                        # print(f'Scanning USDT {super_df.iloc[-1][f"OpenTime"]} trade not found, ma_pos :{super_df.iloc[-1][f"{ma_condition}_pos"]} and uptrend :{super_df.iloc[-1]["in_uptrend"]}, bsud_poisiton :{in_trade_busd.value},usdt_position :{in_trade_usdt.value}')
                        # print(f'ma : {super_df.iloc[-1][ma_condition]},close :{super_df.iloc[-1]["close"]},ma_pos :{super_df.iloc[-1][f"{ma_condition}_pos"]}')
                        notifier(f'{timeframe} candle closed : {coin}')


                        if in_trade_usdt.value==1 and weight_reduce>=1:
                            weight_reduce=0
                            open_orders=client.futures_get_open_orders(symbol=f'{coin}USDT')
                            if len(open_orders)==0:
                                lock.acquire()
                                in_trade_usdt.value=0
                                notifier('USDT Pos closed in profit')
                                lock.release()
                        
                        
                        if indicator>1:
                            indicator=0   #notification every 60 minutes
                            information=client.futures_account()
                            totalUnrealizedProfit=round(float(information['totalUnrealizedProfit']),2)
                            bal=round(float(information['totalCrossWalletBalance']),2)
                            if bal > 50: #Month initial
                                bal_pos='Profit'
                            else:
                                bal_pos='Loss'

                            if totalUnrealizedProfit > 0:
                                profit_pos='Green'
                            elif totalUnrealizedProfit == 0:
                                profit_pos='Neutral'
                            else:
                                profit_pos='Red'



                            notifier(f'SARAVANA BHAVA ! Running... ,USDT POS:{in_trade_usdt.value} , BUSD POS: {in_trade_busd.value},Bal :{bal_pos},PNL:{profit_pos}')                    
                        weight_reduce+=1
                        indicator+=1
                    
        except Exception as err:
            notifier(err)
            notifier(f'Restarting USDT function : {coin}')
            print(err)
            restart=1
            ws.close()
            tb = traceback.extract_tb(sys.exc_info()[2])
            filename, line, func, text = tb[-1]
            print(f'An error occurred on line USDT {line}: {e}')
            time.sleep(10)


            
            
def condition_busdt(timeframe,pivot_period,atr1,period,ma_condition,exchange,client,coin,sleep_time,in_trade_usdt,in_trade_busd,lock):
    notifier(f'Starting BUSD function,SARAVANA BHAVA')
    print(f'timeframe : {timeframe}')
    restart=0
    
    while(True):
        if restart==1:
            notifier('BUSD Restarted succesfully')
            restart=0
        try:
            ws = websocket.WebSocket()
            ws.connect(f"wss://fstream.binance.com/ws/{str.lower(coin)}usdt@kline_{timeframe}")
            ws.settimeout(15)
            notifier(f'Started BUSD function : {timeframe}' )
            risk=0.005
            bars = exchange.fetch_ohlcv(f'{coin}/USDT', timeframe=timeframe, limit=1998)
            df = pd.DataFrame(bars[:-1], columns=['OpenTime', 'open', 'high', 'low', 'close', 'volume'])
            df.drop(['OpenTime'],axis=1,inplace=True)
            x_str = str(df['close'].iloc[-1])
            decimal_index = x_str.find('.')
            round_price = len(x_str) - decimal_index - 1
            exchange_info = client.futures_exchange_info()
            notifier(f'from bsud {coin}')
            print(coin)
            for symbol in exchange_info['symbols']:
                if symbol['symbol'] == f"{coin}BUSD":                   
                    round_quantity=symbol['quantityPrecision']
                    break
                elif symbol['symbol'] == f"{coin}USDT":
                    round_quantity=symbol['quantityPrecision']
                    break
            notifier(f'Round Quantity :{round_quantity}')
            while True:
                result = ws.recv()
                data = json.loads(result)
                if data['k']['x']==True:
                    candle=data['k']
                    candle_data=[candle['t'],candle['o'],candle['h'],candle['l'],candle['c'],candle['v']]
                    temp_df = pd.DataFrame([candle_data], columns=['OpenTime','open', 'high', 'low', 'close', 'volume'])                    
                    df=pd.concat([df,temp_df])
                    df=df[2:]
                    df=df.reset_index(drop=True)
                    df = df.astype(float)
                    super_df=supertrend(coin,df, period, atr1,pivot_period)
                    super_df[f'{ma_condition}_pos']=super_df[[ma_condition,'close']].apply(ema_pos,col_name=ma_condition,axis=1)
                    ma_pos=super_df.iloc[-1][f'{ma_condition}_pos']
                    if super_df.iloc[-1]['in_uptrend'] != super_df.iloc[-2]['in_uptrend']:

                        trade_df=create_signal_df(super_df,df,coin,timeframe,atr1,period,100,100)
                       
                        time.sleep(1)
                        lastTradeOpenTime = trade_df.iloc[-1]['OpenTime']
                        lastTradePerc = trade_df.iloc[-1]['percentage']
                        lastTradeOutcome = trade_df.iloc[-1]['trade']
                        notifier(f'previous trade opentime : {lastTradeOpenTime} , perc : {lastTradePerc} , trade : {lastTradeOutcome}')
                        if trade_df['trade'].iloc[-1]=='W':
                            notifier('Last one was a win reducing the risk')
                            risk = risk/2

                        if trade_df['trade'].iloc[-1]=='W' & trade_df['trade'].iloc[-2]=='W':
                            notifier('Last two were wins reducing the risk drastically')
                            risk = risk/3

                        lock.acquire()
                        
                        try:
                            close_position_busd(client,coin,'Sell') #close open position if any
                            notifier(f'Position Closed {timeframe}')
                            in_trade_busd.value=0   
                        except Exception as err:
                            try:
                                close_position_busd(client,coin,'Buy')
                                notifier(f'Position Closed {timeframe}')
                                in_trade_busd.value=0
                            except Exception as e: 
                                notifier(f'No Position to close {timeframe}')
                                
            
                        # print(f'scanning busd {super_df.iloc[-1][f"OpenTime"]} trade found, ma_pos :{super_df.iloc[-1][f"{ma_condition}_pos"]} and uptrend :{super_df.iloc[-1]["in_uptrend"]}, bsud_poisiton :{in_trade_busd.value},usdt_position :{in_trade_usdt.value} , sleeping for {sleep_time*60} seconds')
                        acc_balance = round(float(client.futures_account()['totalCrossWalletBalance']),2)
                        
                        
                        
                        stake=(acc_balance*0.88)
                       

                        
                        notifier(f'Allocated stake:{round(stake,2)}')
                        
                        signal = ['Buy' if super_df.iloc[-1]['in_uptrend'] == True else 'Sell'][0]
                        entry=super_df.iloc[-1]['close']
                        
                        if signal == 'Buy':
                            sl=super_df.iloc[-1]['lower_band']
                            sl_perc=(entry-sl)/entry
                        else:
                            sl=super_df.iloc[-1]['upper_band']
                            sl_perc=(sl-entry)/entry
                            
                        stake=(stake*risk)/sl_perc
                        quantity=round(stake/entry,round_quantity)


                        
                        if signal == 'Buy' and ma_pos == 1:
                            #buy order
                            client.futures_create_order(symbol=f'{coin}BUSD', side='BUY', type='MARKET', quantity=quantity,dualSidePosition=True,positionSide='LONG')
                            notifier(f'Trend Changed {signal} and ma condition {ma_condition} is {ma_pos},close : {entry} , ma: {super_df.iloc[-1][ma_condition]}')

                            notifier(f'Bought BUSD @{entry} , Timeframe : {timeframe} , pivot_period: {pivot_period},atr:{atr1},period : {period},ma :{ma_condition}')
                            in_trade_busd.value=1
                            notifier(f'Risk adjusted stake:{round(stake,2)},entry:{entry},sl_perc: {round(sl_perc,3)}')
                            
                        elif signal == 'Sell' and ma_pos == -1:
                                
                            #sell order
                            client.futures_create_order(symbol=f'{coin}BUSD', side='SELL', type='MARKET', quantity=quantity,dualSidePosition=True,positionSide='SHORT')
                            notifier(f'Trend Changed {signal} and ma condition {ma_condition} is {ma_pos},close : {entry} , ma: {super_df.iloc[-1][ma_condition]}')

                            notifier(f'Sold BUSD @{entry},Timeframe : {timeframe} , pivot_period: {pivot_period},atr:{atr1},period : {period},ma :{ma_condition}')
                            in_trade_busd.value=1
                            notifier(f'Risk adjusted stake:{round(stake,3)},entry:{entry},sl_perc: {round(sl_perc,3)}')
                        else:
                            notifier(f'Not taking the trade')

                        lock.release()
                    else:
                        notifier(f'{timeframe} candle closed : {coin}')
        except Exception as e:
            notifier(e)
            notifier(f'Restarting BUSD function : {coin}')
            print(e)
            ws.close()
            tb = traceback.extract_tb(sys.exc_info()[2])
            filename, line, func, text = tb[-1]
            print(f'An error occurred on line USDT {line}: {e}')
            time.sleep(10)
            restart=1


def send_mail(filename,subject='SARAVANA BHAVA'):
    from_= 'gannamanenilakshmi1978@gmail.com'
    to= 'vamsikrishnagannamaneni@gmail.com'
    
    message = MIMEMultipart()
    message['From'] = from_
    message['To'] = to
    message['Subject'] =subject
    body_email ='SARAVANA BHAVA !'
    
    message.attach(MIMEText(body_email, 'plain'))
    
    attachment = open(filename, 'rb')
    
    x = MIMEBase('application', 'octet-stream')
    x.set_payload((attachment).read())
    encoders.encode_base64(x)
    
    x.add_header('Content-Disposition', 'attachment; filename= %s' % filename)
    message.attach(x)
    
    s_e = smtplib.SMTP('smtp.gmail.com', 587)
    s_e.starttls()
    
    s_e.login(from_, 'upsprgwjgtxdbwki')
    text = message.as_string()
    s_e.sendmail(from_, to, text)
    print(f'Sent {filename}')