import requests
import pandas as pd
from datetime import datetime
import time
import ccxt
import config
from binance.client import Client
import pandas as pd
from bot_funtions import *
import warnings
warnings.filterwarnings('ignore')
import multiprocessing

telegram_auth_token='5515290544:AAG9T15VaY6BIxX2VYX8x2qr34aC-zVEYMo'
telegram_group_id='notifier2_scanner_bot_link'


        
exchange = ccxt.binance({
    "apiKey": config.api_key,
    "secret": config.secret_key,
    'options': {
    'defaultType': 'future',
    },
})

timeframes_dict={
'1m':1,
'3m':3,
'5m':5,
'15m':15,
'30m':30,
'1h':60,
'2h':120,
'4h':240,
'1d':1440
}


coin='ETH'
timeframe_usdt='30m' 
period_usdt=12
atr1_usdt=1
pivot_period_usdt=10
ma_condition_usdt='ema_100'
time_usdt=timeframes_dict[timeframe_usdt]

timeframe_busd='30m'  
period_busd=12
atr1_busd=1
pivot_period_busd=10
ma_condition_busd='ema_100'
time_busd=timeframes_dict[timeframe_busd]

while(True):
    try:
        client=Client(config.api_key,config.secret_key)

        client.futures_change_leverage(symbol=f'{coin}USDT', leverage=18)
        client.futures_change_leverage(symbol=f'{coin}BUSD', leverage=18)
        notifier(f'SARAVANA BHAVA')
        break
    except Exception as e:
        notifier(f'Met with exception {e}, sleeping for 5 minutes and trying again')
        time.sleep(300)


in_trade_usdt=multiprocessing.Value('i',0)
in_trade_busd=multiprocessing.Value('i',0)
lock=multiprocessing.Lock()

pos=client.futures_position_information(symbol=f'{coin}USDT')
if float(pos[0]['positionAmt']) !=0 or float(pos[1]['positionAmt']) !=0 or float(pos[2]['positionAmt']) !=0:
    in_trade_usdt.value=1

pos=client.futures_position_information(symbol=f'{coin}BUSD')
if float(pos[0]['positionAmt']) !=0 or float(pos[1]['positionAmt']) !=0 or float(pos[2]['positionAmt']) !=0:
    in_trade_busd.value=1


#condition_usdt(timeframe_usdt,
#                                                    pivot_period_usdt,
#                                                    atr1_usdt,
#                                                    period_usdt,
#                                                    ma_condition_usdt,
#                                                    exchange
#                                                    ,client,
#                                                    coin,
#                                                    time_usdt,
#                                                    in_trade_usdt,
#                                                    in_trade_busd,
#                                                    lock)

notifier_with_photo("data/vadivela-karthikeya.jpg", "SARAVANA BHAVA")
notifier_with_gif("data/engine.gif", "REVVING UP")

p1=multiprocessing.Process(target=condition_usdt,args=[timeframe_usdt,
                                                     pivot_period_usdt,
                                                     atr1_usdt,
                                                     period_usdt,
                                                     ma_condition_usdt,
                                                     exchange
                                                     ,client,
                                                     coin,
                                                     time_usdt,
                                                     in_trade_usdt,
                                                     in_trade_busd,
                                                     lock])

p2=multiprocessing.Process(target=condition_busdt,args=[timeframe_busd,
                                                        pivot_period_busd,
                                                        atr1_busd,
                                                        period_busd,
                                                        ma_condition_busd,
                                                        exchange,
                                                        client,
                                                        coin,
                                                        time_busd,
                                                        in_trade_usdt,
                                                        in_trade_busd,
                                                        lock])    
            

if __name__=='__main__':
    p1.start()
    p2.start()

            