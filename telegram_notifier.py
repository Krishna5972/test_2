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


if len(sys.argv) > 1:
    coin = sys.argv[1]  # Take coin from command line argument if provided
else:
    coin = 'ETH'


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

usdt_leverage,busd_leverage = 18,18

max_usdt_leverage,max_busd_leverage = get_max_leverage(coin, config.api_key, config.secret_key)

usdt_leverage = min(usdt_leverage, max_usdt_leverage)
busd_leverage = min(busd_leverage, max_busd_leverage)


while(True):
    try:
        client=Client(config.api_key,config.secret_key)

        client.futures_change_leverage(symbol=f'{coin}USDT', leverage=usdt_leverage)
        client.futures_change_leverage(symbol=f'{coin}BUSD', leverage=busd_leverage)
        notifier(f'SARAVANA BHAVA')
        break
    except Exception as e:
        notifier(f'Met with exception {e}, sleeping for 5 minutes and trying again')
        time.sleep(300)


in_trade_usdt=multiprocessing.Value('i',0)
in_trade_busd=multiprocessing.Value('i',0)
watchdog_usdt=multiprocessing.Value('i',2)
watchdog_busd=multiprocessing.Value('i',2)
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

notifier(f'SARAVANA BHAVA ! Running... ,USDT POS:{in_trade_usdt.value} , BUSD POS: {in_trade_busd.value}')

usdt_args = [timeframe_usdt,
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
                                                     lock,watchdog_usdt]


busd_args = [timeframe_busd,
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
                                                        lock,watchdog_busd]





p1=multiprocessing.Process(target=condition_usdt,args=usdt_args)

p2=multiprocessing.Process(target=condition_busdt,args=busd_args)    
            

if __name__=='__main__':
    p1.start()
    p2.start()

    while True:
        print(f'Checking USDT currently {watchdog_usdt.value}')
        if watchdog_usdt.value < 0:
            print('main : USDT Sleeping for 10 seconds and checking again if there is a change if not restarting')
            time.sleep(10)

            # If its watchdog_usdt == 0 again, end p1 and restart p1 again
            if watchdog_usdt.value < 0:
                notifier('main : Restarting USDT process')
                p1.terminate()
                p1.join(10)  # make sure p1 has finished
                
                p1 = multiprocessing.Process(target=condition_usdt,args =usdt_args)
                p1.start()

        print(f'Checking BUSD currently {watchdog_busd.value}')
        if watchdog_busd.value < 0:
            print('main : BUSD Sleeping for 10 seconds and checking again if there is a change if not restarting')
            time.sleep(10)

            # If its in_trade_usdt == 0 again, end p1 and restart p1 again
            if watchdog_busd.value < 0:
                notifier('main : Restarting BUSD process')
                p2.terminate()
                p2.join(10)  # make sure p1 has finished
                
                p2=multiprocessing.Process(target=condition_busdt,args=busd_args)
                p2.start()

                send_mail("daily_change.png",subject="BUSD restarted check for damage")
        
        watchdog_usdt.value -= 1
        watchdog_busd.value -= 1
        time.sleep(299)

            