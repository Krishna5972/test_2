from datetime import datetime
from binance.client import Client

from bot_funtions import *
from binance.client import Client

import config

coin = 'ETH'

client = Client(config.api_key, config.secret_key)

try:
    now = datetime.utcnow()
    if not (now.hour == 23 and now.minute < 29):
        acc_balance = round(float(client.futures_account()[
                            'totalCrossWalletBalance']), 2)

        current_day_dict = {
            now.strftime('%d-%m-%Y'): acc_balance
        }
        try:
            with open('data/day_over_day_dict.pkl', 'rb') as file:
                day_over_day_dict = pickle.load(file)
        except Exception as e:
            day_over_day_dict = {}
            with open('data/day_over_day_dict.pkl', 'wb') as file:
                pickle.dump(day_over_day_dict, file)

        day_over_day_dict = combine_dicts(day_over_day_dict, current_day_dict)

        with open('data/day_over_day_dict.pkl', 'wb') as file:
            pickle.dump(day_over_day_dict, file)

        print(day_over_day_dict)
        notifier(f'Daily price captured')
    else:
        print(f'Different time : {now}')
except Exception as e:
    notifier('Error while capturing the price')


day_over_day()
