from bot_funtions import *
from binance.client import Client

import config

coin = 'ETH'

client = Client(config.api_key, config.secret_key)

acc_balance = round(float(client.futures_account()[
                    'totalCrossWalletBalance']), 2)

week_over_week(client, coin, acc_balance)
