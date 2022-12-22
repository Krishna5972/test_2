from binance.client import Client
import config


client=Client(config.api_key,config.secret_key)

account_info = client.futures_account()

print(account_info['totalMarginBalance'])

