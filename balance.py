from binance.client import Client
import config


def create_futures_api_uri(self, path: str) -> str:
        url = self.FUTURES_URL
        print('Updated to v2')
        if self.testnet:
            url = self.FUTURES_TESTNET_URL
        return url + '/' + 'v2' + '/' + path


client=Client(config.api_key,config.secret_key)

client._create_futures_api_uri = create_futures_api_uri.__get__(client)

account_info = client.futures_account()

print(account_info['totalWalletBalance'])

