from datetime import datetime, timedelta
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
import pytz
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np


def candle_size(x, coin):
    return abs(((x['close']-x['open'])/x['open'])*100)


def supertrend(coin, df, period, atr_multiplier, pivot_period):

    pivot_period = pivot_period
    trend_atr = atr_multiplier
    trend_period = period

    df['OpenTime'] = df['OpenTime'].apply(
        lambda x: pd.to_datetime(x, unit='ms') if isinstance(x, int) else x)
    df['size'] = df.apply(candle_size, axis=1, coin=coin)

    df['ma_7'] = talib.MA(df['close'], timeperiod=7)
    df['ma_25'] = talib.MA(df['close'], timeperiod=25)
    df['ma_40'] = talib.MA(df['close'], timeperiod=40)
    df['ma_55'] = talib.MA(df['close'], timeperiod=55)
    df['ma_99'] = talib.MA(df['close'], timeperiod=99)
    df['ma_100'] = talib.MA(df['close'], timeperiod=100)
    df['ma_200'] = talib.MA(df['close'], timeperiod=200)

    df['ema_5'] = talib.EMA(df['close'], 5)
    df['ema_20'] = talib.EMA(df['close'], 20)
    df['ema_55'] = talib.EMA(df['close'], 55)
    df['ema_100'] = talib.EMA(df['close'], 100)
    df['ema_200'] = talib.EMA(df['close'], 200)

    df['ema_9'] = talib.EMA(df['close'], 9)

    df['prev_close'] = df['close'].shift(1)
    df['prev_open'] = df['open'].shift(1)

    df['color'] = df.apply(lambda x: 1 if x['close'] >
                           x['open'] else -1, axis=1)

    df['ema_33'] = talib.EMA(df['close'], 33)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9)

    df['slowk'], df['slowd'] = talib.STOCH(
        df['high'], df['low'], df['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    df['pivot_high'] = pivot(df['high'], pivot_period, pivot_period, 'high')
    df['pivot_low'] = pivot(df['low'], pivot_period, pivot_period, 'low')
    df['atr'] = talib.ATR(df['high'], df['low'],
                          df['close'], timeperiod=trend_period)

    df['pivot_high'] = df['pivot_high'].shift(pivot_period)
    df['pivot_low'] = df['pivot_low'].shift(pivot_period)

    center = np.NaN
    lastpp = np.NaN
    centers = [np.NaN]
    for idx, row in df.iterrows():
        ph = row['pivot_high']
        pl = row['pivot_low']

        if ph:
            lastpp = ph
        elif pl:
            lastpp = pl
        else:
            lastpp = np.NaN

        if not math.isnan(lastpp):
            if math.isnan(centers[-1]):
                centers.append(lastpp)
            else:
                center = round(((centers[-1] * 2) + lastpp)/3, 3)
                centers.append(center)
        df.at[idx, 'center'] = center

    df.ffill(axis=0, inplace=True)
    df['up'] = df['center']-(trend_atr*df['atr'])
    df['down'] = df['center']+(trend_atr*df['atr'])

    Tup = [np.NaN]
    Tdown = [np.NaN]
    Trend = [0]
    df['prev_close'] = df['close'].shift(1)
    for idx, row in df.iterrows():
        if row['prev_close'] > Tup[-1]:
            Tup.append(max(row['up'], Tup[-1]))
        else:
            Tup.append(row['up'])

        if row['prev_close'] < Tdown[-1]:
            Tdown.append(min(row['down'], Tdown[-1]))
        else:
            Tdown.append(row['down'])

        if row['close'] > Tdown[-1]:
            df.at[idx, 'in_uptrend'] = True
            Trend.append(True)
        elif row['close'] < Tup[-1]:
            df.at[idx, 'in_uptrend'] = False
            Trend.append(False)
        else:
            if math.isnan(Trend[-1]):
                df.at[idx, 'in_uptrend'] = True
                Trend.append(True)
            else:
                df.at[idx, 'in_uptrend'] = Trend[-1]
                Trend.append(Trend[-1])

    Tup.pop(0)
    Tdown.pop(0)
    df['lower_band'] = Tup
    df['upper_band'] = Tdown
    return df


def pivot(osc, LBL, LBR, highlow):
    left = []
    right = []
    pivots = []
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


def ema_pos(x, col_name):
    if x['close'] > x[col_name]:
        return 1
    else:
        return -1


def close_position(client, coin, signal):
    if signal == 'Buy':
        client.futures_create_order(
            symbol=f'{coin}USDT', side='SELL', type='MARKET', quantity=1000, dualSidePosition=True, positionSide='LONG')
    else:
        client.futures_create_order(
            symbol=f'{coin}USDT', side='BUY', type='MARKET', quantity=1000, dualSidePosition=True, positionSide='SHORT')


def close_position_busd(client, coin, signal):
    if signal == 'Buy':
        client.futures_create_order(
            symbol=f'{coin}BUSD', side='SELL', type='MARKET', quantity=1000, dualSidePosition=True, positionSide='LONG')
    else:
        client.futures_create_order(
            symbol=f'{coin}BUSD', side='BUY', type='MARKET', quantity=1000, dualSidePosition=True, positionSide='SHORT')


@njit
def cal_numba(opens, highs, lows, closes, in_uptrends, profit_perc, sl_perc, upper_bands, lower_bands, colors, rsis, macdhists, slowks, slowds, volumes):
    entries = np.zeros(len(opens))
    signals = np.zeros(len(opens))  # characters  1--> buy  2--->sell
    tps = np.zeros(len(opens))
    trades = np.zeros(len(opens))  # characters   1--->w  0---->L
    close_prices = np.zeros(len(opens))
    time_index = np.zeros(len(opens))
    candle_count = np.zeros(len(opens))
    local_max = np.zeros(len(opens))
    local_min = np.zeros(len(opens))
    upper = np.zeros(len(opens))
    lower = np.zeros(len(opens))
    next_colors = np.zeros(len(opens))
    local_max_bar = np.zeros(len(opens))
    local_min_bar = np.zeros(len(opens))

    next_colors = np.zeros(len(opens))
    local_max_bar = np.zeros(len(opens))
    local_min_bar = np.zeros(len(opens))
    next_close = np.zeros(len(opens))
    indication = 0
    buy_search = 0
    sell_search = 1
    change_index = 0
    local_max_bar_2 = np.zeros(len(opens))
    local_min_bar_2 = np.zeros(len(opens))
    local_max_2 = np.zeros(len(opens))
    local_min_2 = np.zeros(len(opens))

    prev_candle_0_color = np.zeros(len(opens), dtype=np.float64)
    prev_candle_1_color = np.zeros(len(opens), dtype=np.float64)
    prev_candle_2_color = np.zeros(len(opens), dtype=np.float64)
    prev_candle_3_color = np.zeros(len(opens), dtype=np.float64)
    prev_candle_4_color = np.zeros(len(opens), dtype=np.float64)

    prev_candle_0_rsi = np.zeros(len(opens))
    prev_candle_1_rsi = np.zeros(len(opens), dtype=np.float64)
    prev_candle_2_rsi = np.zeros(len(opens), dtype=np.float64)
    prev_candle_3_rsi = np.zeros(len(opens), dtype=np.float64)
    prev_candle_4_rsi = np.zeros(len(opens), dtype=np.float64)

    prev_candle_0_macd = np.zeros(len(opens), dtype=np.float64)
    prev_candle_1_macd = np.zeros(len(opens), dtype=np.float64)
    prev_candle_2_macd = np.zeros(len(opens), dtype=np.float64)
    prev_candle_3_macd = np.zeros(len(opens), dtype=np.float64)
    prev_candle_4_macd = np.zeros(len(opens), dtype=np.float64)

    prev_candle_0_slowk = np.zeros(len(opens), dtype=np.float64)
    prev_candle_1_slowk = np.zeros(len(opens), dtype=np.float64)
    prev_candle_2_slowk = np.zeros(len(opens), dtype=np.float64)
    prev_candle_3_slowk = np.zeros(len(opens), dtype=np.float64)
    prev_candle_4_slowk = np.zeros(len(opens), dtype=np.float64)

    prev_candle_0_slowd = np.zeros(len(opens), dtype=np.float64)
    prev_candle_1_slowd = np.zeros(len(opens), dtype=np.float64)
    prev_candle_2_slowd = np.zeros(len(opens), dtype=np.float64)
    prev_candle_3_slowd = np.zeros(len(opens), dtype=np.float64)
    prev_candle_4_slowd = np.zeros(len(opens), dtype=np.float64)

    prev_candle_0_volume = np.zeros(len(opens))
    prev_candle_1_volume = np.zeros(len(opens))
    prev_candle_2_volume = np.zeros(len(opens))
    prev_candle_3_volume = np.zeros(len(opens))
    prev_candle_4_volume = np.zeros(len(opens))

    indication = 0
    buy_search = 0
    sell_search = 1
    change_index = 0
    i = -1
    while (i < len(opens)):
        i = i+1

        if (indication == 0) & (sell_search == 1) & (buy_search == 0) & (change_index == i):

            sell_search = 0
            flag = 0
            trade = 5
            while (indication == 0):

                entry = closes[i]
                tp = entry - (entry * profit_perc)
                sl = entry + (entry * sl_perc)

                upper[i] = upper_bands[i]
                lower[i] = lower_bands[i]

                entries[i] = entry
                tps[i] = tp
                signals[i] = 2
                local_max[i] = highs[i+1]
                local_min[i] = lows[i+1]
                local_max_2[i] = highs[i+2]
                local_min_2[i] = lows[i+2]
                next_colors[i] = colors[i+1]
                next_close[i] = closes[i+1]

                prev_candle_0_color[i] = colors[i]
                prev_candle_1_color[i] = colors[i-1]
                prev_candle_2_color[i] = colors[i-2]
                prev_candle_3_color[i] = colors[i-3]
                prev_candle_4_color[i] = colors[i-4]

                prev_candle_0_rsi[i] = rsis[i]
                prev_candle_1_rsi[i] = rsis[i-1]
                prev_candle_2_rsi[i] = rsis[i-2]
                prev_candle_3_rsi[i] = rsis[i-3]
                prev_candle_4_rsi[i] = rsis[i-4]

                prev_candle_0_macd[i] = macdhists[i]
                prev_candle_1_macd[i] = macdhists[i-1]
                prev_candle_2_macd[i] = macdhists[i-2]
                prev_candle_3_macd[i] = macdhists[i-3]
                prev_candle_4_macd[i] = macdhists[i-4]

                prev_candle_0_slowk[i] = slowks[i]
                prev_candle_1_slowk[i] = slowks[i-1]
                prev_candle_2_slowk[i] = slowks[i-2]
                prev_candle_3_slowk[i] = slowks[i-3]
                prev_candle_4_slowk[i] = slowks[i-4]

                prev_candle_0_slowd[i] = slowds[i]
                prev_candle_1_slowd[i] = slowds[i-1]
                prev_candle_2_slowd[i] = slowds[i-2]
                prev_candle_3_slowd[i] = slowds[i-3]
                prev_candle_4_slowd[i] = slowds[i-4]

                prev_candle_0_volume[i] = volumes[i]
                prev_candle_1_volume[i] = volumes[i-1]
                prev_candle_2_volume[i] = volumes[i-2]
                prev_candle_3_volume[i] = volumes[i-3]
                prev_candle_4_volume[i] = volumes[i-4]

                for j in range(i+1, len(opens)):
                    candle_count[i] = candle_count[i]+1
                    if candle_count[i] > 2:
                        if lows[j] < local_min_2[i]:
                            local_min_2[i] = lows[j]
                            local_min_bar_2[i] = candle_count[i]
                        if highs[j] > local_max_2[i]:
                            local_max_2[i] = highs[j]
                            local_max_bar_2[i] = candle_count[i]

                    if lows[j] < local_min[i]:
                        local_min[i] = lows[j]
                        local_min_bar[i] = candle_count[i]
                    if highs[j] > local_max[i]:
                        local_max[i] = highs[j]
                        local_max_bar[i] = candle_count[i]

                    if lows[j] < tp and flag == 0:

                        trades[i] = 1
                        close_prices[i] = tp
                        time_index[i] = i

                        indication = 1
                        buy_search = 1
                        flag = 1

                    elif (highs[j] > sl and flag == 0) or (in_uptrends[j] == 'True'):
                        if highs[j] > sl and flag == 0:
                            trades[i] = 0
                            close_prices[i] = sl
                            time_index[i] = i

                            indication = 1
                            buy_search = 1
                            flag = 1

                        if in_uptrends[j] == 'True':

                            if trades[i] == 1:
                                change_index = j
                            elif trades[i] == 0 and flag == 1:
                                change_index = j
                            else:
                                trades[i] = 0
                                close_prices[i] = closes[j]
                                time_index[i] = i
                                change_index = j

                            indication = 1
                            buy_search = 1
                            break
                    else:
                        pass
                break
        elif (indication == 1) & (sell_search == 0) & (buy_search == 1) & (change_index == i):

            buy_search = 0
            flag = 0

            while (indication == 1):

                entry = closes[i]
                tp = entry + (entry * profit_perc)
                sl = entry - (entry * sl_perc)

                upper[i] = upper_bands[i]
                lower[i] = lower_bands[i]

                entries[i] = entry
                tps[i] = tp
                signals[i] = 1
                local_max[i] = highs[i+1]
                local_min[i] = lows[i+1]
                next_colors[i] = colors[i+1]
                local_max_2[i] = highs[i+2]
                local_min_2[i] = lows[i+2]

                prev_candle_0_color[i] = colors[i]
                prev_candle_1_color[i] = colors[i-1]
                prev_candle_2_color[i] = colors[i-2]
                prev_candle_3_color[i] = colors[i-3]
                prev_candle_4_color[i] = colors[i-4]

                prev_candle_0_rsi[i] = rsis[i]
                prev_candle_1_rsi[i] = rsis[i-1]
                prev_candle_2_rsi[i] = rsis[i-2]
                prev_candle_3_rsi[i] = rsis[i-3]
                prev_candle_4_rsi[i] = rsis[i-4]

                prev_candle_0_macd[i] = macdhists[i]
                prev_candle_1_macd[i] = macdhists[i-1]
                prev_candle_2_macd[i] = macdhists[i-2]
                prev_candle_3_macd[i] = macdhists[i-3]
                prev_candle_4_macd[i] = macdhists[i-4]

                prev_candle_0_slowk[i] = slowks[i]
                prev_candle_1_slowk[i] = slowks[i-1]
                prev_candle_2_slowk[i] = slowks[i-2]
                prev_candle_3_slowk[i] = slowks[i-3]
                prev_candle_4_slowk[i] = slowks[i-4]

                prev_candle_0_slowd[i] = slowds[i]
                prev_candle_1_slowd[i] = slowds[i-1]
                prev_candle_2_slowd[i] = slowds[i-2]
                prev_candle_3_slowd[i] = slowds[i-3]
                prev_candle_4_slowd[i] = slowds[i-4]

                prev_candle_0_volume[i] = volumes[i]
                prev_candle_1_volume[i] = volumes[i-1]
                prev_candle_2_volume[i] = volumes[i-2]
                prev_candle_3_volume[i] = volumes[i-3]
                prev_candle_4_volume[i] = volumes[i-4]

                next_close[i] = closes[i+1]

                for j in range(i+1, len(opens)):
                    candle_count[i] = candle_count[i]+1
                    if candle_count[i] > 2:
                        if lows[j] < local_min_2[i]:
                            local_min_2[i] = lows[j]
                            local_min_bar_2[i] = candle_count[i]
                        if highs[j] > local_max_2[i]:
                            local_max_2[i] = highs[j]
                            local_max_bar_2[i] = candle_count[i]
                    if lows[j] < local_min[i]:
                        local_min[i] = lows[j]
                        local_min_bar[i] = candle_count[i]
                    if highs[j] > local_max[i]:
                        local_max[i] = highs[j]
                        local_max_bar[i] = candle_count[i]

                    if highs[j] > tp and flag == 0:
                        trades[i] = 1
                        sell_search = 1
                        close_prices[i] = tp
                        time_index[i] = i

                        flag = 1
                        indication = 0
                    elif (lows[j] < sl and flag == 0) or (in_uptrends[j] == 'False'):
                        if lows[j] < sl and flag == 0:

                            trades[i] = 0
                            close_prices[i] = sl
                            time_index[i] = i
                            indication = 0
                            sell_search = 1
                            flag = 1

                        if in_uptrends[j] == 'False':

                            if trades[i] == 1:
                                change_index = j
                            elif trades[i] == 0 and flag == 1:
                                change_index = j
                            else:
                                trades[i] = 0
                                close_prices[i] = closes[j]
                                time_index[i] = i
                                change_index = j

                            indication = 0
                            sell_search = 1
                            break

                    else:
                        pass
                break
        else:
            continue

    return entries, signals, tps, trades, close_prices, time_index, candle_count, local_max, local_min, local_max_bar, local_min_bar, upper, lower, next_colors, next_close, \
        prev_candle_0_color, prev_candle_1_color, prev_candle_2_color, prev_candle_3_color, prev_candle_4_color, \
        prev_candle_0_rsi, prev_candle_1_rsi, prev_candle_2_rsi, prev_candle_3_rsi, prev_candle_4_rsi, \
        prev_candle_0_macd, prev_candle_1_macd, prev_candle_2_macd, prev_candle_3_macd, prev_candle_4_macd, \
        prev_candle_0_slowk, prev_candle_1_slowk, prev_candle_2_slowk, prev_candle_3_slowk, prev_candle_4_slowk, \
        prev_candle_0_slowd, prev_candle_1_slowd, prev_candle_2_slowd, prev_candle_3_slowd, prev_candle_4_slowd, \
        prev_candle_0_volume, prev_candle_1_volume, prev_candle_2_volume, prev_candle_3_volume, prev_candle_4_volume, \
        local_max_bar_2, local_min_bar_2, local_max_2, local_min_2


def create_signal_df(super_df, df, coin, timeframe, atr1, period, profit, sl):
    opens = super_df['open'].to_numpy(dtype='float64')
    highs = super_df['high'].to_numpy(dtype='float64')
    lows = super_df['low'].to_numpy(dtype='float64')
    closes = super_df['close'].to_numpy(dtype='float64')
    in_uptrends = super_df['in_uptrend'].to_numpy(dtype='U5')
    upper_bands = super_df['upper_band'].to_numpy(dtype='float64')
    lower_bands = super_df['lower_band'].to_numpy(dtype='float64')
    colors = super_df['color'].to_numpy(dtype='float64')

    super_df['rsi'] = round(super_df['rsi'], 2)
    rsis = super_df['rsi'].to_numpy(dtype='float64')

    macdhists = super_df['macdhist'].to_numpy(dtype=np.float64)
    slowks = super_df['slowk'].to_numpy(dtype=np.float64)

    slowds = super_df['slowd'].to_numpy(dtype=np.float64)
    volumes = super_df['volume'].to_numpy(dtype=np.float64)

    entries, signals, tps, trades, close_prices, time_index, candle_count, local_max, local_min, local_max_bar, local_min_bar, upper, lower, colors, next_close, \
        prev_candle_0_color, prev_candle_1_color, prev_candle_2_color, prev_candle_3_color, prev_candle_4_color, \
        prev_candle_0_rsi, prev_candle_1_rsi, prev_candle_2_rsi, prev_candle_3_rsi, prev_candle_4_rsi, \
        prev_candle_0_macd, prev_candle_1_macd, prev_candle_2_macd, prev_candle_3_macd, prev_candle_4_macd, \
        prev_candle_0_slowk, prev_candle_1_slowk, prev_candle_2_slowk, prev_candle_3_slowk, prev_candle_4_slowk, \
        prev_candle_0_slowd, prev_candle_1_slowd, prev_candle_2_slowd, prev_candle_3_slowd, prev_candle_4_slowd, \
        prev_candle_0_volume, prev_candle_1_volume, prev_candle_2_volume, prev_candle_3_volume, prev_candle_4_volume, \
        local_max_bar_2, local_min_bar_2, local_max_2, local_min_2 = cal_numba(
            opens, highs, lows, closes, in_uptrends, profit, sl, upper_bands, lower_bands, colors, rsis, macdhists, slowks, slowds, volumes)

    trade_df = pd.DataFrame({'signal': signals, 'entry': entries, 'tp': tps, 'trade': trades, 'close_price': close_prices, 'candle_count': candle_count,
                             'local_max': local_max, 'local_min': local_min, 'local_max_bar': local_max_bar, 'local_min_bar': local_min_bar,
                             'upper_band': upper, 'lower_band': lower, 'next_color': colors, 'next_close': next_close,
                             'prev_candle_0_color': prev_candle_0_color, 'prev_candle_1_color': prev_candle_1_color, 'prev_candle_2_color': prev_candle_2_color, 'prev_candle_3_color': prev_candle_3_color, 'prev_candle_4_color': prev_candle_4_color,
                             'prev_candle_0_rsi': prev_candle_0_rsi, 'prev_candle_1_rsi': prev_candle_1_rsi, 'prev_candle_2_rsi': prev_candle_2_rsi, 'prev_candle_3_rsi': prev_candle_3_rsi, 'prev_candle_4_rsi': prev_candle_4_rsi,
                             'prev_candle_0_macd': prev_candle_0_macd, 'prev_candle_1_macd': prev_candle_1_macd, 'prev_candle_2_macd': prev_candle_2_macd, 'prev_candle_3_macd': prev_candle_3_macd, 'prev_candle_4_macd': prev_candle_4_macd,
                             'prev_candle_0_slowk': prev_candle_0_slowk, 'prev_candle_1_slowk': prev_candle_1_slowk, 'prev_candle_2_slowk': prev_candle_2_slowk, 'prev_candle_3_slowk': prev_candle_3_slowk, 'prev_candle_4_slowk': prev_candle_4_slowk,
                            'prev_candle_0_slowd': prev_candle_0_slowd, 'prev_candle_1_slowd': prev_candle_1_slowd, 'prev_candle_2_slowd': prev_candle_2_slowd, 'prev_candle_3_slowd': prev_candle_3_slowd, 'prev_candle_4_slowd': prev_candle_4_slowd,
                             'prev_candle_0_volume': prev_candle_0_volume, 'prev_candle_1_volume': prev_candle_1_volume, 'prev_candle_2_volume': prev_candle_2_volume, 'prev_candle_3_volume': prev_candle_3_volume, 'prev_candle_4_volume': prev_candle_4_volume,
                             'local_max_bar_2': local_max_bar_2, 'local_min_bar_2': local_min_bar_2, 'local_max_2': local_max_2, 'local_min_2': local_min_2




                             })
    # before_drop=trade_df.shape[0]
    # print(f'Number of columns before drop : {before_drop}')

    trade_df_index = trade_df[trade_df['entry'] != 0]

    indexes = trade_df_index.index.to_list()

    df = super_df

    print(df.shape[0])
    print(trade_df.shape[0])
    print(super_df.shape[0])
    for i in indexes:
        try:
            trade_df.at[i, 'TradeOpenTime'] = df[df.index ==
                                                 i+1]['OpenTime'][(i+1)]
        except KeyError:
            trade_df.at[i, 'TradeOpenTime'] = (
                df[df.index == i]['OpenTime'][(i)])
    for i in indexes:
        try:
            trade_df.at[i, 'signalTime'] = df[df.index == i]['OpenTime'][(i)]
        except KeyError:
            trade_df.at[i, 'signalTime'] = (df[df.index == i]['OpenTime'][(i)])

    trade_df['signal'] = trade_df['signal'].apply(signal_decoding)

    trade_df.dropna(inplace=True)

    entries = trade_df['entry'].to_numpy(dtype='float64')
    closes = trade_df['close_price'].to_numpy(dtype='float64')
    # trades=trade_df['trade'].to_numpy(dtype='U1')
    signals = trade_df['signal'].to_numpy(dtype='U5')
    outputs = np.zeros(len(entries))

    percentages = df_perc_cal(entries, closes, signals, outputs)
    trade_df['percentage'] = percentages.tolist()
    trade_df['trade'] = trade_df['percentage'].apply(trade_decoding)
    # after_drop=trade_df.shape[0]
    # print(f'Number of columns after drop : {after_drop}')
    trade_df = trade_df.reset_index(drop=True)
    if (trade_df['percentage'][trade_df.shape[0]-1] == -1) | (trade_df['percentage'][trade_df.shape[0]-1] == 1):
        trade_df = trade_df[:-1]
    else:
        pass
    trade_df['signalTime'] = pd.to_datetime(trade_df['signalTime'])
    super_df['OpenTime'] = pd.to_datetime(super_df['OpenTime'])

    trade_df = pd.merge(trade_df, super_df, how='left', left_on=[
                        'signalTime'], right_on=['OpenTime'])

    trade_df = trade_df[['signal',
                         'entry',
                        'tp',
                         'trade',
                         'close_price',
                         'TradeOpenTime',
                         'percentage',
                         'OpenTime',

                         'size', 'ma_7', 'ma_25', 'ma_99',
                         'ema_9',
                         'ma_40', 'ma_55', 'ema_20', 'ema_5', 'ema_55', 'ma_100', 'ma_200', 'ema_100', 'ema_200',
                         'ema_33',
                         'rsi',
                         'macd',
                         'macdsignal',
                         'macdhist',
                         'slowk',
                         'slowd',
                         'candle_count',
                         'local_max', 'local_min',
                         'local_max_bar', 'local_min_bar', 'next_color', 'next_close',
                         'prev_candle_0_color', 'prev_candle_1_color', 'prev_candle_2_color', 'prev_candle_3_color', 'prev_candle_4_color',
                         'prev_candle_0_rsi', 'prev_candle_1_rsi', 'prev_candle_2_rsi', 'prev_candle_3_rsi', 'prev_candle_4_rsi',
                         'prev_candle_0_macd', 'prev_candle_1_macd', 'prev_candle_2_macd', 'prev_candle_3_macd', 'prev_candle_4_macd',
                         'prev_candle_0_slowk', 'prev_candle_1_slowk', 'prev_candle_2_slowk', 'prev_candle_3_slowk', 'prev_candle_4_slowk',
                         'prev_candle_0_slowd', 'prev_candle_1_slowd', 'prev_candle_2_slowd', 'prev_candle_3_slowd', 'prev_candle_4_slowd',
                         'prev_candle_0_volume', 'prev_candle_1_volume', 'prev_candle_2_volume', 'prev_candle_3_volume', 'prev_candle_4_volume',
                         'local_max_bar_2', 'local_min_bar_2', 'local_max_2', 'local_min_2']]

    trade_df = trade_df.dropna()
    trade_df = trade_df[2:]
    # trade_df.to_csv(f'data/file.csv',index=False,mode='w+')

    return trade_df


telegram_auth_token = '5515290544:AAG9T15VaY6BIxX2VYX8x2qr34aC-zVEYMo'
telegram_group_id = 'notifier2_scanner_bot_link'


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
def df_perc_cal(entries, closes, signals, percentages):
    for i in range(0, len(entries)):
        if signals[i] == 'Buy':
            percentages[i] = (closes[i]-entries[i])/entries[i]
        else:
            percentages[i] = -(closes[i]-entries[i])/entries[i]
    return percentages


def notifier(message, tries=0):
    telegram_api_url = f'https://api.telegram.org/bot{telegram_auth_token}/sendMessage?chat_id=@{telegram_group_id}&text={message}'
    # https://api.telegram.org/bot5515290544:AAG9T15VaY6BIxX2VYX8x2qr34aC-zVEYMo/sendMessage?chat_id=@notifier2_scanner_bot_link&text=hii
    tel_resp = requests.get(telegram_api_url)
    if tel_resp.status_code == 200:
        pass
    else:
        while (tries < 25):
            print(f'Telegram notifier problem retrying {tries}')
            tries += 1
            time.sleep(0.5)
            notifier(message, tries)


def notifier_with_photo(file_path, caption, tries=0):
    telegram_api_url = f'https://api.telegram.org/bot{telegram_auth_token}/sendPhoto'
    files = {'photo': open(file_path, 'rb')}
    data = {'chat_id': f'@{telegram_group_id}', 'caption': caption}
    tel_resp = requests.post(telegram_api_url, files=files, data=data)

    if tel_resp.status_code == 200:
        pass
    else:
        while tries < 25:
            print(f'Telegram notifier problem retrying {tries}')
            tries += 1
            time.sleep(0.5)
            notifier_with_photo(file_path, caption, tries)


def notifier_with_gif(file_path, caption, tries=0):
    telegram_api_url = f'https://api.telegram.org/bot{telegram_auth_token}/sendDocument'
    files = {'document': open(file_path, 'rb')}
    data = {'chat_id': f'@{telegram_group_id}', 'caption': caption}
    tel_resp = requests.post(telegram_api_url, files=files, data=data)

    if tel_resp.status_code == 200:
        pass
    else:
        while tries < 25:
            print(f'Telegram notifier problem retrying {tries}')
            tries += 1
            time.sleep(0.5)
            notifier_with_gif(file_path, caption, tries)


def condition_usdt(timeframe, pivot_period, atr1, period, ma_condition, exchange, client, coin, sleep_time, in_trade_usdt, in_trade_busd, lock):
    print(f'timeframe : {timeframe}')
    notifier(f'Starting USDT function,SARAVANA BHAVA')
    sayings_and_gifs = [
        ("data/1.gif", "Bigger the patience, bigger the reward."),
        ("data/2.gif", "The market is a device for transferring money from the impatient to the patient."),
        ("data/3.gif", "Trading is a marathon, not a sprint; stamina beats speed in the long run."),
        ("data/4.gif", "Emotions are a trader's worst enemy. Practice patience, stay disciplined, and keep a level head."),
        ("data/5.gif", "Profit comes to those who wait. The market will always present new opportunities."),
        ("data/6.gif", "Success in the market is not about brilliance, but resilience. Stay patient, stay focused."),
        ("data/7.gif", "In trading, money is made in waiting, not in the transaction."),
        ("data/8.gif", "Bulls make money, bears make money, pigs get slaughtered never let greed take over your trading."),
        ("data/9.gif", "Those who rush to riches will be met with poverty at the finish line."),
        ("data/10.gif", "Beware of jumping into trades for quick money, the pursuit of easy gains can lead to heavy losses."),
        ("data/11.gif", "In the face of uncertainty, choose patience over greed. It's better to be safe than sorry."),
        ("data/12.gif", "In trading, patience is the virtue that separates the successful from the impulsive."),
        ("data/13.gif", "The patient trader understands that success is not about making trades every day but about making the right trades when the opportunity arises."),
        ("data/14.gif", "In trading, impatience can lead to emotional decisions, while patience fosters a rational and disciplined approach.Lets the bot work 010101...."),
        ("data/15.gif", "In the pursuit of financial success, patience is not just a virtue, but a strategy. The market rewards those who can wait."),
        ("data/16.gif", "Rushing is the enemy of profit. In the stock market, the tortoise often beats the hare."),
        ("data/5.gif", "If you are here for quick money, then market will definitely kick you out first.")
    ]
    restart = 0
    risk = 0.028
    neutral_risk = risk
    lower_risk = 0.01  # initial_risk/2
    higher_risk = risk * 1.5
    previous_trade_win_divide = 3
    while (True):
        if restart == 1:
            notifier('USDT Restarted succesfully')
            restart = 0
        try:
            ws = websocket.WebSocket()
            ws.connect(
                f"wss://fstream.binance.com/ws/{str.lower(coin)}usdt@kline_{timeframe}")
            notifier(f'Started USDT function : {timeframe}')
            ws.settimeout(15)
            bars = exchange.fetch_ohlcv(
                f'{coin}/USDT', timeframe=timeframe, limit=1998)
            df = pd.DataFrame(
                bars[:-1], columns=['OpenTime', 'open', 'high', 'low', 'close', 'volume'])
            # df.drop(['OpenTime'],axis=1,inplace=True)
            x_str = str(df['close'].iloc[-1])
            decimal_index = x_str.find('.')
            round_price = len(x_str) - decimal_index - 1
            exchange_info = client.futures_exchange_info()

            for symbol in exchange_info['symbols']:
                if symbol['symbol'] == f"{coin}USDT":
                    round_quantity = symbol['quantityPrecision']
                    break
            notifier(round_quantity)
            indicator = 0
            weight_reduce = 0
            while True:
                result = ws.recv()
                data = json.loads(result)
                if data['k']['x'] == True:
                    candle = data['k']
                    candle_data = [candle['t'], candle['o'],
                                   candle['h'], candle['l'], candle['c'], candle['v']]
                    temp_df = pd.DataFrame([candle_data], columns=[
                                           'OpenTime', 'open', 'high', 'low', 'close', 'volume'])
                    df = pd.concat([df, temp_df])
                    df = df[2:]
                    df = df.reset_index(drop=True)
                    df[['open', 'high', 'low', 'close', 'volume']] = df[[
                        'open', 'high', 'low', 'close', 'volume']].astype(float)
                    super_df = supertrend(coin, df, period, atr1, pivot_period)
                    # print(df[-30:][['OpenTime','open','close','in_uptrend']])
                    super_df[f'{ma_condition}_pos'] = super_df[[ma_condition, 'close']].apply(
                        ema_pos, col_name=ma_condition, axis=1)
                    ma_pos = super_df.iloc[-1][f'{ma_condition}_pos']
                    if super_df.iloc[-1]['in_uptrend'] != super_df.iloc[-2]['in_uptrend']:
                        acc_balance = round(float(client.futures_account()[
                                            'totalCrossWalletBalance']), 2)
                        weekday = pd.to_datetime(
                            super_df.iloc[-1]['OpenTime']).weekday()
                        canTrade = not (weekday == 5 or weekday == 6)
                        print(f'USDT : Can Trade? : {canTrade}')
                        if not canTrade:
                            try:
                                # close open position if any
                                close_position(client, coin, 'Sell')
                                in_trade_usdt.value = 0
                                notifier(f'USDT : Position Closed {timeframe}')
                            except Exception as err:
                                try:
                                    close_position(client, coin, 'Buy')
                                    notifier(
                                        f'USDT : Position Closed {timeframe}')
                                    in_trade_usdt.value = 0
                                except Exception as e:
                                    notifier(
                                        f'USDT : No Open Position to Close {timeframe}')
                            week_over_week(client, coin, acc_balance)
                            notifier(
                                " USDT: Weekly Report is generated and sent via mail")

                            if weekday == 5:
                                notifier(
                                    " USDT:Not taking the trade as it is Saturday")
                            else:
                                notifier(
                                    "USDT:Not taking the trade as it is Sunday")
                            continue

                        trade_df = create_signal_df(
                            super_df, df, coin, timeframe, atr1, period, 100, 100)

                        trade_df['ema_signal'] = trade_df.apply(
                            lambda x: 1 if x['entry'] > x[ma_condition] else -1, axis=1)
                        trade_df['pos_signal'] = trade_df.apply(lambda x: 1 if x['signal'] == 'Buy' and x['ema_signal'] == 1 else (
                            1 if x['signal'] == 'Sell' and x['ema_signal'] == -1 else 0), axis=1)
                        trade_df = trade_df[trade_df['pos_signal'] == 1]

                        trade_df['weekday'] = trade_df['TradeOpenTime'].dt.weekday
                        trade_df = trade_df[(trade_df['weekday'] != 5) & (
                            trade_df['weekday'] != 6)]

                        # Add 'Year' and 'Week' columns to the DataFrame
                        trade_df['Year'] = trade_df['TradeOpenTime'].dt.isocalendar(
                        ).year
                        trade_df['Week'] = trade_df['TradeOpenTime'].dt.isocalendar(
                        ).week

                        # Group by the 'Year' and 'Week' columns and sum the 'percentage' column
                        df_weekly = trade_df.groupby(['Year', 'Week'])[
                            'percentage'].sum().reset_index()
                        current_week = pd.to_datetime(
                            datetime.now()).isocalendar()[1]
                        current_year = pd.to_datetime(
                            datetime.now()).isocalendar()[0]
                        try:
                            previousWeekPercentage = df_weekly[(df_weekly['Week'] == (current_week-1)) & (df_weekly['Year'] == current_year)]['percentage'].values[0]
                        except Exception as week:
                            notifier(week)
                            previousWeekPercentage = 0

                        notifier(
                            f'USDT : Previous week percentage : {round(previousWeekPercentage,3)}')

                        trade_df['ema_signal'] = trade_df.apply(
                            lambda x: 1 if x['entry'] > x[ma_condition] else -1, axis=1)
                        trade_df['pos_signal'] = trade_df.apply(lambda x: 1 if x['signal'] == 'Buy' and x['ema_signal'] == 1 else (
                            1 if x['signal'] == 'Sell' and x['ema_signal'] == -1 else 0), axis=1)
                        trade_df = trade_df[trade_df['pos_signal'] == 1]

                        trend_open_1 = trade_df.iloc[-1]['signal']
                        price_open_1 = trade_df.iloc[-1]['entry']
                        price_close_1 = trade_df.iloc[-1]['close_price']
                        lastTradePerc = trade_df.iloc[-1]['percentage']
                        lastTradeOutcome = trade_df.iloc[-1]['trade']
                        lastTradeOpenTime = trade_df.iloc[-1]['OpenTime']
                        
                        trade_df['OpenTime'] = pd.to_datetime(trade_df['OpenTime'])
                        trade_df['day'] = trade_df['OpenTime'].dt.day
                        trade_df['month'] = trade_df['OpenTime'].dt.month
                        trade_df['Year'] = trade_df['OpenTime'].dt.year


                        day_trade_perc = (trade_df.groupby(['day', 'month', 'Year'])
                                        .agg({'percentage': 'sum'})
                                        .sort_values(by=['Year', 'month', 'day'])
                                        .reset_index())
                                        
                        last_trade_day = day_trade_perc.iloc[-1].day 

                        if last_trade_day == datetime.utcnow().day:
                            last_trade_day_perc = day_trade_perc.iloc[-2].percentage 
                        else:
                            last_trade_day_perc = day_trade_perc.iloc[-1].percentage 

                        notifier(
                            f'USDT : Previous trade 1 :Opentime : {lastTradeOpenTime} singal :{trend_open_1}, open : {price_open_1} close : {price_close_1} Previous_trade_returns : {round(lastTradePerc,2)} lastTradeOutcome : {lastTradeOutcome}')

                        lower_risk, neutral_risk, higher_risk


                        if previousWeekPercentage <= -0.03:
                            notifier(
                                f'USDT : Increasing the risk as previous week was negative {round(previousWeekPercentage,3)}')
                            risk = higher_risk

                            if last_trade_day_perc > 0:
                                notifier(f'Decreasing the risk as previous day was positive {round(last_trade_day_perc,3)}')
                                risk = lower_risk/2

                            if lastTradePerc > 0:
                                notifier(
                                    f'USDT : Decreasing the risk as previous trade was a win {round(lastTradePerc,3)}')
                                risk = lower_risk/previous_trade_win_divide

                        elif previousWeekPercentage >= 0.05:
                            notifier(
                                f'USDT : Decreasing the risk as previous week was positive {round(previousWeekPercentage,3)}')
                            risk = lower_risk

                            if last_trade_day_perc > 0:
                                notifier(f'Decreasing the risk as previous day was positive {round(last_trade_day_perc,3)}')
                                risk = lower_risk/2

                            if lastTradePerc > 0:
                                notifier(
                                    f'USDT : Decreasing the by huge as previous trade was a win {round(lastTradePerc,3)}')
                                risk = lower_risk/previous_trade_win_divide

                        else:
                            notifier(
                                f'USDT : Neutral risk as previous week was between -0.03 and 0.05 {round(previousWeekPercentage,3)}')
                            risk = neutral_risk

                            if last_trade_day_perc > 0:
                                notifier(f'Decreasing the risk as previous day was positive {round(last_trade_day_perc,3)}')
                                risk = lower_risk/2

                                
                            if lastTradePerc > 0:
                                notifier(
                                    f'USDT : Decreasing the risk as previous trade was a win {round(lastTradePerc,3)}')
                                risk = lower_risk/previous_trade_win_divide

                        try:
                            # close open position if any
                            close_position(client, coin, 'Sell')
                            in_trade_usdt.value = 0
                            notifier(f'USDT : Position Closed {timeframe}')
                        except Exception as err:
                            try:
                                close_position(client, coin, 'Buy')
                                notifier(f'USDT : Position Closed {timeframe}')
                                in_trade_usdt.value = 0
                            except Exception as e:
                                notifier(
                                    f'USDT : No Open Position to Close {timeframe}')

                            print(err)

                        # print(f'scanning USDT {super_df.iloc[-1][f"OpenTime"]} trade found, ma_pos :{super_df.iloc[-1][f"{ma_condition}_pos"]} and uptrend :{super_df.iloc[-1]["in_uptrend"]},bsud_poisiton :{in_trade_busd.value},usdt_position :{in_trade_usdt.value},sleeping for {sleep_time*60} seconds')

                        stake = (acc_balance*0.88)

                        notifier(f'USDT : Allocated stake:{round(stake,2)}')

                        signal = ['Buy' if super_df.iloc[-1]
                                  ['in_uptrend'] == True else 'Sell'][0]
                        entry = super_df.iloc[-1]['close']

                        if signal == 'Buy':
                            sl = super_df.iloc[-1]['lower_band']
                            sl_perc = (entry-sl)/entry
                        else:
                            sl = super_df.iloc[-1]['upper_band']
                            sl_perc = (sl-entry)/entry

                        stake = (stake*risk)/sl_perc
                        quantity = round(stake/entry, round_quantity)

                        rr = 88

                        if signal == 'Buy' and ma_pos == 1:
                            notifier(
                                f'Previous week percentage : {round(previousWeekPercentage,2)} Current risk : {risk}')
                            notifier(
                                f'Risk adjusted stake:{round(stake,2)},entry:{entry},sl_perc: {round(sl_perc,3)}')
                            notifier(
                                f'Trend Changed {signal} and ma condition {ma_condition} is {ma_pos}')
                            notifier(
                                f'USDT : Bought @{entry}, Timeframe : {timeframe} , pivot_period: {pivot_period},atr:{atr1},period : {period},ma :{ma_condition}')

                            # buy order
                            client.futures_create_order(
                                symbol=f'{coin}USDT', side='BUY', type='MARKET', quantity=quantity, dualSidePosition=True, positionSide='LONG')

                            take_profit = entry+((entry-sl)*rr)
                            notifier(
                                f'USDT : TP : {round(take_profit,round_price)}')
                            client.futures_create_order(
                                symbol=f'{coin}USDT',
                                price=round(take_profit, round_price),
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
                            in_trade_usdt.value = 1

                        elif signal == 'Sell' and ma_pos == -1:
                            notifier(
                                f'Previous week percentage : {round(previousWeekPercentage,2)} Current risk : {risk}')
                            notifier(
                                f'Risk adjusted stake:{round(stake,2)},entry:{entry},sl_perc: {round(sl_perc,3)}')
                            notifier(
                                f'Trend Changed {signal} and ma condition {ma_condition} is {ma_pos}')
                            notifier(
                                f'USDT : Sold @{entry},Timeframe : {timeframe} , pivot_period: {pivot_period},atr:{atr1},period : {period},ma :{ma_condition}')

                            # sell order
                            client.futures_create_order(
                                symbol=f'{coin}USDT', side='SELL', type='MARKET', quantity=quantity, dualSidePosition=True, positionSide='SHORT')

                            take_profit = entry-((sl-entry)*rr)
                            notifier(
                                f'USDT : TP : {round(take_profit,round_price)}')
                            if take_profit < 0:
                                take_profit = entry/2
                            client.futures_create_order(
                                symbol=f'{coin}USDT',
                                price=round(take_profit, round_price),
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
                            in_trade_usdt.value = 1

                        else:
                            notifier(f'Not taking the trade')

                    else:
                        # print(f'Scanning USDT {super_df.iloc[-1][f"OpenTime"]} trade not found, ma_pos :{super_df.iloc[-1][f"{ma_condition}_pos"]} and uptrend :{super_df.iloc[-1]["in_uptrend"]}, bsud_poisiton :{in_trade_busd.value},usdt_position :{in_trade_usdt.value}')
                        # print(f'ma : {super_df.iloc[-1][ma_condition]},close :{super_df.iloc[-1]["close"]},ma_pos :{super_df.iloc[-1][f"{ma_condition}_pos"]}')
                        notifier(f'USDT : {timeframe} candle closed : {coin}')

                        if in_trade_usdt.value == 1 and weight_reduce >= 1:
                            weight_reduce = 0
                            open_orders = client.futures_get_open_orders(
                                symbol=f'{coin}USDT')
                            if len(open_orders) == 0:
                                in_trade_usdt.value = 0
                                notifier('USDT Pos closed in profit')

                        if indicator > 5:
                            indicator = 0  # notification every 60 minutes
                            information = client.futures_account()
                            totalUnrealizedProfit = round(
                                float(information['totalUnrealizedProfit']), 2)
                            bal = round(
                                float(information['totalCrossWalletBalance']), 2)
                            if bal > 320:  # Month initial
                                bal_pos = 'Profit'
                            else:
                                bal_pos = 'Loss'

                            if totalUnrealizedProfit > 0:
                                profit_pos = 'Green'
                            elif totalUnrealizedProfit == 0:
                                profit_pos = 'Neutral'
                            else:
                                profit_pos = 'Red'

                            notifier(
                                f'SARAVANA BHAVA ! Running... ,USDT POS:{in_trade_usdt.value} , BUSD POS: {in_trade_busd.value},Bal :{bal_pos},PNL:{profit_pos}')

                            makeSense(sayings_and_gifs)

                        weight_reduce += 1
                        indicator += 1

        except Exception as err:
            notifier(err)
            notifier(f'Restarting USDT function : {coin}')
            print(err)
            restart = 1
            ws.close()
            tb = traceback.extract_tb(sys.exc_info()[2])
            filename, line, func, text = tb[-1]
            print(f'An error occurred on line USDT {line}: {err}')
            print("Exception occurred usdt:\n", traceback.format_exc())
            time.sleep(10)


def makeSense(sayings_and_gifs):
    # Randomly select a gif and saying
    gif, saying = random.choice(sayings_and_gifs)

    # Use the selected gif and saying
    notifier_with_gif(gif, saying)


def condition_busdt(timeframe, pivot_period, atr1, period, ma_condition, exchange, client, coin, sleep_time, in_trade_usdt, in_trade_busd, lock):
    notifier(f'Starting BUSD function,SARAVANA BHAVA')
    print(f'timeframe : {timeframe}')
    restart = 0

    while (True):
        if restart == 1:
            notifier('BUSD Restarted succesfully')
            restart = 0
        try:
            ws = websocket.WebSocket()
            ws.connect(
                f"wss://fstream.binance.com/ws/{str.lower(coin)}usdt@kline_{timeframe}")
            ws.settimeout(15)
            notifier(f'Started BUSD function : {timeframe}')
            risk = 0.01
            bars = exchange.fetch_ohlcv(
                f'{coin}/USDT', timeframe=timeframe, limit=1998)
            df = pd.DataFrame(
                bars[:-1], columns=['OpenTime', 'open', 'high', 'low', 'close', 'volume'])
            # df.drop(['OpenTime'],axis=1,inplace=True)
            x_str = str(df['close'].iloc[-1])
            decimal_index = x_str.find('.')
            round_price = len(x_str) - decimal_index - 1
            exchange_info = client.futures_exchange_info()
            notifier(f'from bsud {coin}')
            print(coin)
            for symbol in exchange_info['symbols']:
                if symbol['symbol'] == f"{coin}BUSD":
                    round_quantity = symbol['quantityPrecision']
                    break
                elif symbol['symbol'] == f"{coin}USDT":
                    round_quantity = symbol['quantityPrecision']
                    break
            notifier(f'BUSD : Round Quantity :{round_quantity} ')
            while True:
                result = ws.recv()
                data = json.loads(result)
                if data['k']['x'] == True:
                    candle = data['k']
                    candle_data = [candle['t'], candle['o'],
                                   candle['h'], candle['l'], candle['c'], candle['v']]
                    temp_df = pd.DataFrame([candle_data], columns=[
                                           'OpenTime', 'open', 'high', 'low', 'close', 'volume'])
                    df = pd.concat([df, temp_df])
                    df = df[2:]
                    df = df.reset_index(drop=True)
                    df[['open', 'high', 'low', 'close', 'volume']] = df[[
                        'open', 'high', 'low', 'close', 'volume']].astype(float)
                    super_df = supertrend(coin, df, period, atr1, pivot_period)
                    super_df[f'{ma_condition}_pos'] = super_df[[ma_condition, 'close']].apply(
                        ema_pos, col_name=ma_condition, axis=1)
                    ma_pos = super_df.iloc[-1][f'{ma_condition}_pos']
                    super_df['condition'] = 0
                    print(f'BUSD : {super_df.iloc[-1]["OpenTime"]}')
                    print(
                        f'BUSD : Weekday : {pd.to_datetime(super_df.iloc[-1]["OpenTime"]).weekday()}')

                    if super_df.iloc[-1]['in_uptrend'] != super_df.iloc[-2]['in_uptrend']:

                        weekday = pd.to_datetime(
                            super_df.iloc[-1]['OpenTime']).weekday()
                        canTrade = not (weekday == 5 or weekday == 6)
                        print(f'BUSD : Can Trade? : {canTrade}')
                        if not canTrade:
                            try:
                                # close open position if any
                                close_position_busd(client, coin, 'Sell')
                                notifier(f'BUSD : Position Closed {timeframe}')
                                in_trade_busd.value = 0
                            except Exception as err:
                                try:
                                    close_position_busd(client, coin, 'Buy')
                                    notifier(
                                        f'BUSD : Position Closed {timeframe}')
                                    in_trade_busd.value = 0
                                except Exception as e:
                                    notifier(
                                        f'BUSD : No Position to close {timeframe}')

                            if weekday == 5:
                                notifier(
                                    "BUSD : Not taking the trade as it is Saturday")
                            else:
                                notifier(
                                    "BUSD : Not taking the trade as it is Sunday")

                            continue
                        initial_risk = 0.01
                        risk = initial_risk
                        # super_df.to_csv('super_df.csv',mode='w+',index=False)
                       # df.to_csv('df.csv',index=False)
                        trade_df = create_signal_df(
                            super_df, df, coin, timeframe, atr1, period, 100, 100)

                        trade_df['ema_signal'] = trade_df.apply(
                            lambda x: 1 if x['entry'] > x[ma_condition] else -1, axis=1)
                        trade_df['pos_signal'] = trade_df.apply(lambda x: 1 if x['signal'] == 'Buy' and x['ema_signal'] == 1 else (
                            1 if x['signal'] == 'Sell' and x['ema_signal'] == -1 else 0), axis=1)
                        trade_df = trade_df[trade_df['pos_signal'] == 1]
                        trade_df['weekday'] = trade_df['TradeOpenTime'].dt.weekday
                        trade_df = trade_df[(trade_df['weekday'] != 5) & (
                            trade_df['weekday'] != 6)]
                        trend_open_1 = trade_df.iloc[-1]['signal']
                        price_open_1 = trade_df.iloc[-1]['entry']
                        price_close_1 = trade_df.iloc[-1]['close_price']
                        lastTradePerc = trade_df.iloc[-1]['percentage']
                        lastTradeOutcome = trade_df.iloc[-1]['trade']
                        lastTradeOpenTime = trade_df.iloc[-1]['OpenTime']

                        notifier(
                            f'BUSD : Previous trade 1 :Opentime : {lastTradeOpenTime} singal :{trend_open_1}, open : {price_open_1} close : {price_close_1} previous_trade_returns : {round(lastTradePerc,2)} lastTradeOutcome : {lastTradeOutcome}')

                        trend_open_2 = trade_df.iloc[-2]['signal']  # openprice
                        time_open_2 = trade_df.iloc[-2]['OpenTime']
                        price_open_2 = trade_df.iloc[-2]['entry']
                        lastTradeOpenTime_2 = trade_df.iloc[-2]['OpenTime']

                        price_close_2 = trade_df.iloc[-2]['close_price']
                        lastTradeOutcome_2 = trade_df.iloc[-2]['trade']
                        lastTradePerc_2 = trade_df.iloc[-2]['percentage']

                        notifier(
                            f'BUSD : Previous trade 2 :OpenTime : {lastTradeOpenTime_2} singal :{trend_open_2}, open : {price_open_2} close : {price_close_2} previous_trade_returns : {round(lastTradePerc_2,2)} lastTradeOutcome : {lastTradeOutcome_2}')

                        if lastTradeOutcome == 'W':
                            notifier(
                                'BUSD : Last one was a win reducing the risk')
                            risk = initial_risk/2
                        else:
                            notifier(
                                'BUSD : Last one was a Loss not reducing the risk')

                        if lastTradeOutcome == 'W' and lastTradeOutcome_2 == 'W':
                            notifier(
                                'BUSD : Last two were wins reducing the risk drastically')
                            risk = initial_risk/3
                        else:
                            notifier(
                                'BUSD : One of last two a was win or both L so not reducing the risk drastically')

                        try:
                            # close open position if any
                            close_position_busd(client, coin, 'Sell')
                            notifier(f'BUSD : Position Closed {timeframe}')
                            in_trade_busd.value = 0
                        except Exception as err:
                            try:
                                close_position_busd(client, coin, 'Buy')
                                notifier(f'BUSD : Position Closed {timeframe}')
                                in_trade_busd.value = 0
                            except Exception as e:
                                notifier(
                                    f'BUSD : No Position to close {timeframe}')

                        # print(f'scanning busd {super_df.iloc[-1][f"OpenTime"]} trade found, ma_pos :{super_df.iloc[-1][f"{ma_condition}_pos"]} and uptrend :{super_df.iloc[-1]["in_uptrend"]}, bsud_poisiton :{in_trade_busd.value},usdt_position :{in_trade_usdt.value} , sleeping for {sleep_time*60} seconds')
                        acc_balance = round(float(client.futures_account()[
                                            'totalCrossWalletBalance']), 2)

                        stake = (acc_balance*0.88)

                        notifier(
                            f'BUSD : Allocated stake:{round(stake,2)} Risk : {risk}')

                        signal = ['Buy' if super_df.iloc[-1]
                                  ['in_uptrend'] == True else 'Sell'][0]
                        entry = super_df.iloc[-1]['close']

                        if signal == 'Buy':
                            sl = super_df.iloc[-1]['lower_band']
                            sl_perc = (entry-sl)/entry
                        else:
                            sl = super_df.iloc[-1]['upper_band']
                            sl_perc = (sl-entry)/entry

                        stake = (stake*risk)/sl_perc
                        quantity = round(stake/entry, round_quantity)

                        if signal == 'Buy' and ma_pos == 1:
                            # buy order
                            client.futures_create_order(
                                symbol=f'{coin}BUSD', side='BUY', type='MARKET', quantity=quantity, dualSidePosition=True, positionSide='LONG')
                            notifier(
                                f'BUSD : Trend Changed {signal} and ma condition {ma_condition} is {ma_pos},close : {entry} , ma: {super_df.iloc[-1][ma_condition]}')

                            notifier(
                                f'BUSD : Bought BUSD @{entry} , Timeframe : {timeframe} , pivot_period: {pivot_period},atr:{atr1},period : {period},ma :{ma_condition}')
                            in_trade_busd.value = 1
                            notifier(
                                f'BUSD : Risk adjusted stake:{round(stake,2)},entry:{entry},sl_perc: {round(sl_perc,3)}')

                        elif signal == 'Sell' and ma_pos == -1:

                            # sell order
                            client.futures_create_order(
                                symbol=f'{coin}BUSD', side='SELL', type='MARKET', quantity=quantity, dualSidePosition=True, positionSide='SHORT')
                            notifier(
                                f'BUSD : Trend Changed {signal} and ma condition {ma_condition} is {ma_pos},close : {entry} , ma: {super_df.iloc[-1][ma_condition]}')

                            notifier(
                                f'BUSD : Sold BUSD @{entry},Timeframe : {timeframe} , pivot_period: {pivot_period},atr:{atr1},period : {period},ma :{ma_condition}')
                            in_trade_busd.value = 1
                            notifier(
                                f'BUSD : Risk adjusted stake:{round(stake,3)},entry:{entry},sl_perc: {round(sl_perc,3)}')
                        else:
                            notifier(f'BUSD : Not taking the trade')
                    else:
                        notifier(f'BUSD : {timeframe} candle closed : {coin}')

                    try:
                        now = datetime.utcnow()
                        if now.hour == 23 and now.minute < 29:
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

                            day_over_day_dict = combine_dicts(
                                day_over_day_dict, current_day_dict)

                            with open('data/day_over_day_dict.pkl', 'wb') as file:
                                pickle.dump(day_over_day_dict, file)

                            notifier(f'Daily price captured')

                    except Exception as e:
                        notifier('Error while capturing the price')

        except Exception as e:
            notifier(e)
            notifier(f'BUSD : Restarting BUSD function : {coin}')
            print(e)
            ws.close()
            tb = traceback.extract_tb(sys.exc_info()[2])
            filename, line, func, text = tb[-1]
            print(f'An error occurred on line USDT {line}: {e}')
            print("Exception occurred:\n", traceback.format_exc())
            time.sleep(10)
            restart = 1


def combine_dicts(dict1, dict2):
    dict1.update(dict2)
    return dict1


def day_over_day():
    with open('data/day_over_day_dict.pkl', 'rb') as file:
        day_over_day_dict = pickle.load(file)

    day_over_day_df = pd.DataFrame(
        list(day_over_day_dict.items()), columns=["Date", "Balance"])

    day_over_day_df['Percentage Change'] = round(
        day_over_day_df['Balance'].pct_change() * 100, 3)

    day_over_day_df.dropna(inplace=True)

    day_over_day_df.to_csv('day_over_day_df.csv', mode='w+', index=False)

    plot_day_over_day(day_over_day_df)
    send_mail("daily_change.png")


def plot_day_over_day(df):

    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)

    df['DateLabel'] = df['Date'].dt.strftime('%d-%m')

    # Use a bar plot and use color to differentiate positive and negative values
    bars = ax.bar(df['DateLabel'], df['Percentage Change'], color=[
                  'g' if x >= 0 else 'r' for x in df['Percentage Change']])

    # Rotate x-axis labels for better visibility
    plt.xticks(df['DateLabel'], rotation=90,
               fontsize=12, weight='bold', color='black')

    # Set y-ticks properties
    ax.tick_params(axis='y', colors='black', labelsize=12)

    # Display data labels
    for bar, date in zip(bars, df['Date']):
        yval = bar.get_height()
        if not np.isnan(yval):  # Check if yval is not NaN
            if yval >= 0:
                label_position = yval + 0.01
            else:
                label_position = yval - 0.01
            ax.text(bar.get_x() + bar.get_width()/2., label_position,
                    f"{yval:.2f}%\n{date.strftime('%d-%m')}", ha='center', va='bottom', rotation=0, fontsize=10, weight='bold')

    plt.title("Percentage Change", fontsize=16, weight='bold')
    plt.ylabel("Percentage Change (%)", fontsize=14, weight='bold')
    plt.xlabel("Date", fontsize=14, weight='bold')

    # Find the most common month
    most_common_month = df['Date'].dt.strftime('%B %Y').mode()[0]

    # Display the most common month on the plot
    plt.text(0.99, 0.85, most_common_month, transform=ax.transAxes,
             fontsize=14, weight='bold', ha='right')

    # Adjust layout to ensure labels are not cut off
    fig.tight_layout()

    # Save the plot to disk
    plt.savefig("daily_change.png", bbox_inches='tight')

    plt.show()


def week_over_week(client, coin, acc_balance):
    try:
        week_over_week_df = pd.read_csv('week_over_week_df.csv')
    except Exception as e:
        week_over_week_df = pd.DataFrame(
            columns=['date', 'month', 'income', 'day', 'weekday', 'balance'])

    end_date = datetime.now()
    star_date = datetime.now()-timedelta(days=90)

    end_date = time.mktime(end_date.timetuple())
    star_date = time.mktime(star_date.timetuple())
    end_date = int(end_date)*1000
    star_date = int(star_date)*1000

    data = client.futures_income_history(
        symbol=f'{coin}USDT', startTime=star_date, endTime=end_date, limit=1000)
    data_BUSD = client.futures_income_history(
        symbol=f'{coin}BUSD', startTime=star_date, endTime=end_date, limit=1000)

    data.extend(data_BUSD)

    df = pd.DataFrame(data)
    df['time'] = df['time'].apply(lambda x: datetime.fromtimestamp(x/1000))
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['date'] = df['time'].dt.day
    df['income'] = df['income'].astype('float')

    df_fin = df.groupby(['date', 'month']).agg({'income': 'sum'}).reset_index()
    df_fin['day'] = df_fin[['date', 'month']].apply(lambda x: datetime(
        datetime.now().year, x['month'], x['date'], 0, 0, 0, 0), axis=1)
    df_fin.sort_values(by=['day'], inplace=True)

    df_fin['day'] = pd.to_datetime(df_fin['day'])
    df_weekly = df_fin.resample('W', on='day').agg(
        {'income': 'sum'}).reset_index()

    current_week_dict = {
        df_weekly['day'].iloc[-1]: acc_balance
    }

    with open('data/week_over_week_dict.pkl', 'rb') as file:
        week_over_week_dict = pickle.load(file)

    week_over_week_dict = combine_dicts(week_over_week_dict, current_week_dict)

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

    week_over_week_df = pd.concat([week_over_week_df, df_weekly], axis=0)

    week_over_week_df['day'] = pd.to_datetime(week_over_week_df['day'])

    week_over_week_df['day_duplicates'] = week_over_week_df['day'].dt.date

    week_over_week_df.drop_duplicates(
        subset='day_duplicates', keep='first', inplace=True)

    week_over_week_df.to_csv('week_over_week_df.csv', index=False, mode='w+')

    week_over_week_df = week_over_week_df[week_over_week_df['income'] != 0]

    week_over_week_df['change'] = round(
        (week_over_week_df['income']/week_over_week_df['balance'])*100, 2)

    week_over_week_df[['day', 'change']].to_csv(
        'change.csv', index=False, mode='w+')

    send_mail('change.csv')


def send_mail(filename, subject='SARAVANA BHAVA'):
    from_ = 'gannamanenilakshmi1978@gmail.com'
    to = 'vamsikrishnagannamaneni@gmail.com'

    message = MIMEMultipart()
    message['From'] = from_
    message['To'] = to
    message['Subject'] = subject
    body_email = 'SARAVANA BHAVA !'

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
