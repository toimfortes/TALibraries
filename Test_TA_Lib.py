# import pandas_datareader.data as web
import pandas as pd
import numpy as np
# import talib as tl
import quandl
# import matplotlib.pyplot as plt
import ta as ta
# import datetime as dt
from keras.models import Sequential
from keras.layers import Dense
# from sklearn.model_selection import StratifiedKFold

start_dt = '2014-03-22'
end_dt = '2017-03-27'
quandl.api_config.read_key = '*********************************'

symbol = 'WIKI/AAPL'
symbol2 = 'WIKI/IBM'
max_holding = 100
print('getting data')
price = quandl.get(symbol, start_date=start_dt, end_date=end_dt)
price = price.dropna()
price_test = quandl.get(symbol2, start_date=end_dt, end_date='2018-03-27')
price_test = price_test.dropna()

print('creating TA training set')
# <editor-fold desc="Create the training set">
X = pd.DataFrame(index=price.index, data={'bollinger_hband_indicator': np.array([np.nan] * price.index.shape[0])})
X['bollinger_hband_indicator'] = ta.bollinger_hband_indicator(price['Adj. Close'], n=20, ndev=2, fillna=True)
X['bollinger_lband_indicator'] = ta.bollinger_lband_indicator(price['Adj. Close'], n=20, ndev=2, fillna=True)
X['ema_indicator_21'] = ta.ema_indicator(price['Adj. Close'], n=21, fillna=True)
X['ema_indicator_50'] = ta.ema_indicator(price['Adj. Close'], n=50, fillna=True)
X['ema_indicator_200'] = ta.ema_indicator(price['Adj. Close'], n=200, fillna=True)
X['acc_dist_index'] = ta.acc_dist_index(price['High'], price['Low'], price['Adj. Close'], price['Volume'])
X['on_balance_volume'] = ta.on_balance_volume(price['Adj. Close'], price['Volume'], fillna=True)
X['chaikin_money_flow'] = ta.chaikin_money_flow(price['High'], price['Low'], price['Adj. Close'], price['Volume'], n=20,
                                                fillna=True)
X['force_index'] = ta.force_index(price['Adj. Close'], price['Volume'], n=2, fillna=True)
X['ease_of_movement'] = ta.ease_of_movement(price['High'], price['Low'], price['Adj. Close'], price['Volume'], n=20,
                                            fillna=True)
X['volume_price_trend'] = ta.volume_price_trend(price['Adj. Close'], price['Volume'], fillna=True)
X['negative_volume_index'] = ta.negative_volume_index(price['Adj. Close'], price['Volume'], fillna=True)
X['average_true_range'] = ta.average_true_range(price['High'], price['Low'], price['Adj. Close'], n=14, fillna=True)
# X['KCU'] = ta.keltner_channel_hband_indicator(price['High'], price['Low'], price['Adj. Close'], n=10, fillna=True)
X['keltner_channel_lband_indicator'] = ta.keltner_channel_lband_indicator(price['High'], price['Low'],
                                                                          price['Adj. Close'], n=10, fillna=True)
X['donchian_channel_hband_indicator'] = ta.donchian_channel_hband_indicator(price['Adj. Close'], n=20, fillna=True)
X['donchian_channel_lband_indicator'] = ta.donchian_channel_lband_indicator(price['Adj. Close'], n=20, fillna=True)
X['macd_signal'] = ta.macd_signal(price['Adj. Close'], n_fast=12, n_slow=26, n_sign=9, fillna=True)
X['adx_pos'] = ta.adx_pos(price['High'], price['Low'], price['Adj. Close'], n=14, fillna=True)
X['adx_neg'] = ta.adx_neg(price['High'], price['Low'], price['Adj. Close'], n=14, fillna=True)
X['vortex_indicator_neg'] = ta.vortex_indicator_neg(price['High'], price['Low'], price['Adj. Close'], n=14, fillna=True)
X['vortex_indicator_pos'] = ta.vortex_indicator_pos(price['High'], price['Low'], price['Adj. Close'], n=14, fillna=True)
X['trix'] = ta.trix(price['Adj. Close'], n=15, fillna=True)
X['mass_index'] = ta.mass_index(price['High'], price['Low'], n=9, n2=25, fillna=True)
X['cci'] = ta.cci(price['High'], price['Low'], price['Adj. Close'], n=20, c=1, fillna=True)
X['dpo'] = ta.dpo(price['Adj. Close'], n=20, fillna=True)
X['kst_sig'] = ta.kst_sig(price['Adj. Close'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9,
                          fillna=True)
X['ichimoku_a'] = ta.ichimoku_a(price['High'], price['Low'], n1=9, n2=26, fillna=True)
X['ichimoku_b'] = ta.ichimoku_b(price['High'], price['Low'], n2=26, n3=52, fillna=True)
X['money_flow_index'] = ta.money_flow_index(price['High'], price['Low'], price['Adj. Close'], price['Volume'], n=14,
                                            fillna=True)
X['rsi'] = ta.rsi(price['Adj. Close'], n=14, fillna=True)
X['tsi'] = ta.tsi(price['Adj. Close'], r=25, s=13, fillna=True)
X['uo'] = ta.uo(price['High'], price['Low'], price['Adj. Close'], s=7, m=14, l=28, ws=4, wm=2, wl=1, fillna=True)
X['stoch_signal'] = ta.stoch_signal(price['High'], price['Low'], price['Adj. Close'], n=14, d_n=3, fillna=True)
X['wr'] = ta.wr(price['High'], price['Low'], price['Adj. Close'], lbp=14, fillna=True)
X['ao'] = ta.ao(price['Low'], price['Adj. Close'], s=5, l=34, fillna=True)
X['daily_return'] = ta.daily_return(price['Adj. Close'], fillna=True)
# </editor-fold>

print('creating TA test set')
# <editor-fold desc="Create the test set">
X_test = pd.DataFrame(index=price_test.index,
                      data={'bollinger_hband_indicator': np.array([np.nan] * price_test.index.shape[0])})
X_test['bollinger_hband_indicator'] = ta.bollinger_hband_indicator(price_test['Adj. Close'], n=20, ndev=2, fillna=True)
X_test['bollinger_lband_indicator'] = ta.bollinger_lband_indicator(price_test['Adj. Close'], n=20, ndev=2, fillna=True)
X_test['ema_indicator_21'] = ta.ema_indicator(price_test['Adj. Close'], n=21, fillna=True)
X_test['ema_indicator_50'] = ta.ema_indicator(price_test['Adj. Close'], n=50, fillna=True)
X_test['ema_indicator_200'] = ta.ema_indicator(price_test['Adj. Close'], n=200, fillna=True)
X_test['acc_dist_index'] = ta.acc_dist_index(price_test['High'], price_test['Low'], price_test['Adj. Close'],
                                             price_test['Volume'])
X_test['on_balance_volume'] = ta.on_balance_volume(price_test['Adj. Close'], price_test['Volume'], fillna=True)
X_test['chaikin_money_flow'] = ta.chaikin_money_flow(price_test['High'], price_test['Low'], price_test['Adj. Close'],
                                                     price_test['Volume'], n=20, fillna=True)
X_test['force_index'] = ta.force_index(price_test['Adj. Close'], price_test['Volume'], n=2, fillna=True)
X_test['ease_of_movement'] = ta.ease_of_movement(price_test['High'], price_test['Low'], price_test['Adj. Close'],
                                                 price_test['Volume'], n=20, fillna=True)
X_test['volume_price_trend'] = ta.volume_price_trend(price_test['Adj. Close'], price_test['Volume'], fillna=True)
X_test['negative_volume_index'] = ta.negative_volume_index(price_test['Adj. Close'], price_test['Volume'], fillna=True)
X_test['average_true_range'] = ta.average_true_range(price_test['High'], price_test['Low'], price_test['Adj. Close'],
                                                     n=14, fillna=True)
# X_test['KCU'] = ta.keltner_channel_hband_indicator(price_test['High'], price_test['Low'], price_test['Adj. Close'], n=10, fillna=True)
X_test['keltner_channel_lband_indicator'] = ta.keltner_channel_lband_indicator(price_test['High'], price_test['Low'],
                                                                               price_test['Adj. Close'], n=10,
                                                                               fillna=True)
X_test['donchian_channel_hband_indicator'] = ta.donchian_channel_hband_indicator(price_test['Adj. Close'], n=20,
                                                                                 fillna=True)
X_test['donchian_channel_lband_indicator'] = ta.donchian_channel_lband_indicator(price_test['Adj. Close'], n=20,
                                                                                 fillna=True)
X_test['macd_signal'] = ta.macd_signal(price_test['Adj. Close'], n_fast=12, n_slow=26, n_sign=9, fillna=True)
X_test['adx_pos'] = ta.adx_pos(price_test['High'], price_test['Low'], price_test['Adj. Close'], n=14, fillna=True)
X_test['adx_neg'] = ta.adx_neg(price_test['High'], price_test['Low'], price_test['Adj. Close'], n=14, fillna=True)
X_test['vortex_indicator_neg'] = ta.vortex_indicator_neg(price_test['High'], price_test['Low'],
                                                         price_test['Adj. Close'], n=14, fillna=True)
X_test['vortex_indicator_pos'] = ta.vortex_indicator_pos(price_test['High'], price_test['Low'],
                                                         price_test['Adj. Close'], n=14, fillna=True)
X_test['trix'] = ta.trix(price_test['Adj. Close'], n=15, fillna=True)
X_test['mass_index'] = ta.mass_index(price_test['High'], price_test['Low'], n=9, n2=25, fillna=True)
X_test['cci'] = ta.cci(price_test['High'], price_test['Low'], price_test['Adj. Close'], n=20, c=1, fillna=True)
X_test['dpo'] = ta.dpo(price_test['Adj. Close'], n=20, fillna=True)
X_test['kst_sig'] = ta.kst_sig(price_test['Adj. Close'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9,
                               fillna=True)
X_test['ichimoku_a'] = ta.ichimoku_a(price_test['High'], price_test['Low'], n1=9, n2=26, fillna=True)
X_test['ichimoku_b'] = ta.ichimoku_b(price_test['High'], price_test['Low'], n2=26, n3=52, fillna=True)
X_test['money_flow_index'] = ta.money_flow_index(price_test['High'], price_test['Low'], price_test['Adj. Close'],
                                                 price_test['Volume'], n=14, fillna=True)
X_test['rsi'] = ta.rsi(price_test['Adj. Close'], n=14, fillna=True)
X_test['tsi'] = ta.tsi(price_test['Adj. Close'], r=25, s=13, fillna=True)
X_test['uo'] = ta.uo(price_test['High'], price_test['Low'], price_test['Adj. Close'], s=7, m=14, l=28, ws=4, wm=2, wl=1,
                     fillna=True)
X_test['stoch_signal'] = ta.stoch_signal(price_test['High'], price_test['Low'], price_test['Adj. Close'], n=14, d_n=3,
                                         fillna=True)
X_test['wr'] = ta.wr(price_test['High'], price_test['Low'], price_test['Adj. Close'], lbp=14, fillna=True)
X_test['ao'] = ta.ao(price_test['Low'], price_test['Adj. Close'], s=5, l=34, fillna=True)
X_test['daily_return'] = ta.daily_return(price_test['Adj. Close'], fillna=True)
# </editor-fold>

print('creating the NN')
# <editor-fold desc="Create the neural network">
cols = X.columns
X0 = (X - X.min()) / (X.max() - X.min())
X0 = X0.loc[:, cols[0:-1]]
X0 = X0.iloc[1:, ]
X0.dropna()
X0 = X0.values

Y0 = X.loc[:, cols[-1]]
Y0 = Y0.iloc[0:-1]
Y0 = Y0.values
Y0[Y0 > 0] = 1
Y0[Y0 < 0] = 0

X0_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
X0_test = X0_test.loc[:, cols[0:-1]]
X0_test = X0_test.iloc[1:, ]
X0_test.dropna()
X0_test = X0_test.values

Y0_test = X_test.loc[:, cols[-1]]
Y0_test = Y0_test.iloc[0:-1]
Y0_test = Y0_test.values
Y0_test[Y0_test > 0] = 1
Y0_test[Y0_test < 0] = 0


cvscores = []
# </editor-fold>

# if 0:
#     # fix random seed for reproducibility
#     seed = 7
#     np.random.seed(seed)
#     define 10-fold cross validation test harness
#     kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#     for train, test in kfold.split(X0, Y0): #Only works for classification problems
#         # create model
#         model = Sequential()
#         model.add(Dense(50, input_dim=35, activation='relu'))
#         model.add(Dense(50, activation='relu'))
#         model.add(Dense(50, activation='relu'))
#         model.add(Dense(50, activation='relu'))
#         model.add(Dense(50, activation='relu'))
#         model.add(Dense(50, activation='relu'))
#         model.add(Dense(1, activation='sigmoid'))
#         # Compile model
#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         model.fit(X0[train], Y0[train],  batch_size=10, epochs=150, verbose=0)
#         # evaluate the model
#         scores = model.evaluate(X0[test], Y0[test], verbose=0)
#         print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#         cvscores.append(scores[1] * 100)
#     print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


# create model
print('Starting the NN training')
model = Sequential()
model.add(Dense(50, input_dim=35, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X0, Y0, batch_size=10, epochs=1500, verbose=0)

print('Evaluating the model')
# evaluate the model
scores = model.evaluate(X0_test, Y0_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
out = model.predict(X0_test, batch_size=None, verbose=0, steps=None)

## Do some testing on forward information
#
# px = price.loc[:, ['Adj. Close', 'High', 'Low', 'Volume']]
# cols = X.columns

## Run the analysis on expanding windows to make sure there is no forward info used

# check that the output of the ta library does not use forward information
# ctr = 0
# Xr = pd.DataFrame(index=price.index, data={'bollinger_hband_indicator': np.array([np.nan] * price.index.shape[0])})
# Xr = X
# Xr = np.nan
# for name in X.columns:
#     Xr[name] = 0
#     Xr.dropna()
#     print('Processing = {0}'.format(name))
#     idx = np.where(cols == name)
#     ctr = int(idx[0])
#     for i in range(np.size(price.index)):
#         if (i >= 10):
#             try:
#                 if (name == 'bollinger_hband_indicator'):
#                     tmp = ta.bollinger_hband_indicator(px.iloc[0:i,0], n=20, ndev=2, fillna=True)
#                 elif (name == 'bollinger_lband_indicator'):
#                     tmp = ta.bollinger_lband_indicator(px.iloc[0:i,0], n=20, ndev=2, fillna=True)
#                 elif (name == 'ema_indicator_21'):
#                     tmp = ta.ema_indicator(px.iloc[0:i,0], n=21, fillna=True)
#                 elif (name == 'ema_indicator_50'):
#                     tmp = ta.ema_indicator(px.iloc[0:i,0], n=50, fillna=True)
#                 elif (name == 'ema_indicator_200'):
#                     tmp = ta.ema_indicator(px.iloc[0:i,0], n=200, fillna=True)
#                 elif (name == 'acc_dist_index'):
#                     tmp = ta.acc_dist_index(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0], px.iloc[0:i,3])
#                 elif (name == 'on_balance_volume'):
#                     tmp = ta.on_balance_volume(px.iloc[0:i,0], px.iloc[0:i,3], fillna=True)
#                 elif (name == 'chaikin_money_flow'):
#                     tmp = ta.chaikin_money_flow(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0], px.iloc[0:i,3], n=20, fillna=True)
#                 elif (name == 'force_index'):
#                     tmp = ta.force_index(px.iloc[0:i,0], px.iloc[0:i,3], n=2, fillna=True)
#                 elif (name == 'ease_of_movement'):
#                     tmp = ta.ease_of_movement(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0], px.iloc[0:i,3], n=20, fillna=True)
#                 elif (name == 'volume_price_trend'):
#                     tmp = ta.volume_price_trend(px.iloc[0:i,0], px.iloc[0:i,3],fillna=True)
#                 elif(name == 'negative_volume_index'):
#                     tmp = ta.negative_volume_index(px.iloc[0:i,0], px.iloc[0:i,3],fillna=True)
#                 elif(name == 'average_true_range'):
#                     tmp = ta.average_true_range(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0], n=14, fillna=True)
#                 elif(name == 'keltner_channel_hband_indicator'):
#                     tmp = ta.keltner_channel_hband_indicator(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0], n=10, fillna=True)
#                 elif(name == 'keltner_channel_lband_indicator'):
#                     tmp = ta.keltner_channel_lband_indicator(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0], n=10, fillna=True)
#                 elif(name == 'donchian_channel_hband_indicator'):
#                     tmp = ta.donchian_channel_hband_indicator(px.iloc[0:i,0], n=20, fillna=True)
#                 elif(name == 'donchian_channel_lband_indicator'):
#                     tmp = ta.donchian_channel_lband_indicator(px.iloc[0:i,0], n=20, fillna=True)
#                 elif(name == 'macd_signal'):
#                     tmp = ta.macd_signal(px.iloc[0:i,0], n_fast=12, n_slow=26, n_sign=9, fillna=True)
#                 elif(name == 'adx_pos'):
#                     tmp = ta.adx_pos(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0], n=14, fillna=True)
#                 elif(name == 'adx_neg'):
#                     tmp = ta.adx_neg(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0], n=14, fillna=True)
#                 elif(name == 'vortex_indicator_neg'):
#                     tmp = ta.vortex_indicator_neg(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0], n=14, fillna=True)
#                 elif(name == 'vortex_indicator_pos'):
#                     tmp = ta.vortex_indicator_pos(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0], n=14, fillna=True)
#                 elif(name == 'trix'):
#                     tmp = ta.trix(px.iloc[0:i,0], n=15, fillna=True)
#                 elif(name == 'mass_index'):
#                     tmp = ta.mass_index(px.iloc[0:i,1], px.iloc[0:i,2], n=9, n2 = 25, fillna=True)
#                 elif(name == 'cci'):
#                     tmp = ta.cci(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0], n=20, c=0.015, fillna=True)
#                 elif(name == 'dpo'):
#                     tmp = ta.dpo(px.iloc[0:i,0], n=20, fillna=True)
#                 elif(name == 'kst_sig'):
#                     tmp = ta.kst_sig(px.iloc[0:i,0], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9, fillna=True)
#                 elif(name == 'ichimoku_a'):
#                     tmp = ta.ichimoku_a(px.iloc[0:i,1], px.iloc[0:i,2], n1=9, n2=26, fillna=True)
#                 elif(name == 'ichimoku_b'):
#                     tmp = ta.ichimoku_b(px.iloc[0:i,1], px.iloc[0:i,2], n2=26, n3=52, fillna=True)
#                 elif(name == 'money_flow_index'):
#                     tmp = ta.money_flow_index(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0], px.iloc[0:i,3], n=14, fillna=True)
#                 elif(name == 'rsi'):
#                     tmp = ta.rsi(px.iloc[0:i,1], n=14, fillna=True)
#                 elif(name == 'tsi'):
#                     tmp = ta.tsi(px.iloc[0:i,0], r=25, s=13, fillna=True)
#                 elif(name == 'uo'):
#                     tmp = ta.uo(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0],s=7,m=14,l=28,ws=4,wm=2,wl=1,fillna=True)
#                 elif(name == 'stoch_signal'):
#                     tmp = ta.stoch_signal(px.iloc[0:I,1], px.iloc[0:i,2], px.iloc[0:i,0],n=14,d_n=3,fillna=True)
#                 elif(name == 'wr'):
#                     tmp = ta.wr(px.iloc[0:i,1], px.iloc[0:i,2], px.iloc[0:i,0],lbp=14,fillna=True)
#                 elif(name == 'ao'):
#                     tmp = ta.ao(px.iloc[0:i,2], px.iloc[0:i,0], s=5, l=34, fillna=True)
#                 elif(name == 'daily_return'):
#                     tmp = ta.daily_return(px.iloc[0:i,0], fillna=True)
#             except:
#                 print('Error in name={0}, i={1}, ctr={2}'.format(name,i,ctr))
#                 tmp = [0]
#
#             Xr.iloc[i,int(ctr)] = tmp[-1]
# #     ctr += 1


# In[61]:
