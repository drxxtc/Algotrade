import os
import numpy as np
import pandas as pd
from pytz import UTC
from datetime import timedelta
from datetime import datetime

from ops import neutralize,normalize,truncate

def load_data(data_dir):
    names = os.listdir(data_dir)
    data,tickers = [],[]
    for name in names:
        ticker,ext = os.path.splitext(name)
        if ext != '.parquet':
            continue
        df = pd.read_parquet(os.path.join(data_dir,name))
        if 'end_t' in df.columns:
            df = df.drop('end_t',axis=1).set_index('start_t')
        elif df.index.name == 'Date':
            pass
        else:
            df = df[df.index.map(lambda x : x[-8:]>='10:00:00' and x[-8:]<'19:00:00')]
        data.append(df)
        tickers.append(ticker)
    if len(data):
        data = pd.concat(data,keys=tickers)
        f_names = data.columns.values
        data = data.unstack(level=0).sort_index()
        if data.index.name == 'start_t':
            data.index = data.index.map(lambda x : datetime.fromtimestamp(x/1000).astimezone(UTC)).map(lambda x : x.astimezone(None))
        elif data.index.name == 'Date':
            _f_names = [x.lower() for x in f_names]
            data = data.rename(columns=dict(zip(f_names,_f_names)))
            f_names = _f_names
        else:
            data.index = data.index.map(lambda x : datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        data.rename_axis(index=None,inplace=True)
        data = dict((f_name,data[f_name]) for f_name in f_names)
        return data
    else:
        return dict()

class Simulator:
    def __init__(self,data_dir):
        self.data_dir = data_dir
        
        self.data = dict()
        names = os.listdir(self.data_dir)
        for name in names:
            if os.path.isfile(os.path.join(self.data_dir,name)):
                n,ext = os.path.splitext(name)
                if ext == '.csv':
                    df = pd.read_csv(os.path.join(self.data_dir,name),index_col=0)
                    if df.shape[1] == 1:
                        df = df[df.columns[0]]
                    self.data[n] = df
                elif ext == '.parquet':
                    self.data[n] = pd.read_parquet(os.path.join(self.data_dir,name))
            else:
                self.data[name] = load_data(os.path.join(self.data_dir,name))
        
        self.datetimes = self.data['spot']['close'].index.values
        self.time_step = int(min([(x-y).astype('timedelta64[s]').item().total_seconds() for x,y in zip(self.datetimes[1:],self.datetimes[:-1])]))

        self.universe = dict()
        for market in self.data.keys():
            if type(self.data[market]) is not dict:
                continue
            self.universe[market] = dict()
            if 'quote_volume' in self.data[market]:
                liq_rank = self.data[market]['quote_volume'].rolling(int(28*24*60*60/self.time_step)).sum().rank(axis=1,ascending=False).shift()
                for n in [25,50,100,200,300]:
                    self.universe[market][n] = liq_rank <= n
            elif 'dividends' in self.data[market]:
                liq_rank = (self.data[market]['volume']*self.data[market]['close']).rolling(28).sum().rank(axis=1,ascending=False).shift()
                for n in [100,200,300,400,450,500]:
                    self.universe[market][n] = liq_rank <= n
            else:
                liq_rank = self.data[market]['value'].rolling(int(20*9*6)).sum().rank(axis=1,ascending=False).shift()
                for n in [25,50,100,200,300]:
                    self.universe[market][n] = liq_rank <= n
    
    def run_weights(self,weights,max_weight=0.05,market='spot',universe_size=None,delay=0):
        if delay > 0:
            weights = weights.shift(delay)
        if universe_size:
            weights = weights.where(self.universe[market][universe_size])
            weights = weights.where(self.universe[market][universe_size].sum(axis=1)>=universe_size,axis=0)
        
        weights = neutralize(weights)
        weights = normalize(weights)
        for _ in range(5):
            weights = truncate(weights,max_weight)
            weights = neutralize(weights)
            weights = normalize(weights)
        
        if delay == 0:
            close = self.data[market]['close']
            returns = close/close.shift()-1
            returns = returns.clip(-4,4)
        else:
            vwap = self.data[market]['quote_volume']/self.data[market]['volume']
            returns = vwap/vwap.shift()-1
            returns = returns.clip(-4,4)
        
        pnl = weights.shift()*returns
        tvr = (weights-weights.shift()*(1+returns)).abs()
        
        return dict(
            weights = weights,
            pnl = pnl,
            tvr = tvr
        )
    
    def show_result(self,result):
        valid = (result['weights'].notnull().sum(axis=1)>0).values
        start_dt = self.datetimes[valid][0]
        pnl = result['pnl'].loc[start_dt:].sum(axis=1).resample('1d').sum()
        pnl.cumsum().plot(grid=True)
        tvr = result['tvr'].loc[start_dt:].sum(axis=1).resample('1d').sum()
        print ('avg daily tvr           = {}%'.format(np.round(100*tvr.mean(),2)))
        print ('annualized ret          = {}%'.format(np.round(100*365*pnl.mean(),2)))
        print ('ir                      = {}'.format(np.round(pnl.mean()/pnl.std(),4)))
        print ('annualized sharpe ratio = {}'.format(np.round(np.sqrt(365)*pnl.mean()/pnl.std(),2)))
