'''
Homework1 of computational investing

@author: Zigang Xiao
'''

import numpy as np

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd


class Simulator:
  def __init__(self):
    # Creating an object of the dataaccess class with Yahoo as the source.
    self.c_dataobj = da.DataAccess('Yahoo', cachestalltime=0)

    # Keys to be read from the data, it is good to read everything in one go.
    self.ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

  def simulate(self, startdate, enddate, ls_symbols, allocations):
    '''
    Simulate and access the performance of a portfolio
    @return: volalitility, daily_ret, sharp, cum_ret
    '''
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(startdate, enddate, dt_timeofday)

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = self.c_dataobj.get_data(ldt_timestamps, ls_symbols, self.ls_keys)
    self.d_data = dict(zip(self.ls_keys, ldf_data))

    # Filling the data for NAN
    for s_key in self.ls_keys:
        self.d_data[s_key] = self.d_data[s_key].fillna(method='ffill')
        self.d_data[s_key] = self.d_data[s_key].fillna(method='bfill')
        self.d_data[s_key] = self.d_data[s_key].fillna(1.0)

    # Getting the numpy ndarray of close prices.
    self.na_price = self.d_data['close'].values

    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_price = self.na_price / self.na_price[0, :]

    # cumulative daily portfolio value
    na_cumret = np.sum(na_normalized_price * allocations, axis = 1)

    # Calculate the daily returns of the prices. (Inplace calculation)
    na_rets = na_cumret.copy()
    na_rets = tsu.returnize0(na_rets)

    # average daily return
    avg_ret = np.mean(na_rets)

    # standard dev of daily returns
    vol = np.std(na_rets)

    sharpe = avg_ret / vol
    return vol, avg_ret, sharpe, na_cumret[-1]

def report_stats(dt_start, dt_end, ls_symbols, allocations,
                 vol, daily_ret, cum_ret):
  print 'Start Date:', dt_start
  print 'End Date:', dt_end
  print 'Symbols:', ls_symbols
  print 'Optimal Allocations:', allocations
  print 'Volatility (stdev of daily returns):', vol
  print 'Average Daily Return:', daily_ret
  print 'Cumulative Return:', cum_ret

def test1():
  # List of symbols
  ls_symbols = ['AAPL', 'GLD', 'GOOG', 'XOM']

  # Start and End date of the charts
  dt_start = dt.datetime(2011, 1, 1)
  dt_end = dt.datetime(2011, 12, 31)

  # allocations to the equities
  allocations = [0.4, 0.4, 0.0, 0.2]
  opt = Simulator()
  vol, daily_ret, sharpe, cum_ret = opt.simulate(
                                  dt_start, dt_end, ls_symbols, allocations)

  report_stats(dt_start, dt_end, ls_symbols, allocations, 
               vol, daily_ret, cum_ret)

def test2():
  # List of symbols
  ls_symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']

  # Start and End date of the charts
  dt_start = dt.datetime(2010, 1, 1)
  dt_end = dt.datetime(2010, 12, 31)

  # allocations to the equities
  allocations = [0.0, 0.0, 0.0, 1.0]
  opt = Simulator()
  vol, daily_ret, sharpe, cum_ret = opt.simulate(
                                  dt_start, dt_end, ls_symbols, allocations)

  report_stats(dt_start, dt_end, ls_symbols, allocations, 
               vol, daily_ret, cum_ret)

def main():
  ''' Main Function'''
  test1()
  test2()

if __name__ == '__main__':
    main()