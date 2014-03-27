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

  def get_data(self, startdate, enddate, ls_symbols):
    # We need closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(startdate, enddate, dt_timeofday)

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = self.c_dataobj.get_data(ldt_timestamps, ls_symbols, self.ls_keys)
    d_data = dict(zip(self.ls_keys, ldf_data))

    # Filling the data for NAN
    for s_key in self.ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    return d_data

  def do_optimize(self, d_data, allocations = 'uniform', step = 0.1,
                        maxIter = 100, EPSILON = 1e-7):
    # randomly initialize portfolio
    na_price = d_data['close'].values
    nstocks = na_price.shape[1]

    if not allocations or allocations == 'uniform':
      allocations = np.ones(nstocks) / nstocks
    elif allocations == 'random':
      allocations = np.random.random(nstocks)
      allocations = allocations / np.mean(allocations)

    def search_direction(allocations):
      # simple strategy: explore each possible direction
      best_sharpe = 0.0
      best_allocations = allocations
      pair = (-1, -1)
      idx = range(nstocks)
      for i in idx:
        for j in idx:
          if i == j:
            continue

          delta = np.ones(nstocks)
          new = list(allocations)
          new[i] += 0.1
          new[j] -= 0.1
          if new[i] > 1.0 or new[j] < 0.0:
            continue
          _, _, sharpe, _ = self.do_simulate(d_data, new)
          # print 'Searching %d, %d, ' % (i, j)
          # print "Sharpe = %f. " % sharpe
          # print 'allocations = ', new
          if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_allocations = new
            pair = (i, j)

      return best_sharpe, best_allocations

    numIter = 0
    old_sharpe = 0
    old_allocations = allocations
    diff = 100
    while numIter < maxIter:
      numIter += 1
      sharpe, allocations = search_direction(old_allocations)
      print "Iter #%d, sharpe = %f. " % (numIter, sharpe)
      print "Allocation = ", allocations
      diff = sharpe - old_sharpe
      if diff < EPSILON:
        break

      old_sharpe = sharpe
      old_allocations = allocations

    return old_allocations

  def optimize_portfolio(self, startdate, enddate, ls_symbols,
                         allocations = None):
    d_data = self.get_data(startdate, enddate, ls_symbols)
    return self.do_optimize(d_data, allocations)

  def do_simulate(self, d_data, allocations):
    # Getting the numpy ndarray of close prices.
    na_price = d_data['close'].values

    # Normalizing the prices to start at 1 and see relative returns
    na_normalized_price = na_price / na_price[0, :]

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

  def simulate(self, startdate, enddate, ls_symbols, allocations):
    '''
    Simulate and access the performance of a portfolio
    @return: volalitility, daily_ret, sharp, cum_ret
    '''
    d_data = self.get_data(startdate, enddate, ls_symbols)
    return self.do_simulate(d_data, allocations)


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