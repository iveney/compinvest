'''
Homework1 of computational investing

@author: Zigang Xiao
'''

import operator
import numpy as np
import scipy.optimize as so

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

# given an iterable of pairs return the key corresponding to the greatest value
def argmax(pairs):
  if not pairs: return None, 0.0
  return argmax_pair(pairs)[0]

def argmax_pair(pairs):
  if not pairs: return None, 0.0
  return max(pairs, key = operator.itemgetter(1))

class PortfolioOptimizer:
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

    def hill_climbing(allocations):
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
    old_sharpe = float("-inf")
    old_allocations = allocations
    while numIter < maxIter:
      numIter += 1
      sharpe, allocations = hill_climbing(old_allocations)
      diff = sharpe - old_sharpe
      print "Iter #%d, sharpe = %f. Diff = %f" % (numIter, sharpe, diff)
      print "Allocation = ", allocations
      if diff < EPSILON:
        break

      old_sharpe = sharpe
      old_allocations = allocations

    return old_allocations

  def spherical2cartesian(self, polars):
    n = len(polars) + 1
    x = [0] * n

    # [1, sin(phi_1), ..., sin(phi_n-1)]
    sin = np.sin(polars)
    sin = np.concatenate([[1], sin])

    # [cos(phi_1), ..., cos(phi_n-1), 1]
    cos = np.cos(polars)
    cos = np.concatenate([cos, [1]])

    x = np.cumprod(sin)
    x *= cos

    return x

  def spherical2allocation(self, polars):
    return np.square(self.spherical2cartesian(polars))

  def spherical_optimize(self, d_data, x0):
    '''
    Use spherical coordinate:
    x1, x2 ... xn <-> phi_1 phi_2, ... phi_n-1, where phi_n-1 ranges [0, 2pi]
    and the others [0, pi]
    @params: x0 is the initial guess
    '''

    # since its maximize, negate the sharpe
    def f(x): return -self.sharpe(d_data, self.spherical2allocation(x))

    n = len(x0)
    bounds = [(0, np.pi/2)] * n
    # bounds[-1] = (0, 2*np.pi)
    x, nfeval, rc = so.fmin_tnc(f, x0, approx_grad = True, bounds = bounds,
                                disp = 0)

    return self.spherical2allocation(x)

  def multiple_restart(self, d_data, n):
    '''
    Two pass strategy
    1. use pi's divisions as starting point
    2. use random points

    @params: n is the length of the spherical coordinates
    '''

    # pass 1
    ndiv = 10
    divs = np.array([np.pi] * ndiv) / range(1, ndiv + 1)
    divs = np.array([divs] * n).transpose()

    # pass 2
    nrnd = 10
    rand = np.random.random(nrnd)
    rand = np.array([np.pi * rand] * n).transpose()

    init = np.concatenate([divs, rand])
    xs = [self.spherical_optimize(d_data, x0) for x0 in init]
    sharpes = [self.sharpe(d_data, x) for x in xs]
    pairs = zip(xs, sharpes)
    xopt = argmax(pairs)
    return xopt

  def optimize_portfolio(self, startdate, enddate, ls_symbols,
                         allocations = None):
    '''
    if x0 (allocations) is given, just use it.
    Otherwise use random restart technique
    '''
    d_data = self.get_data(startdate, enddate, ls_symbols)
    # return self.do_optimize(d_data, allocations)

    # generate initial spherical coordinate
    if not allocations:
      n = len(ls_symbols)
      return self.multiple_restart(d_data, n-1)
    else:
      x0 = allocations
      return self.spherical_optimize(d_data, x0)

  def sharpe(self, d_data, allocations):
    return self.do_simulate(d_data, allocations)[2]

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

    sharpe = np.sqrt(252) * avg_ret / vol
    return vol, avg_ret, sharpe, na_cumret[-1]

  def simulate(self, startdate, enddate, ls_symbols, allocations):
    '''
    Simulate and access the performance of a portfolio
    @return: volalitility, daily_ret, sharp, cum_ret
    '''
    d_data = self.get_data(startdate, enddate, ls_symbols)
    return self.do_simulate(d_data, allocations)

def report_stats(dt_start, dt_end, ls_symbols, allocations, sharpe,
                 vol, daily_ret, cum_ret):
  print 'Start Date:', dt_start
  print 'End Date:', dt_end
  print 'Symbols:', ls_symbols
  print 'Optimal Allocations:', allocations
  print 'Sharpe Ratio:', sharpe
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
  opt = PortfolioOptimizer()
  vol, daily_ret, sharpe, cum_ret = opt.simulate(
                                  dt_start, dt_end, ls_symbols, allocations)

  report_stats(dt_start, dt_end, ls_symbols, allocations, sharpe,
               vol, daily_ret, cum_ret)

def test2():
  # List of symbols
  ls_symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']

  # Start and End date of the charts
  dt_start = dt.datetime(2010, 1, 1)
  dt_end = dt.datetime(2010, 12, 31)

  # allocations to the equities
  allocations = [0.0, 0.0, 0.0, 1.0]
  opt = PortfolioOptimizer()
  vol, daily_ret, sharpe, cum_ret = opt.simulate(
                                  dt_start, dt_end, ls_symbols, allocations)

  report_stats(dt_start, dt_end, ls_symbols, allocations, sharpe,
               vol, daily_ret, cum_ret)

def Question1():
  ls_symbols = ['AAPL', 'GOOG', 'IBM', 'MSFT']

  # Start and End date of the charts
  dt_start = dt.datetime(2010, 1, 1)
  dt_end = dt.datetime(2010, 12, 31)

  # allocations to the equities
  allocations = [0.0, 0.0, 0.0, 1.0]
  opt = PortfolioOptimizer()
  allocations = opt.optimize_portfolio(dt_start, dt_end, ls_symbols)
  sharpe = opt.simulate(dt_start, dt_end, ls_symbols, allocations)[2]
  print allocations, sharpe

def Question2():
  ls_symbols = ['C', 'GS', 'IBM', 'HNZ']

  # Start and End date of the charts
  dt_start = dt.datetime(2010, 1, 1)
  dt_end = dt.datetime(2010, 12, 31)

  # allocations to the equities
  allocations = [0.0, 0.0, 0.0, 1.0]
  opt = PortfolioOptimizer()
  allocations = opt.optimize_portfolio(dt_start, dt_end, ls_symbols)
  sharpe = opt.simulate(dt_start, dt_end, ls_symbols, allocations)[2]
  print allocations, sharpe

def main():
  ''' Main Function'''
  test1()
  test2()

if __name__ == '__main__':
    main()