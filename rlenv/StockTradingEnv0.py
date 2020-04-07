import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np


MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
MAX_DAY_CHANGE = 1

INITIAL_ACCOUNT_BALANCE = 10000
DATA_HIS_PERIOD = 5


# position constant
FLAT = 0    #  no position
LONG = 1    # buy position
SHORT = 2   # sell position

# action constant
HOLD = 0
BUY = 1
SELL = 2


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df,show_trade=True):
        super(StockTradingEnv, self).__init__()

        #  show the trade info
        self.show_trade = show_trade
        self.actions=["FLAT","LONG","SHORT"]
        self.fee = 0.0005   # brokage commission
        self.df = df
        self.closeprices = self.df['close'].values
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Discrete(len(self.actions))
        # self.action_space = spaces.Box(
        #     low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(DATA_HIS_PERIOD+1,6), dtype=np.float16)

        self.history = []

    def _next_observation(self):
        obs = np.array([
            self.df.loc[self.current_step-DATA_HIS_PERIOD:self.current_step, 'open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step-DATA_HIS_PERIOD:self.current_step, 'high'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step-DATA_HIS_PERIOD:self.current_step, 'low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step-DATA_HIS_PERIOD:self.current_step, 'close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step-DATA_HIS_PERIOD:self.current_step, 'volume'].values / MAX_NUM_SHARES,
            ])
        # Append additional data and scale each value to between 0-1
        obs = np.append(obs,[[self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE)]],axis=0)
        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        # current_price = random.uniform(
        #     self.df.loc[self.current_step, "open"], self.df.loc[self.current_step, "close"])

        # Set the current price to the last close price
        self.close_price = self.df.loc[self.current_step,"close"]

        amount = 0.5   #the old version has this variable, so reserve

        # action comes from the agent
        # 1 buy, 2 sell, 0 hold
        # single position can be opened per trade
        # valid action sequence would be
        # LONG : buy - hold - hold - sell
        # SHORT : sell - hold - hold - buy
        # invalid action sequence is just considered hold
        # (e.g.) "buy - buy" would be considred "buy - hold"
        self.action = HOLD   #hold
        if action == BUY:  #buy
            if self.position == FLAT:   # if previous position was flat
                self.position = LONG   #update position to long
                self.action = BUY      # record action as buy
                self.entry_price = self.close_price
                # Buy amount % of balance in shares
                total_possible = int(self.balance / self.close_price)
                shares_bought = int(total_possible * amount)//100 *100

                self.krw_balance = shares_bought * self.entry_price    # buy balance
                commission = round(self.fee * self.krw_balance,2)  # commission fee
                self.shares_held = shares_bought
                self.balance -= self.krw_balance-commission
                #self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            elif self.position == SHORT: # if previous position was short
                self.position = FLAT   # update position to flat
                self.action = BUY      # record action as buy
                self.exit_price = self.close_price
                self.reward += ((self.entry_price - self.exit_price) / self.exit_price + 1) * (
                        1 - self.fee) ** 2 - 1  # calculate reward
                #self.krw_balance = self.krw_balance * (1.0 + self.reward)  # evaluate cumulative return in krw-won
                self.balance += round(self.krw_balance * (1.0 + self.reward),2)    # calcuate the total balance
                self.n_short += 1  # record number of short
                self.total_shares_sold += self.shares_held
                self.total_sales_value += self.shares_held * self.close_price
                self.entry_price = 0  # clear entry price
                self.shares_held = 0  # clear the shares_
        elif action == SELL:
            if self.position == FLAT:
                self.position = SHORT
                self.action = SELL
                self.entry_price = self.close_price
                # Sell amount % of shares held
                total_possible = int(self.balance / self.close_price)
                self.shares_held = int(total_possible * amount)//100 *100

                self.krw_balance = self.shares_held * self.entry_price    # buy balance
                commission = round(self.fee * self.krw_balance,2)  # commission fee
                self.balance -= self.krw_balance-commission
            elif self.position == LONG:
                self.position = FLAT
                self.action = SELL
                self.exit_price = self.close_price
                self.reward += ((self.exit_price - self.entry_price) / self.entry_price + 1) * (1 - self.fee) ** 2 - 1
                #self.krw_balance = self.krw_balance * (1.0 + self.reward)
                self.balance += round(self.krw_balance*(1.0+self.reward),2)
                self.n_long += 1
                self.total_shares_buy += self.shares_held
                self.total_buys_value += self.shares_held * self.close_price
                self.shares_held = 0
                self.entry_price = 0

        # [coin + krw_won] total value evaluated in krw won
        if (self.position == LONG):
            temp_reward = ((self.close_price - self.entry_price) / self.entry_price + 1) * (
                        1 - self.fee) ** 2 - 1
            new_portfolio = self.krw_balance * (1.0 + temp_reward)
        elif (self.position == SHORT):
            temp_reward = ((self.entry_price - self.close_price) / self.close_price + 1) * (
                        1 - self.fee) ** 2 - 1
            new_portfolio = self.krw_balance * (1.0 + temp_reward)
        else:
            temp_reward = 0
            new_portfolio = 0



        self.net_worth = self.balance + new_portfolio

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

        self.portfolio = round(new_portfolio,2)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        done = False

        self.current_step += 1

        delay_modifier = (self.current_step / MAX_STEPS)

        # profits
        #reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
        #reward = 1 if reward > 0 else -100

        if self.net_worth <= 0:
            done = True

        if self.current_step > len(self.df.loc[:, 'open'].values) - 1:
            self.current_step = DATA_HIS_PERIOD  # loop training
            # when loop training, then clear the history
            self.action = HOLD
            self.position = FLAT
            self.balance = INITIAL_ACCOUNT_BALANCE
            self.net_worth = INITIAL_ACCOUNT_BALANCE
            self.max_net_worth = INITIAL_ACCOUNT_BALANCE
            self.krw_balance = 0
            self.reward = 0
            self.portfolio = 0
            self.shares_held = 0
            self.cost_basis = 0
            self.total_shares_buy = 0
            self.total_buys_value = 0
            self.total_shares_sold = 0
            self.total_sales_value = 0
            self.n_long = 0
            self.n_short = 0
            self.history=[]
            # done = True


        if (self.show_trade and self.current_step % 1 == 0):
            print("Tick: {0}/ Portfolio (krw-won): {1}, balance: {2}".format(self.current_step, self.portfolio,self.net_worth))
            print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))

        # save the history data
        self.history.append([
                             self.action,
                             self.position,
                             self.current_step,
                             self.close_price,
                             self.krw_balance,
                             self.balance,
                             self.max_net_worth,
                             self.shares_held,
                             self.portfolio,
                             self.total_shares_buy,
                             self.total_buys_value,
                             self.total_shares_sold,
                             self.total_sales_value])
        #self.history.append((self.action, self.current_step, self.closingPrice, self.portfolio, self.reward))
        obs = self._next_observation()
        if (self.current_step > (self.df.shape[0]) - 1):
            self.done = True
            self.reward = self.get_profit()  # return reward at end of the game
        return obs, self.net_worth, done, {'portfolio': np.array([self.portfolio]),
                                                    "history": self.history,
                                                    "n_trades": {'long': self.n_long, 'short': self.n_short}}

        #return obs, reward, done, {}
    def get_profit(self):
        if(self.position == LONG):
            profit = ((self.close_Price - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
        elif(self.position == SHORT):
            profit = ((self.entry_price - self.close_Price)/self.close_Price + 1)*(1-self.fee)**2 - 1
        else:
            profit = 0
        return profit

    def reset(self, new_df=None):
        # Reset the state of the environment to an initial state
        self.action = HOLD
        self.position = FLAT
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.krw_balance = 0
        self.reward =0
        self.portfolio =0
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_buy =0
        self.total_buys_value=0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.n_long=0
        self.n_short=0

        self.history=[]

        # pass test dataset to environment
        if new_df:
            self.df = new_df

        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'open'].values) - 6)

        # the observation include the given period history data
        self.current_step = DATA_HIS_PERIOD  #random.randint(DATA_HIS_PERIOD,len(self.df.loc[:,'open'].values)-1)

        # for i in range(DATA_HIS_PERIOD):
        #     self.history.append([0.0,0.0,0.0,0.0,0.0,0.0])

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print('-'*30)
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        return profit
