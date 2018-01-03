import gdax, time

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras import initializers
from websocket import create_connection
import websocket
import json
from keras.layers import LSTM
from keras.models import model_from_json

import numpy as np

# public_client = gdax.PublicClient()
# Paramters are optional
# Do other stuff...


class trading_manager(object):
	"""docstring for RL_manager"""
	def __init__(self, starting_USD):
		self.curr_eth_to_US_ratio = 0
		self.curr_ETH_volume = 0
		self.USD = starting_USD
		self.ETH = 0
		self.public_client = gdax.PublicClient()

	def update_currency_values(self):
		self.curr_eth_to_US_ratio = float(self.public_client.get_product_ticker(product_id='ETH-USD')['price'])#pull from GDAX

	def buy_eth(self, USD_amount):
		if(self.USD >= USD_amount):
			self.ETH += (float(USD_amount)/self.curr_eth_to_US_ratio)*0.997 #.3% guestimated fee
			self.USD -= USD_amount
		else:
			self.ETH += (float(self.USD)/self.curr_eth_to_US_ratio)*0.997 #.3% guestimated fee
			self.USD -= self.USD

	def sell_eth(self, ETH_amount):
		if(self.ETH>=ETH_amount):
			self.USD += float(ETH_amount)*self.curr_eth_to_US_ratio*0.997 #.3% fee
			self.ETH -= ETH_amount
		else:
			self.USD += float(self.ETH)*self.curr_eth_to_US_ratio*0.997 #.3% fee
			self.ETH -= self.ETH
			
	def hold_eth(self):
		return 
	# def update(self, last_input, last_choice):

	def print_portfolio(self):
		self.update_currency_values()
		print("\n")
		print("new portfolio value: " + str(self.USD + self.ETH*float(new_price)))
		print("USD: " + str(self.USD))
		print("ETH: " + str(self.ETH) + "   USD amount: " + str(self.ETH*float(new_price)))
		
	
	def calculate_current_value(self, currency='USD'):
		self.update_currency_values()
		USD_value = self.USD + self.ETH*float(self.curr_eth_to_US_ratio)
		if(currency=="ETH"): return (float(USD_value)/self.curr_eth_to_US_ratio)
		else: return USD_value 

class kerras_trading_net(object):
	"""docstring for kerras_trading_net"""
	def __init__(self, manager):
		self.NN = Sequential()
		self.trade_manager = manager 
		self.lr = 0.01
		self.mini_batch_training = []	
		self.mini_batch_targets = []


	def init_network(self):
		# initializer = initializers.RandomNormal(mean=0.0, stddev=0.0005, seed=None)
		# self.NN.add(Dense(48, kernel_initializer=initializer, activation='relu', input_shape=(100,)))
		# self.NN.add(Dense(24, kernel_initializer=initializer, activation= 'relu'))
		# self.NN.add(Dense(12, kernel_initializer=initializer, activation='relu'))
		# self.NN.add(Dense(3, kernel_initializer=initializer, activation= 'linear'))
		# rms = RMSprop(lr=self.lr)
		# self.NN.compile(loss='mse', optimizer=rms)
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights("model.h5")
		rms = RMSprop(lr=self.lr)
		self.NN = loaded_model
		self.NN.compile(loss='mse', optimizer=rms)
		print("Loaded model from disk")		

	def trade(self, price_history):
		state = price_history
		prediction = self.NN.predict(state.reshape(1, 100), batch_size=1)[0]
		return prediction

	def update(self, price_history, decision, prediction, new_price, new_portfolio, old_portfolio, portfolio_balance):
		reward = (new_portfolio - old_portfolio)*50
		print(reward)
		reward -= portfolio_balance/500
		print("reward: " + str(reward))
		gamma = .8
		updated_state_prediction = reward+gamma*np.max(prediction)
		print(updated_state_prediction)
		learning_rate = 0.2 #do i need this second learning rate or is the NN arleady doing this for me?
		print(prediction)
		prediction[decision] = ((1-learning_rate)*prediction[decision] + 
		learning_rate*updated_state_prediction)
		print(prediction)
		# record updated prediction and old state in your mini-batch
		updated_prediction = prediction
		self.mini_batch_targets.append(updated_prediction)
		self.mini_batch_training.append(price_history)
		sub_batch_size = 10
		mini_batch_size = 30
		epochs = 1000
		print("mini batch size: " + str(len(self.mini_batch_targets)))
		if(len(self.mini_batch_targets)>=mini_batch_size):
			print("back propogating")
			# indexes = xrange(sub_batch_size)
			indexes = np.random.choice(mini_batch_size, sub_batch_size, replace=False)
			self.NN.fit(np.array(self.mini_batch_training)[indexes], 
				np.array(self.mini_batch_targets)[indexes], batch_size=sub_batch_size, epochs=epochs, verbose=0)
			sample_to_forget = indexes[0] #TA told other kids to forget one of used samples.
			del self.mini_batch_targets[sample_to_forget]
			del self.mini_batch_training[sample_to_forget]


trader = trading_manager(1000.0)
trader.update_currency_values()
network = kerras_trading_net(trader)
network.init_network()
print("starting funds")
print(str(trader.USD) + "\n")

sleep = 30
index = 0
curr_price = float(trader.public_client.get_product_ticker(product_id='ETH-USD')['price'])
# 	
starting_price_history = np.repeat(curr_price, 100)
# starting_price_history = np.zeros([60])
# fill the input array with 5 mins of trading prices
# while(index<=59):
# 	curr_price = float(trader.public_client.get_product_ticker(product_id='ETH-USD')['price'])
# 	starting_price_history[index] = curr_price
# 	print(starting_price_history)
# 	time.sleep(sleep)
# 	index+=1

price_history = starting_price_history
print("starting live RL and trading")
#start Reinforcement training with 5 minutes of data and train with new data every seconds.

#train for 2 hours
reps = 2000
starting_rep = 0
while(starting_rep<=reps):
	#make decision
	outcome = network.trade(price_history)
	
	#parse outcome and perform corresponding trade, hold, or buy
	# outcome = outcome-np.min(outcome)
	# weighted_outcome = outcome #weight outcomes? 
	decision = np.argmax(outcome)
	pos_outcome = outcome-np.min(outcome)
	decision_weight = float(pos_outcome[decision]/np.sum(pos_outcome))*20
	print(outcome)
	print(decision)
	print(decision_weight)
	time.sleep(sleep)
	if(decision==0): 
		trader.buy_eth(decision_weight) #buy USD weight worth of ETH
	if(decision==2):
		trader.sell_eth(decision_weight/float(price_history[99])) #sell USD weigth worth of ETH.
	old_portfolio = trader.calculate_current_value()
	new_portfolio = trader.calculate_current_value()
	new_price = trader.curr_eth_to_US_ratio
	trader.print_portfolio()
	#reinforcemnt train the network
	portfolio_balance = abs(trader.USD - float(trader.ETH/new_price))
	network.update(price_history, decision, outcome, new_price, new_portfolio, old_portfolio, portfolio_balance)
	#add current pricing hist and output to minibatch

	#update pricing
	price_history = np.roll(price_history, -1)
	price_history[99] = new_price
	print("repetition: " + str(starting_rep))
	if(starting_rep%50==0):
		print("writing network weights to file")
		#save network weights
		json_network = network.NN.to_json()
		with open("model.json", "w") as json_file:
		    json_file.write(json_network)
		network.NN.save_weights("model.h5")
		print("Saved model to disk")	
	starting_rep+=1

public_client.close()
