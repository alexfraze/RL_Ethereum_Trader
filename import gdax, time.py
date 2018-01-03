# public_client = gdax.PublicClient()
# Paramters are optional
# result = ws.recv()
# print('Result: {}'.format(result))

subscribe = {
    "type": "subscribe",
    "channels": [{ "name": "level2", "product_ids": ["ETH-USD"] }]
}

ws.send(json.dumps(subscribe))
result = ws.recv()
print('Result: {}'.format(result))




updates = 0
while(updates<100):
	result = ws.recv()
	print('Result: {}'.format(result))



wsClient.close()
# 1684865891
