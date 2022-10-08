import pickle

with open('KinokkoTakenoko.pkl', 'rb') as f:
	model2 = pickle.load(f)

suzuki = [[180, 75, 30]]

print(model2.predict(suzuki))