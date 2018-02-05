from keras.models import Sequential
from keras.layers import Dense
import numpy

numpy.random.seed(7)

# load file

dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")

# split data

X = dataset[:,0:8] # inputs
Y = dataset[:,8] # outputs

# create model

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model

model.fit(X, Y , epochs=150, batch_size=10)

# evaluate the model
#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# predict

predictions = model.predict(X)

rounded = [round(x[0]) for x in predictions]
print(rounded)