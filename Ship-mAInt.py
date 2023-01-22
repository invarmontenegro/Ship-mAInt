import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Ako su input i output vec definisani 
x = Input.T
t = Output.T

# definise "classifier"
clf = MLPClassifier(hidden_layer_sizes=(10,), solver='adam')

# podjela podataka na trening i testnu grupu 
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=42)

# ubacivanje classifier u testnu grupu 
clf.fit(x_train, t_train)

# kreira predikciju testnih vrijednosti 
t_pred = clf.predict(x_test)

# racunanje preciznosti modela 
performance = accuracy_score(t_test, t_pred)

# printanje classifier parametara
print(clf)


# importovati matplotlib.pyplot kao plt
# plt.plot(performance)
# plt.show()
# Napomena da su funkcije train_test_split i accuracy_score iz scikit-learn biblioteke
# MLPClassifier je klasifikator za neuronsku mrežu, funkcija view(net) zamijenjena je sa print(clf) koja će ispisati parametre klasifikatora
# Možete koristiti matplotlib biblioteku za crtanje performansi modela
# Prilikom testiranja koda obavezno importati "Numpy" library
# Da bi program radio potrebno je ubaciti bazu podataka za potrebne varijable 
