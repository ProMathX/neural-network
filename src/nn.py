
# fetching the mnist dataset --> see example in assets/mnist-img/MnistExamples.png
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Import the dataset
print("Impoting the MNIST dataset...")
mnist = fetch_openml('mnist_784')
S_X = mnist.data
y = mnist.target

# split the dataset
print("Splitting the dataset")
X_train, X_test, y_train, y_test= train_test_split(S_X, y, test_size=0.33)

# train the dataset
print("Training the neural network")
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(15,15))

clf.fit(X_train,y_train)


# evaluate the neural network
print("Scoring Neural Network")
score = clf.score(X_test, y_test)
print("Accuracy of Neural Network")
print(score)

# visualize the dataset
print("Visualizing the dataset")
fig,ax = plt.subplots(5,10)

for i in range(5):
    for j in range(10):
        digit_image = S_X.iloc[(10*i)+j]
        ax[i,j].imshow(digit_image.to_numpy().reshape(28, 28), cmap='binary')
        y_predicted_title = str(clf.predict(digit_image.to_frame().transpose()))
        ax[i,j].title.set_text(y_predicted_title)

plt.show()

