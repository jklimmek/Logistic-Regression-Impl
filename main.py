import numpy as np
from mnist import MNIST

# loading the data
data = MNIST(r'data')
images, labels = data.load_training()
im_test, lab_test = data.load_testing()

# set of hyperparameters for better results 
LEARNING_RATES = [0.001, 0.003, 0.01]
ITERATIONSS = [5, 10, 20]
BATCH_SIZES = [1, 32, 128]
BETAS = [0.9, 0.95]

# cleaning the data
def preprocess_data(x, y):
    print('preprocessing {} samples'.format(len(x)))
    indexes = []
    cleaned_labels = []
    for idx, val in enumerate(y):
        if val is not 0 and val is not 1:
            indexes.append(idx)
            if val in [2, 3, 5, 7]:
                cleaned_labels.append(1)  # prime number 1
            else:
                cleaned_labels.append(0)  # else   0

    cleaned_imgs = [x[i] for i in indexes]
    for image in cleaned_imgs:
        for idx in range(len(image)):
            image[idx] /= 255

    return cleaned_imgs, cleaned_labels

# actual model
class Model:

    def __init__(self):
        self.values = np.zeros((1, 785), dtype=float)

    def fit(self, x, y) -> None:
        for it in range(ITERATIONS):
            print('iteration no.: {} ...'.format(it+1))
            x_batches, y_batches = self.random_data_shuffle(x, y)
            n = len(x)
            x_batches = [x_batches[k:k + BATCH_SIZE] for k in range(0, n, BATCH_SIZE)]
            y_batches = [y_batches[k:k + BATCH_SIZE] for k in range(0, n, BATCH_SIZE)]
            for xx, yy in zip(x_batches, y_batches):
                update = np.zeros((785, 1))
                vt = np.zeros(self.values.shape)
                for number, label in zip(xx, yy):
                    number = np.insert(number, 0, 1).reshape(785, 1)
                    poly = np.dot(self.values, number)
                    val = self.sigmoid(poly)
                    update += self.cost_derivative(label, val, number)
                update = np.array(update).reshape(1, 785)/len(xx)
                for index in range(len(update)):
                    vt[index] = BETA*vt[index]+LEARNING_RATE*update[index]
                    self.values += update

    def predict(self, x) -> np.ndarray:
        print('predicting {} values ...'.format(len(x)))
        vals = []
        for number in x:
            number = np.insert(number, 0, 1).reshape(785, 1)
            poly = np.dot(self.values, number)
            val = self.sigmoid(poly)
            vals.append(val)
        return np.rint(vals)

    @staticmethod
    def evaluate(y_true, y_pred) -> float:
        counter = 0
        for x, y in zip(y_true, y_pred):
            counter += 1 if x == y else 0
        return counter/len(y_pred)

    @staticmethod
    def random_data_shuffle(x, y):
        index = [i for i in range(len(x))]
        np.random.shuffle(index)
        x_shuffled = [x[i] for i in index]
        y_shuffled = [y[i] for i in index]
        return x_shuffled, y_shuffled

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def cost_derivative(x, y, f):
        return (x - y)*f
    
    
# cleaning our dataset
images, labels = preprocess_data(images, labels)
im_test, lab_test = preprocess_data(im_test, lab_test)

# storing all results in .txt file
f = open('Results.txt', 'a')
for q in LEARNING_RATES:
    for w in ITERATIONSS:
        for e in BATCH_SIZES:
            for r in BETAS:
                LEARNING_RATE = q
                ITERATIONS = w
                BATCH_SIZE = e
                BETA = r
                m = Model()
                m.fit(images, labels)
                v = m.predict(im_test)
                score = m.evaluate(lab_test, v)
                f.write('#'*40)
                f.write('\nlearning rate: {}\niterations: {}\nbatch size: {}\nbeta: {}\nSCORE: {}\n'.format(q, w, e, r, score))
f.close()
