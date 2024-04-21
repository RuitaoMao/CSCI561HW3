import csv
import math
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import numpy as np


def read_dataset(path):
    field_TYPE = [
        "Co-op for sale",
        "House for sale",
        "Condo for sale",
        "Multi-family home for sale",
        "Townhouse for sale",
        "Pending",
        "Contingent",
        "Land for sale",
        "For sale",
        "Foreclosure",
    ]

    mapping_SUBLOCALITY = {
        "New York": "New York County",
        "Kings County": "Kings County",
        "Queens County": "Queens County",
        "Queens": "Queens County",
        "Richmond County": "Richmond County",
        "Brooklyn": "Kings County",
        "Bronx County": "Bronx County",
        "New York County": "New York County",
        "The Bronx": "Bronx County",
        "Staten Island": "Richmond County",
        "Manhattan": "New York County",
        "Riverdale": "Bronx County",
        "Flushing": "Queens County",
        "Coney Island": "Kings County",
        "East Bronx": "Bronx County",
        "Brooklyn Heights": "Kings County",
        "Jackson Heights": "Queens County",
        "Rego Park": "Queens County",
        "Fort Hamilton": "Kings County",
        "Dumbo": "Kings County",
        "Snyder Avenue": "Kings County",
    }

    field_SUBLOCALITY = [
        "New York County",
        "Kings County",
        "Queens County",
        "Richmond County",
        "Bronx County",
    ]

    with open(path, "r") as fin:
        reader = csv.reader(fin)
        rows = list(reader)
        header, rows = rows[0], rows[1:]

        res = []
        for row in rows:
            if not row:
                continue

            x = []

            index = header.index("TYPE")
            value = row[index]
            index = field_TYPE.index(value) if value in field_TYPE else len(field_TYPE)
            t = [0] * (len(field_TYPE) + 1)
            t[index] = 1
            x.extend(t)

            index = header.index("SUBLOCALITY")
            value = row[index]
            value = mapping_SUBLOCALITY.get(value, value)
            t = [0] * (len(field_SUBLOCALITY) + 1)
            for i, state in enumerate(field_SUBLOCALITY):
                if state == value:
                    t[i] = 1
            else:
                t[len(field_SUBLOCALITY)] = 1
            x.extend(t)

            index = header.index("PRICE")
            value = row[index]
            value = float(value)
            value = max(value, 1)
            value = math.log(value)
            x.append(value)

            index = header.index("BATH")
            value = row[index]
            value = float(value)
            value = max(value, 1)
            value = math.log(value)
            x.append(value)

            index = header.index("PROPERTYSQFT")
            value = row[index]
            value = float(value)
            value = max(value, 1)
            value = math.log(value)
            x.append(value)

            res.append(x)

    X = np.array(res)
    return X


def read_label(path):
    with open(path, "r") as fin:
        reader = csv.reader(fin)
        rows = list(reader)
        header, rows = rows[0], rows[1:]

        res = []
        for row in rows:
            if not row:
                continue

            value = row[0]
            value = float(value)

            value = max(value, 1)
            value = math.log(value)

            res.append(value)

    y = np.array(res)
    y = y.reshape(-1, 1)
    return y


def train_test_split(X, y, validation_split):
    assert X.shape[0] == y.shape[0]

    indices = np.random.permutation(X.shape[0])
    train_size = int(X.shape[0] * (1 - validation_split))

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


class MLPRegressor:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            # Kaiming uniform distribution
            self.weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i + 1])
                * np.sqrt(2 / layer_sizes[i])
            )
            self.biases.append(np.zeros(layer_sizes[i + 1]))

        self.best_loss = float("inf")
        self.best_epoch = None
        self.best_weights = None
        self.best_biases = None

    def leaky_relu(self, x):
        return np.maximum(0, x) + 1e-2 * np.minimum(0, x)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, 1e-2)

    def forward(self, X, use_best_weights=False):
        h = [None]
        L = [X]

        if use_best_weights:
            weights = self.best_weights
            biases = self.best_biases
        else:
            weights = self.weights
            biases = self.biases

        for i in range(len(weights)):
            W_ij = weights[i]
            V_j = L[-1]
            B_i = biases[i]
            g = self.leaky_relu

            h_i = np.dot(V_j, W_ij) + B_i
            h.append(h_i)

            L_i = g(h_i)
            L.append(L_i)

        return L, h

    def predict(self, X, use_best_weights=False):
        L, _ = self.forward(X, use_best_weights=use_best_weights)
        return L[-1]

    def loss(self, y, y_hat):
        # MSE(Y, Y_hat) = (1 / 2k) * Σ(Y - Y_hat)^2
        return (1 / 2) * np.mean((y - y_hat) ** 2)

    def gradient(self, X, y):
        # L: O_i, V_j, Xi_k
        # h: h_i, h_j, h_k
        L, h = self.forward(X)

        # ζ_i
        Zeta_i = y
        # O_i
        O_i = L[-1]
        # h_i
        h_i = h[-1]
        # g'
        g_prime = self.leaky_relu_derivative

        # dMSE/dY_hat = Y - Y_hat -> ζ_i - O_i
        # δ_i = [ζ_i - O_i] * g'(h_i)
        delta_i = ((Zeta_i - O_i) / X.shape[0]) * g_prime(h_i)

        SIGMAs = []
        deltas = [delta_i]
        # index of L: 0, 1, 2, 3
        # index of h:    1, 2, 3
        # index of W: 0, 1, 2
        # i: 2, 1, 0
        for i in range(len(self.weights) - 1, -1, -1):
            # δ_i
            delta_i = deltas[-1]
            # V_j
            V_j = L[i]
            # Σ_ij = V_j * δ_i
            SIGMA_ij = np.dot(V_j.T, delta_i)
            SIGMAs.append(SIGMA_ij)

            if i == 0:
                break

            # W_ij
            W_ij = self.weights[i]
            # h_j
            h_j = h[i]
            # δ_j = δ_i * W_ij * g'(h_j)
            delta_j = np.dot(delta_i, W_ij.T) * g_prime(h_j)
            deltas.append(delta_j)

        SIGMAs.reverse()
        deltas.reverse()
        return SIGMAs, deltas

    def update_weights(self, gradients, learning_rate):
        eta = learning_rate
        for i in range(len(self.weights)):
            SIGMA = gradients[i]
            self.weights[i] += eta * SIGMA
            self.biases[i] += eta * np.mean(SIGMA, axis=0)

    def fit(
            self,
            X,
            y,
            epochs,
            batch_size,
            learning_rate,
            validation_split=0.1,
            verbose=False,
    ):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, validation_split=validation_split
        )

        for epoch in range(1, epochs + 1):
            for batch_start in range(0, X_train.shape[0], batch_size):
                batch_X = X_train[batch_start : batch_start + batch_size]
                batch_y = y_train[batch_start : batch_start + batch_size]

                gradients, _ = self.gradient(batch_X, batch_y)
                current_lr = learning_rate * (1 - epoch / epochs)
                self.update_weights(gradients, current_lr)

            if self.best_epoch and epoch - self.best_epoch > 800:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            if epoch % 10 == 0:
                y_pred = self.predict(X_val)
                loss = self.loss(y_val, y_pred)
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_epoch = epoch
                    self.best_weights = self.weights.copy()
                    self.best_biases = self.biases.copy()
                    if verbose:
                        print(f"New best loss: {loss} at epoch {epoch}")
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")


X_train = read_dataset(f"train_data.csv")
y_train = read_label(f"train_label.csv")
X_test = read_dataset(f"test_data.csv")

filenames = ["test_data1.csv", "test_data2.csv", "test_data3.csv", "test_data4.csv", "test_data5.csv"]
X_tests = []

# Using a for loop to read each dataset and append it to X_tests
for filename in filenames:
    X_tests.append(read_dataset(filename))

# Unpack X_tests into individual variables
X_test1, X_test2, X_test3, X_test4, X_test5 = X_tests

hidden_sizes = [275,275]
learning_rate = 0.0075
epochs = 1500
batch_size = 16



# Creating and training MLP regressor
mlp_regressor = MLPRegressor(
    input_size=X_train.shape[-1],
    hidden_sizes=hidden_sizes,
    output_size=y_train.shape[-1],
)
mlp_regressor.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    verbose=True,
)

y_preds = []
# Predictions on test set
for i in range(1,6):

    y_preds.append(mlp_regressor.predict(X_tests[i-1], use_best_weights=True))


def write_label(path, y_pred):

    y_pred = np.round(np.power(np.e, y_pred))
    y_pred = y_pred.reshape(-1).astype(int)
    y_pred = y_pred.tolist()

    with open(path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["BEDS"])
        for value in y_pred:
            writer.writerow([value])

for i in range(1,6):
    write_label(f"output"+str(i)+".csv", y_preds[i-1])

def compare_csv(file1, file2):
    data1 = read_csv(file1)
    data2 = read_csv(file2)

    if len(data1) != len(data2):
        print("Error: Number of rows in the CSV files is different.")
        return None

    total_rows = len(data1)
    same_count = 0

    for i in range(total_rows):
        if data1[i] == data2[i]:
            same_count += 1

    accuracy = (same_count / total_rows) * 100
    return accuracy

def read_csv(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(int(row['BEDS']))
    return data

for i in range(1,6):
    file1_path = 'output'+str(i)+'.csv'
    file2_path = 'test_label'+str(i)+'.csv'

    accuracy = compare_csv(file1_path, file2_path)

    if accuracy is not None:
        print(f"Accuracy: {accuracy:.2f}%")