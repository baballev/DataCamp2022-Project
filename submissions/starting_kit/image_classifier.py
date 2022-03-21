from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense

filtered_categories = ["Beverages", "Sweet snacks", "Dairies", "Cereals and potatoes", "Meats", "Fermented foods",
                       "Fermented milk products",
                       "Groceries", "Meals", "Cereals and their products", "Cheeses", "Sauces", "Spreads",
                       "Confectioneries", "Prepared meats",
                       "Frozen foods", "Breakfasts", "Desserts", "Canned foods", "Seafood", "Cocoa and its products",
                       "Fats", "Condiments",
                       "Fishes", "Breads", "Yogurts", "Cakes", "Biscuits", "Pastas", "Legumes"]
filtered_categories = [s.lower() for s in filtered_categories]
nb_classes = len(filtered_categories)

CLASS_TO_INDEX = {filtered_categories[i]: i for i in range(len(filtered_categories))}
INDEX_TO_CLASS = {i: filtered_categories[i] for i in range(len(filtered_categories))}

# convolution kernel size
kernel_size = (3, 3)
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)


class ImageClassifier:
    def __init__(self, img_rows, img_cols):
        input_shape1 = (img_rows, img_cols, 3)
        input_shape2 = (img_rows, img_cols, nb_filters)
        input_shape3 = (img_rows // 2, img_cols // 2, nb_filters)

        self.model = Sequential()
        self.model.add(Conv2D(filters=nb_filters, kernel_size=kernel_size, input_shape=input_shape1, padding='same'))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(filters=nb_filters, kernel_size=kernel_size, input_shape=input_shape2, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2), padding='same'))

        self.model.add(Conv2D(filters=nb_filters, kernel_size=kernel_size, input_shape=input_shape3, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=pool_size, strides=(2, 2), padding='same'))

        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(nb_classes))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, data_loader, batch_size, n_epochs):
        X_train = []
        Y_train = []
        for data in data_loader:
            img, labels = data
            X_train.append(img)
            Y_train.append(labels)
        print(self.model.summary())
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epochs)

        return self

    def predict_proba(self, data_loader):
        Y_pred = []
        for data in data_loader:
            img, labels = data
            Y_pred.append((self.model(img)))
        return Y_pred
