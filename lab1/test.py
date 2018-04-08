def skimage_test():
    from skimage import data, io, filters
    image = data.coins()
    # ... or any other NumPy array!
    edges = filters.sobel(image)
    io.imshow(edges)
    io.show()


def sklearn_test():
    from sklearn import datasets
    from sklearn.model_selection import cross_val_predict
    from sklearn import linear_model
    import matplotlib.pyplot as plt
    lr = linear_model.LinearRegression()
    boston = datasets.load_boston()
    y = boston.target

    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:
    predicted = cross_val_predict(lr, boston.data, y, cv=10)

    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


def opencv_test():
    import cv2
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def keras_test():
    from keras.models import Sequential

    model = Sequential()

    from keras.layers import Dense, Activation

    model.add(Dense(output_dim=64, input_dim=100))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    from keras.optimizers import SGD
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))


def tensorflow_test():
    import tensorflow as tf
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    sess.run(hello)
    a = tf.constant(10)
    b = tf.constant(32)
    sess.run(a + b)


import numpy as np

print("Hello openCV!")
skimage_test()
sklearn_test()
opencv_test()
# keras_test()
# tensorflow_test()
