from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam


def create_model(model_path, mi=512, ni=512, loss_function='mse'):

    input_img = Input(shape=(mi, ni, 1))

    # network definition
    c1e = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1e)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)

    c2e = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2e)
    p2 = MaxPooling2D((2, 2), padding='same')(c2)

    c3e = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3e)
    p3 = MaxPooling2D((2, 2), padding='same')(c3)

    c4e = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4e)
    p4 = MaxPooling2D((2, 2), padding='same')(c4)

    c5e = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5e)

    u4 = UpSampling2D((2, 2), interpolation='bilinear')(c5)
    a4 = Concatenate(axis=3)([u4, c4])
    c6e = Conv2D(256, (3, 3), activation='relu', padding='same')(a4)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6e)

    u3 = UpSampling2D((2, 2), interpolation='bilinear')(c6)
    a3 = Concatenate(axis=3)([u3, c3])
    c7e = Conv2D(128, (3, 3), activation='relu', padding='same')(a3)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7e)

    u2 = UpSampling2D((2, 2), interpolation='bilinear')(c7)
    a2 = Concatenate(axis=3)([u2, c2])
    c8e = Conv2D(64, (3, 3), activation='relu', padding='same')(a2)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8e)

    u1 = UpSampling2D((2, 2), interpolation='bilinear')(c8)
    a1 = Concatenate(axis=3)([u1, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(a1)

    c10 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    markers = Conv2D(2, (1, 1), activation='softmax', padding='same')(c10)
    cell_mask = Conv2D(2, (1, 1), activation='softmax', padding='same')(c10)
    output = Concatenate(axis=3)([markers, cell_mask])

    model = Model(input_img, output)
    model.compile(optimizer=Adam(lr=0.0001), loss=loss_function)

    print ('Model was created')

    model.load_weights(model_path)

    return model


def create_model_bf(model_path, mi=512, ni=512, loss_function='mse'):

    input_img = Input(shape=(mi, ni, 1))

    # network definition
    c1e = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1e)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)

    c2e = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2e)
    p2 = MaxPooling2D((2, 2), padding='same')(c2)

    c3e = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3e)
    p3 = MaxPooling2D((2, 2), padding='same')(c3)

    c4e = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4e)

    u3 = UpSampling2D((2, 2), interpolation='bilinear')(c6)
    a3 = Concatenate(axis=3)([u3, c3])
    c7e = Conv2D(128, (3, 3), activation='relu', padding='same')(a3)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7e)

    u2 = UpSampling2D((2, 2), interpolation='bilinear')(c7)
    a2 = Concatenate(axis=3)([u2, c2])
    c8e = Conv2D(64, (3, 3), activation='relu', padding='same')(a2)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8e)

    u1 = UpSampling2D((2, 2), interpolation='bilinear')(c8)
    a1 = Concatenate(axis=3)([u1, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(a1)

    c10 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    markers = Conv2D(2, (1, 1), activation='softmax', padding='same')(c10)
    cell_mask = Conv2D(2, (1, 1), activation='softmax', padding='same')(c10)
    output = Concatenate(axis=3)([markers, cell_mask])

    model = Model(input_img, output)
    model.compile(optimizer=Adam(lr=0.0001), loss=loss_function)

    print ('Model was created')

    model.load_weights(model_path)

    return model

