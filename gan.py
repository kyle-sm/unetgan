import math
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

NOISE_SIZE = 256
WARMUP_EPOCHS = 20
EPOCHS = 50
BATCH_SIZE = 25
LEARN_RATE_GENERATOR = 1e-4
LEARN_RATE_DISCRIMINATOR = 5e-4
CHANNEL_WIDTH = 64

generator_optimizer = tf.keras.optimizers.Adam(LEARN_RATE_GENERATOR)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARN_RATE_DISCRIMINATOR)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def main():
    discriminator = make_discriminator_model()
    # discriminator.summary()
    # tf.keras.utils.plot_model(discriminator, "u-net.png", show_shapes=True)
    generator = make_generator_model()
    # generator.summary()
    # tf.keras.utils.plot_model(generator, "deepgan.png", show_shapes=True)

    seed = int(time.time())
    # norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'dataset',
        # validation_split=0.2,
        # subset='training',
        seed=seed,
        image_size=(512, 512),
        batch_size=BATCH_SIZE)
    """
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'dataset',
        validation_split=0.2,
        subset='validation',
        seed=seed,
        image_size=(512, 512),
        batch_size=BATCH_SIZE)
    """

    train(train_ds, generator, discriminator)

    gen_builder = tf.saved_model.builder.SavedModelBuilder("gen_export")
    gen_inputs = {
        'input': tf.saved_model.utils.build_tensor_info(generator.input)
    }
    gen_outputs = {
        'image': tf.saved_model.utils.build_tensor_info(generator.output)
    }
    gen_sig = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=gen_inputs,
        outputs=gen_outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    gen_builder.add_meta_graph_variables(
        tf.keras.backend.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.tag_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            gen_sig
        })
    gen_builder.save()

    disc_builder = tf.saved_model.builder.SavedModelBuilder("disc_export")
    disc_inputs = {
        'input': tf.saved_model.utils.build_tensor_info(discriminator.input)
    }
    disc_outputs = {
        'image': tf.saved_model.utils.build_tensor_info(discriminator.output)
    }
    disc_sig = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=disc_inputs,
        outputs=disc_outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    disc_builder.add_meta_graph_variables(
        tf.keras.backend.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.tag_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            disc_sig
        })
    disc_builder.save()


@tf.function
def train_step(image_batch, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, NOISE_SIZE, 1, 1])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output, real_pixel = discriminator(image_batch, training=True)
        fake_output, fake_pixel = discriminator(generated_images,
                                                training=True)

        gen_loss = generator_loss(fake_output, fake_pixel)
        disc_loss = discriminator_loss(real_output, real_pixel, fake_output,
                                       fake_pixel)
    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss,
                                   discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gen_grad, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(disc_grad, discriminator.trainable_variables))


def train(dataset, generator, discriminator):
    print('start training.')
    for epoch in range(EPOCHS):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, generator, discriminator)

        print(f'Time for epoch {epoch - 1} is {time.time()-start}\n')


def generator_loss(fake_output, fake_pixel):
    out_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    pixel_loss = cross_entropy(tf.ones_like(fake_pixel), fake_pixel)
    return out_loss + pixel_loss


def discriminator_loss(fake_output, fake_pixel, real_output, real_pixel):
    out_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    pixel_loss = cross_entropy(tf.zeros_like(fake_pixel), fake_pixel)
    real_out_loss = cross_entropy(tf.ones_like(real_output), real_output)
    real_pixel_loss = cross_entropy(tf.ones_like(real_pixel), real_pixel)
    return out_loss + pixel_loss + real_out_loss + real_pixel_loss


def make_generator_model():
    """Generates the model for the generator."""
    gen_inputs = tf.keras.Input(shape=(NOISE_SIZE, 1, 1))
    model = layers.Dense(CHANNEL_WIDTH)(gen_inputs)
    model = layers.Reshape((4, 4, 16 * CHANNEL_WIDTH))(model)
    model = res_block(gen_inputs, model, 16 * CHANNEL_WIDTH)
    model = res_block_up(gen_inputs, model, 16 * CHANNEL_WIDTH)
    model = res_block(gen_inputs, model, 16 * CHANNEL_WIDTH)
    model = res_block_up(gen_inputs, model, 8 * CHANNEL_WIDTH)
    model = res_block(gen_inputs, model, 8 * CHANNEL_WIDTH)
    model = res_block_up(gen_inputs, model, 8 * CHANNEL_WIDTH)
    model = res_block(gen_inputs, model, 8 * CHANNEL_WIDTH)
    model = res_block_up(gen_inputs, model, 4 * CHANNEL_WIDTH)
    # TODO: Nonlocal block(model)
    model = res_block(gen_inputs, model, 4 * CHANNEL_WIDTH)
    model = res_block_up(gen_inputs, model, 2 * CHANNEL_WIDTH)
    model = res_block(gen_inputs, model, 2 * CHANNEL_WIDTH)
    model = res_block_up(gen_inputs, model, CHANNEL_WIDTH)
    model = res_block(gen_inputs, model, CHANNEL_WIDTH)
    model = res_block_up(gen_inputs, model, CHANNEL_WIDTH)
    model = layers.Conv2D(3, (3, 3), padding='same')(model)
    model = layers.Activation('tanh')(model)

    return tf.keras.Model(gen_inputs, model, name='generator')


def make_discriminator_model():
    """Generates the model for the discriminator. Based on U-Net
    architecture"""
    img_inputs = tf.keras.Input(shape=(512, 512, 3))
    encode1 = layers.Conv2D(64, (3, 3), activation='relu',
                            padding='same')(img_inputs)
    encode1 = layers.Conv2D(64, (3, 3), activation='relu',
                            padding='same')(encode1)

    encode2 = layers.MaxPool2D(pool_size=(2, 2))(encode1)
    encode2 = layers.Conv2D(128, (3, 3), activation='relu',
                            padding='same')(encode2)
    encode2 = layers.Conv2D(128, (3, 3), activation='relu',
                            padding='same')(encode2)

    encode3 = layers.MaxPool2D(pool_size=(2, 2))(encode2)
    encode3 = layers.Conv2D(256, (3, 3), activation='relu',
                            padding='same')(encode3)
    encode3 = layers.Conv2D(256, (3, 3), activation='relu',
                            padding='same')(encode3)

    encode4 = layers.MaxPool2D(pool_size=(2, 2))(encode3)
    encode4 = layers.Conv2D(512, (3, 3), activation='relu',
                            padding='same')(encode4)
    encode4 = layers.Conv2D(512, (3, 3), activation='relu',
                            padding='same')(encode4)

    encode5 = layers.MaxPool2D(pool_size=(2, 2))(encode4)
    encode5 = layers.Conv2D(1024, (3, 3), activation='relu',
                            padding='same')(encode5)
    encode5 = layers.Conv2D(1024, (3, 3), activation='relu',
                            padding='same')(encode5)

    encode_output = layers.Flatten()(encode5)
    encode_output = layers.Dense(1)(encode_output)

    decode1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2))(encode5)
    decode1 = layers.concatenate([decode1, encode4])
    decode1 = layers.Conv2D(512, (3, 3), activation='relu',
                            padding='same')(decode1)
    decode1 = layers.Conv2D(512, (3, 3), activation='relu',
                            padding='same')(decode1)

    decode2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2))(decode1)
    decode2 = layers.concatenate([decode2, encode3])
    decode2 = layers.Conv2D(256, (3, 3), activation='relu',
                            padding='same')(decode2)
    decode2 = layers.Conv2D(256, (3, 3), activation='relu',
                            padding='same')(decode2)

    decode3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2))(decode2)
    decode3 = layers.concatenate([decode3, encode2])
    decode3 = layers.Conv2D(128, (3, 3), activation='relu',
                            padding='same')(decode3)
    decode3 = layers.Conv2D(128, (3, 3), activation='relu',
                            padding='same')(decode3)

    decode4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2))(decode3)
    decode4 = layers.concatenate([decode4, encode1])
    decode4 = layers.Conv2D(64, (3, 3), activation='relu',
                            padding='same')(decode4)
    decode4 = layers.Conv2D(64, (3, 3), activation='relu',
                            padding='same')(decode4)

    decode_output = layers.Conv2D(1, (1, 1))(decode4)

    return tf.keras.Model(img_inputs, [encode_output, decode_output],
                          name='discriminator')


def res_block_up(z, input, out_channels):
    """A residual block for the generator. This one upsamples."""
    input_channels = input.shape[3]
    bottleneck_channels = math.ceil(input_channels / 4)

    skip = layers.Conv2D(out_channels, (1, 1), padding='same')(input)
    skip = layers.UpSampling2D()(skip)

    # skip_z = layers.Reshape((1, 1, input_channels))(z)
    skip_z1 = layers.Flatten()(z)
    skip_z1 = layers.Dense(input_channels)(skip_z1)
    skip_z1 = layers.Reshape((1, 1, input_channels))(skip_z1)

    res = layers.BatchNormalization()(input)
    res = layers.ReLU()(res)
    # reduce channels by a factor of 4
    res = layers.Conv2D(bottleneck_channels, (1, 1), padding='same')(res)
    res = layers.BatchNormalization()(res)
    res = layers.ReLU()(res)
    res = layers.UpSampling2D()(res)
    # maintain channels
    res = layers.Conv2D(bottleneck_channels, (3, 3), padding='same')(res)
    res = layers.BatchNormalization()(res)
    res = layers.ReLU()(res)
    res = layers.Conv2D(bottleneck_channels, (3, 3), padding='same')(res)
    res = layers.BatchNormalization()(res)
    res = layers.ReLU()(res)
    res = layers.Conv2D(out_channels, (1, 1), padding='same')(res)

    return layers.add([res, skip])


def res_block(z, input, out_channels):
    """A residual block for the generator. This one upsamples."""
    input_channels = input.shape[3]
    bottleneck_channels = math.ceil(input_channels / 4)

    skip = input

    # skip_z = layers.Reshape((1, 1, input_channels))(z)

    res = layers.BatchNormalization()(input)
    res = layers.ReLU()(res)
    # reduce channels by a factor of 4
    res = layers.Conv2D(bottleneck_channels, (1, 1), padding='same')(res)
    res = layers.BatchNormalization()(res)
    res = layers.ReLU()(res)
    # maintain channels
    res = layers.Conv2D(bottleneck_channels, (3, 3), padding='same')(res)
    res = layers.BatchNormalization()(res)
    res = layers.ReLU()(res)
    res = layers.Conv2D(bottleneck_channels, (3, 3), padding='same')(res)
    res = layers.BatchNormalization()(res)
    res = layers.ReLU()(res)
    res = layers.Conv2D(out_channels, (1, 1), padding='same')(res)

    return layers.add([res, skip])


if __name__ == '__main__':
    main()
