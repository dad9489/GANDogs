from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import os


def load_model():
    with open(model_path+'models.json', 'r') as f:
        json_str = f.readline()

    model = model_from_json(json_str)
    model.load_weights(model_path+'generator_'+str(epoch)+'.h5')
    return model


def sample_images():
    gen_imgs = model.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c, figsize=(15, 15))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :,:])
            axs[i, j].axis('off')
            cnt += 1
    plt.show()
    plt.close()


def get_epoch(path):
    num = -1
    for filename in os.listdir(path):
        if filename[-3:] == '.h5':
            num = int(filename[-8:-3])
            break
    return num


if __name__ == '__main__':
    model_path = 'images/dog_gen/models/'
    epoch = get_epoch(model_path)
    r, c = 5, 5
    latent_dim = 100
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    model = load_model()
    sample_images()
