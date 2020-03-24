# %%
import os
import numpy as np
import tensorflow as tf
import dolhasz as ad
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = [10, 5]
import matplotlib
matplotlib.use('TkAgg')
# %matplotlib inline


# %% DATA
def load_xue(path='C:/Users/aland_000/Documents/DeepCQA/imgs_test'):
    comps = []
    for f in os.listdir(path):
        c = ad.image.read(
            os.path.join(path, f, 'ori.jpg'), 
            resize=(224,224), 
            dtype='float'
            )
        comps.append(c*2-1)
    return comps


# %% MODELS

def load_cvpr_model(trim_layer='concatenate_1'):
    model = tf.keras.models.load_model('data/pretrained_classifier_fold_5.hdf5', compile=False)
    first_conv_output = model.get_layer(trim_layer).output
    short_model = tf.keras.Model(model.input, first_conv_output)
    print(model.summary())
    return model, short_model


def visualise_activations(image_list, model, short_model):
    for c in image_list:
        p_b = ad.image.squeeze(model.predict(ad.image.stretch(c)))
        p = ad.image.squeeze(short_model.predict(ad.image.stretch(c)))
        f, ax = plt.subplots(1,2)
        ax[0].imshow((c+1)/2)
        ax[1].imshow(p_b, vmin=0.0, vmax=1.0)

        f, ax = plt.subplots(
            np.round(5), 
            np.round(3), figsize=(15,9)
        )
        for i, a in enumerate(ax):
            for ii, aa in enumerate(a):
                idx = i*len(a) + ii
                if idx < p.shape[-1]:
                    aa.imshow(p[:,:,idx])
        plt.show()
        # break


if __name__ == "__main__":
    images = load_xue()
    model, short_model = load_cvpr_model()
    visualise_activations(images, model, short_model)
    

# %%
