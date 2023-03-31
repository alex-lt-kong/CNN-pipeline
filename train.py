from tensorflow import keras

import utils
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd

import shutil
import sys
import tensorflow as tf



def preview_samples(dest_dir: str, dataset: tf.data.Dataset, data_augmentation):

    count = {
        '0': 0,
        '1': 0
    }
    dataset = dataset.shuffle(buffer_size=128)
    for images, labels in dataset:
        augmented_images = data_augmentation(images)
        for i in range(len(augmented_images)):
            label = str(labels[i].numpy())
            count[label] += 1
            label_dir = os.path.join(dest_dir, label)
            if os.path.isdir(label_dir) is False:
                os.mkdir(label_dir)

            tf.keras.utils.save_img(
                os.path.join(label_dir, f'{count[label]}.jpg'), augmented_images[i].numpy().astype("uint8")
            )

        enough_sample = 0
        for key in count.keys():
            if count[key] > 5:
                enough_sample += 1
        if enough_sample >= len(count.keys()):
            break


def main():
    utils.set_environment_vars()
    settings = utils.read_config_file()
    utils.initialize_logger()
    sys.path.insert(1, settings['model']['path'])
    import definition
    
    image_size = definition.target_image_size


    
    logging.info(settings)
    logging.info('Checking valid GPUs')
    utils.check_gpu()
    #logging.info('Removing invalid samples, this could take a while...')
    #utils.remove_invalid_samples(settings['dataset']['path'])
    logging.info('Separating data into a training set and a test set')
    train_ds, val_ds = utils.prepare_dataset(settings['dataset']['path'], image_size=image_size, batch_size=definition.batch_size)

    func = definition.data_augmentation()
    logging.info('Saving some samples as preview')
    preview_samples(
        dest_dir=settings['dataset']['preview_save_to'],
        dataset=train_ds,
        data_augmentation=func)
    
    logging.info('calling make_model()')
    model = definition.make_model(input_shape=image_size + (3,), data_augmentation=func, num_classes=2)
    # + (1,) for grayscale, + (3,) for rgb
    model.build((None,) + image_size + (3,))
    # https://stackoverflow.com/questions/55908188/this-model-has-not-yet-been-built-error-on-model-summary
    keras.utils.plot_model(
        model, show_shapes=True, to_file=settings['model']['save_to']['model_plot']
    )

    with open(settings['model']['save_to']['summary'], 'w') as f:        
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )


    history = model.fit(train_ds, epochs=definition.epochs, validation_data=val_ds)
    # epochs: an epoch is an iteration over the entire x and y data provided
    # (unless the steps_per_epoch flag is set to something other than None)
    
    if os.path.isdir(settings['model']['save_to']['model']):
        shutil.rmtree(settings['model']['save_to']['model'])
    
    tf.keras.models.save_model(model=model, filepath=settings['model']['save_to']['model'])
    df = pd.DataFrame(data=history.history)
    df['accuracy_ma'] = df['accuracy'].rolling(window=5).mean()
    df['val_accuracy_ma'] = df['val_accuracy'].rolling(window=5).mean()
    plt.figure(figsize = (16/2, 9/2))
    plt.rcParams.update({'font.size': 15})
    #fig = df[['accuracy', 'accuracy_ma', 'val_accuracy', 'val_accuracy_ma']].plot(kind='line', figsize=(16, 9/2), fontsize=12).get_figure()
    plt.plot(df['accuracy'],     linewidth=1.75, label="accuracy",     color='C0')    
    plt.plot(df['val_accuracy'], linewidth=1.75, label="val_accuracy", color='C1')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(settings['model']['save_to']['history_plot'], bbox_inches='tight')

    df.to_csv(settings['model']['save_to']['history'])
    

if __name__ == '__main__':
    main()