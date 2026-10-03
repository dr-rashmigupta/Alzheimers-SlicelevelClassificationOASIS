"""Training engine shared by the VGG19 and ResNet50 entry points."""
from __future__ import annotations
import numpy as np
from mri_common import prepare, training_records, load_image, class_weights, report, save_json


def run(args, architecture):
    root, splits = prepare(args)
    if args.prepare_only:
        return
    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    keras.mixed_precision.set_global_policy('mixed_float16' if args.mixed_precision and tf.config.list_physical_devices('GPU') else 'float32')
    size = 224 if architecture == 'vgg19' else 160
    records = training_records(splits['train'], args.seed, not args.no_noise)
    weights = class_weights(records, not args.no_class_weights)
    save_json(args.output_dir / 'class_weights.json', weights.tolist())
    preprocess = keras.applications.vgg19.preprocess_input if architecture == 'vgg19' else keras.applications.resnet50.preprocess_input

    class Batches(keras.utils.PyDataset):
        def __init__(self, rows, training=False):
            super().__init__()
            self.rows, self.training = rows, training
            self.order = np.arange(len(rows))
            self.rng = np.random.default_rng(args.seed)
            if training:
                self.rng.shuffle(self.order)

        def __len__(self):
            return (len(self.rows) + args.batch_size - 1) // args.batch_size

        def __getitem__(self, index):
            rows = [self.rows[i] for i in self.order[index * args.batch_size:(index + 1) * args.batch_size]]
            x = np.stack([load_image(root, r, size, args.seed, args.noise_amount) for r in rows]).astype(np.float32)
            y = keras.utils.to_categorical([r['label'] for r in rows], num_classes=4)
            return x, y

        def on_epoch_end(self):
            if self.training:
                self.rng.shuffle(self.order)

    backbone_fn = keras.applications.VGG19 if architecture == 'vgg19' else keras.applications.ResNet50
    backbone = backbone_fn(weights=None if args.no_pretrained else 'imagenet', include_top=False, input_shape=(size, size, 3))
    for i, layer in enumerate(backbone.layers):
        layer.trainable = architecture == 'resnet50' and i >= len(backbone.layers) - 10
    inputs = keras.Input((size, size, 3), dtype='float32')
    x = inputs
    if architecture == 'resnet50':
        x = keras.Sequential([
            keras.layers.RandomRotation(10 / 360, fill_mode='nearest', seed=args.seed),
            keras.layers.RandomTranslation(.05, .05, fill_mode='nearest', seed=args.seed + 1),
            keras.layers.RandomZoom(.05, .05, fill_mode='nearest', seed=args.seed + 2),
        ], name='spatial_augmentation')(x)
        # Multiplicative brightness matches the original 0.9–1.1 range.
        class Brightness(keras.layers.Layer):
            def call(self, values, training=None):
                if training:
                    scales = tf.random.uniform((tf.shape(values)[0], 1, 1, 1), .9, 1.1, dtype=values.dtype)
                    return tf.clip_by_value(values * scales, 0, 255)
                return values
        x = Brightness(name='brightness')(x)
    # Application-specific preprocessing expects RGB intensities in [0,255].
    x = keras.layers.Lambda(preprocess, name='imagenet_preprocessing')(x)
    x = backbone(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(.3)(x)
    regularizer = keras.regularizers.l2(.001) if architecture == 'resnet50' else None
    x = keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizer)(x)
    if architecture == 'resnet50':
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(.3)(x)
    outputs = keras.layers.Dense(4, activation='softmax', dtype='float32')(x)
    model = keras.Model(inputs, outputs, name=architecture)
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = args.output_dir / 'best.weights.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True, save_weights_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(str(args.output_dir / 'training_log.csv')),
    ]
    if architecture == 'resnet50':
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=4, min_lr=1e-6))
    history = model.fit(Batches(records, True), validation_data=Batches(splits['val']), epochs=args.epochs,
                        class_weight={i: float(w) for i, w in enumerate(weights)}, callbacks=callbacks)
    if not checkpoint.exists():
        raise RuntimeError('No finite validation checkpoint was saved')
    model.load_weights(checkpoint)
    test = Batches(splits['test'])
    save_json(args.output_dir / 'test_evaluation.json', model.evaluate(test, return_dict=True, verbose=0))
    report(args.output_dir, splits['test'], model.predict(test, verbose=0), history.history)
    # Export a self-contained inference SavedModel; augmentation is disabled at inference.
    model.export(str(args.output_dir / 'inference_savedmodel'))
