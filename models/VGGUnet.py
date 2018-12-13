from keras.models import *
from keras.layers import *

from keras.applications.vgg16 import VGG16


def VGGUnet(n_classes,
			input_height=256,
			input_width=256,
			opt='RMSprop',
			loss='categorical_crossentropy'):

	assert input_height%32 == 0
	assert input_width%32 == 0

	img_input = Input(shape=(input_height, input_width, 3))

	base_model = VGG16(input_tensor=img_input,
					   weights='imagenet',
					   include_top=False)

	o = base_model.get_layer('block5_pool').output

	o = (ZeroPadding2D( (1,1)))(o)
	o = (Conv2D(512, (3, 3), padding='valid'))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D( (2,2)))(o)
	o = (concatenate([o, base_model.get_layer('block4_pool').output],axis=3))
	o = (ZeroPadding2D( (1,1)))(o)
	o = (Conv2D(256, (3, 3), padding='valid'))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D( (2,2)))(o)
	o = (concatenate([o, base_model.get_layer('block3_pool').output],axis=3))
	o = (ZeroPadding2D( (1,1)))(o)
	o = (Conv2D(128, (3, 3), padding='valid'))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D( (2,2)))(o)
	o = (concatenate([o, base_model.get_layer('block2_pool').output],axis=3))
	o = (ZeroPadding2D((1,1)  ))(o)
	o = (Conv2D(64, (3, 3), padding='valid'  ) )(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D( (2,2)))(o)
	o = (concatenate([o, base_model.get_layer('block1_pool').output],axis=3))
	o = (ZeroPadding2D((1,1)))(o)
	o = (Conv2D(32, (3, 3), padding='valid'))(o)
	o = (BatchNormalization())(o)

	o = (UpSampling2D( (2,2)))(o)
	o = (Conv2D(n_classes, (3, 3), padding='same'))(o)
	o = (Activation('softmax'))(o)
	model = Model(img_input, o)

	for layer in base_model.layers:
		layer.trainable = False

	model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

	return model


