from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Add, Activation, Concatenate, Input
from tensorflow.keras.models import Model
import tensorflow as tf


def res_block(x,sz,filter_sz=3,in_conv_size=1):
	xi  = x
	for i in range(in_conv_size):
		xi  = Conv2D(sz, filter_sz, activation='linear', padding='same')(xi)
		xi  = BatchNormalization()(xi)
		xi 	= Activation('relu')(xi)
	xi  = Conv2D(sz, filter_sz, activation='linear', padding='same')(xi)
	xi  = BatchNormalization()(xi)
	xi 	= Add()([xi,x])
	xi 	= Activation('relu')(xi)
	return xi

def conv_batch(_input,fsz,csz,activation='relu',padding='same',strides=(1,1)):
	output = Conv2D(fsz, csz, activation='linear', padding=padding, strides=strides)(_input)
	output = BatchNormalization()(output)
	output = Activation(activation)(output)
	return output


def build_head(x):
	xprobs    = conv_batch(x, 64, 3, activation='relu')
	xprobs    = conv_batch(xprobs, 32, 3, activation='linear')
	xprobs    = Conv2D(1, 3, activation='sigmoid', padding='same',  kernel_initializer = 'he_uniform')(xprobs)
	xbbox    = conv_batch(x, 64, 3, activation='relu')
	xbbox    = conv_batch(xbbox, 32, 3, activation='linear')
	xbbox     = Conv2D(6, 3, activation='linear' , padding='same',  kernel_initializer = 'he_uniform')(xbbox)
	return Concatenate(3)([xprobs,xbbox])

def get_backbone(name='ResNet50'):
	outs=[]
	backbone = None
	if name == 'ResNet50':
		backbone = tf.keras.applications.resnet50.ResNet50(
			include_top=False, input_shape=[None, None, 3]
		)
		outs = [
			backbone.get_layer(layer_name).output for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
		]
	return tf.keras.Model(
		inputs=backbone.inputs, outputs=outs
	)
	
class FeaturePyramid(tf.keras.layers.Layer):
	"""Builds the Feature Pyramid with the feature maps from the backbone.
	Attributes:
	num_classes: Number of classes in the dataset.
	backbone: The backbone to build the feature pyramid from.
		Currently supports ResNet50 only.
	"""
	def __init__(self, backboneName='ResNet50', **kwargs):
		super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
		self.backbone = get_backbone(backboneName)
		self.conv_c3_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
		self.conv_c4_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
		self.conv_c5_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
		self.conv_c3_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
		self.conv_c4_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
		self.conv_c5_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
		self.conv_c6_3x3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
		self.conv_c7_3x3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
		self.upsample_2x = tf.keras.layers.UpSampling2D(2)
	def call(self, images, training=False):
		c3_output, c4_output, c5_output = self.backbone(images, training=training)
		p3_output = self.conv_c3_1x1(c3_output)
		p4_output = self.conv_c4_1x1(c4_output)
		p5_output = self.conv_c5_1x1(c5_output)
		p4_output = p4_output + self.upsample_2x(p5_output)
		p3_output = p3_output + self.upsample_2x(p4_output)
		p3_output = self.conv_c3_3x3(p3_output)
		p4_output = self.conv_c4_3x3(p4_output)
		p5_output = self.conv_c5_3x3(p5_output)
		p6_output = self.conv_c6_3x3(c5_output)
		p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
		return p3_output, p4_output, p5_output

class IWpod_Net(tf.keras.Model):
	def __init__(self, backboneName='ResNet50', **kwargs):
		super(IWpod_Net, self).__init__(name="IWpod_Net", **kwargs)
		self.fpn = FeaturePyramid(backboneName)

	def call(self, image, training=False):
		features = self.fpn(image, training=training)
		box_outputs = []
		for feature in features:
			box_outputs.append(tf.reshape(build_head(feature), [tf.shape(image)[0], -1, 7]))

		box_outputs = tf.concat(box_outputs, axis=1)
		tf.reshape(box_outputs, [tf.shape(image)[0], -1, -1, 7])
		return box_outputs



def create_model_iwpodnet():
	#
	#  Creates additonal layers to discriminate the tasks of detection and
	#  localization. Can freeze common layers and train specialized layers
	#  separately
	#
	
	input_layer = Input(shape=(None,None,3),name='input')
	
	x = conv_batch(input_layer, 16, 3)
	x = conv_batch(x, 16, 3)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = conv_batch(x, 32, 3)
	x = res_block(x, 32)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = conv_batch(x, 64, 3)
	x = res_block(x,64)
	x = res_block(x,64)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = conv_batch(x, 64, 3)
	x = res_block(x,64)
	x = res_block(x,64)
	x = MaxPooling2D(pool_size=(2,2))(x)
	x = conv_batch(x, 128, 3)
	x = res_block(x,128)
	x = res_block(x,128)
	x = res_block(x,128)
	x = res_block(x,128)
	x = build_head(x)

	return Model(inputs=input_layer,outputs=x)


if __name__ == '__main__':

	model = create_model_iwpodnet()
	print ('Finished')

