from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Add, Activation, Concatenate, Input
from tensorflow.keras.models import Model
import tensorflow as tf


def autopad(k, p=None):  # kernel, padding
  # Pad to 'same'
  if p is None:
      p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p

def Conv(filters, kernel_size, strides=(1, 1), padding=None):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.ZeroPadding2D(autopad(kernel_size,padding)))
  model.add(tf.keras.layers.Conv2D(filters,kernel_size,strides))
  model.add(tf.keras.layers.BatchNormalization())
  # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
  model.add(tf.keras.layers.Activation('swish'))
  return model

def MP(_kernel_size=(2, 2)):
  mp = tf.keras.layers.MaxPool2D(pool_size=_kernel_size, strides=_kernel_size)
  return mp

def Concat(dimension=-1):
  concat = tf.keras.layers.Concatenate(axis=dimension)
  return concat

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
	concat = Concatenate(3)([xprobs,xbbox])
	return concat

def get_backbone(name='ResNet50'):
	outs=[]
	backbone = None
	if name == 'ResNet50':
		backbone = tf.keras.applications.resnet50.ResNet50(
			include_top=False, input_shape=[None, None, 3], weights=None
		)
		outs = [
			backbone.get_layer("conv4_block6_out").output
		]

	elif name == 'MobileNetV3Small':
		backbone = tf.keras.applications.MobileNetV3Small(
			include_top=False, input_shape=[None, None, 3], alpha=0.75, weights=None
		)
		outs = [
			backbone.layers[159].output
		]

	elif name == 'MobileNetV3Large':
		backbone = tf.keras.applications.MobileNetV3Large(
			include_top=False, input_shape=[None, None, 3], alpha=0.75, weights=None
		)
		outs = [
			backbone.layers[198].output
		]

	elif name == 'Yolov7':
		input = tf.keras.Input((None,None,3))
		a = Conv(32, 3, 1)(input)
		a = Conv(64, 3, 2)(a)
		a = Conv(64, 3, 1)(a)  
		b = Conv(128, 3, 2)(a)
		d = Conv(64, 1, 1)(b)
		c = Conv(64, 1, 1)(b)
		a = Conv(64, 3, 1)(c)
		b = Conv(64, 3, 1)(a)
		a = Conv(64, 3, 1)(b)
		a = Conv(64, 3, 1)(a)
		concat_1 = Concat()([a,b,c,d])

		a = Conv(256, 1, 1)(concat_1)
		mp_1 = MP()(a)
		b = Conv(128, 1, 1)(mp_1)
		a = Conv(128, 1, 1)(a)
		a = Conv(128, 3, 2)(a)
		concat_2 = Concat()([a,b])

		d = Conv(128, 1, 1)(concat_2)
		c = Conv(128, 1, 1)(concat_2)
		a = Conv(128, 3, 1)(c)
		b = Conv(128, 3, 1)(a)
		a = Conv(128, 3, 1)(b)
		a = Conv(128, 3, 1)(a)
		concat_3 = Concat()([a,b,c,d])

		b = Conv(512, 1, 1)(concat_3)
		mp_2 = MP()(b)
		c = Conv(256, 1, 1)(mp_2)
		a = Conv(256, 1, 1)(b)
		a = Conv(256, 3, 2)(a)
		concat_4 = Concat()([a,c])

		d = Conv(256, 1, 1)(concat_4)
		c = Conv(256, 1, 1)(concat_4)
		a = Conv(256, 3, 1)(c)
		b = Conv(256, 3, 1)(a)
		a = Conv(256, 3, 1)(b)
		a = Conv(256, 3, 1)(a)
		concat_5 = Concat()([a,b,c,d])
		a = Conv(1024, 1, 1)(concat_5)
		backbone = tf.keras.Model(inputs=input,outputs=a)
		outs = backbone.outputs

	elif name == 'Original':
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
		backbone = tf.keras.Model(inputs=input_layer,outputs=x)
		outs=backbone.outputs
	return tf.keras.Model(
		inputs=backbone.inputs, outputs=outs
	)
	
def buildFPN(input, backboneName='ResNet50'):
	backbone = get_backbone(backboneName)
	c3_output, c4_output, c5_output = backbone(input)
	p3_output = tf.keras.layers.Conv2D(256, 1, 1, "same")(c3_output)
	p4_output = tf.keras.layers.Conv2D(256, 1, 1, "same")(c4_output)
	p5_output = tf.keras.layers.Conv2D(256, 1, 1, "same")(c5_output)
	p4_output = p4_output + tf.keras.layers.UpSampling2D(2)(p5_output)
	p3_output = p3_output + tf.keras.layers.UpSampling2D(2)(p4_output)
	p3_output = tf.keras.layers.Conv2D(256, 3, 1, "same")(p3_output)
	p4_output = tf.keras.layers.Conv2D(256, 3, 1, "same")(p4_output)
	p5_output = tf.keras.layers.Conv2D(256, 3, 1, "same")(p5_output)
	
	return p3_output, p4_output, p5_output

def buildModel(backboneName='ResNet50'):
	input_layer = Input(shape=(None,None,3),name='input')
	backbone = get_backbone(backboneName)
	x = backbone(input_layer)
	x = build_head(x)
	#p3_output = build_head(p3_output)
	#p4_output = build_head(p4_output)
	#p5_output = build_head(p5_output)
	return tf.keras.Model(inputs=input_layer,outputs=x)


#class FeaturePyramid(tf.keras.layers.Layer):
#	"""Builds the Feature Pyramid with the feature maps from the backbone.
#	Attributes:
#	num_classes: Number of classes in the dataset.
#	backbone: The backbone to build the feature pyramid from.
#		Currently supports ResNet50 only.
#	"""
#	def __init__(self, backboneName='ResNet50', **kwargs):
#		super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
#		self.backbone = get_backbone(backboneName)
#		self.conv_c3_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
#		self.conv_c4_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
#		self.conv_c5_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
#		self.conv_c3_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
#		self.conv_c4_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
#		self.conv_c5_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
#		self.conv_c6_3x3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
#		self.conv_c7_3x3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
#		self.upsample_2x = tf.keras.layers.UpSampling2D(2)
#	def call(self, images, training=False):
#		c3_output, c4_output, c5_output = self.backbone(images, training=training)
#		p3_output = self.conv_c3_1x1(c3_output)
#		p4_output = self.conv_c4_1x1(c4_output)
#		p5_output = self.conv_c5_1x1(c5_output)
#		p4_output = p4_output + self.upsample_2x(p5_output)
#		p3_output = p3_output + self.upsample_2x(p4_output)
#		p3_output = self.conv_c3_3x3(p3_output)
#		p4_output = self.conv_c4_3x3(p4_output)
#		p5_output = self.conv_c5_3x3(p5_output)
#		#p6_output = self.conv_c6_3x3(c5_output)
#		#p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
#		return p3_output, p4_output, p5_output
#
#class IWpod_Net(tf.keras.Model):
#	def __init__(self, backboneName='ResNet50', **kwargs):
#		super(IWpod_Net, self).__init__(name="IWpod_Net", **kwargs)
#		self.fpn = FeaturePyramid(backboneName)
#		self.head = build_head()
#
#	def call(self, image, training=False):
#		features = self.fpn(image, training=training)
#		box_outputs = []
#		for feature in features:
#			box_outputs.append(self.head(feature))
#
#		#box_outputs = tf.concat(box_outputs, axis=1)
#		#tf.reshape(box_outputs, [tf.shape(image)[0], -1, -1, 7])
#		return box_outputs



def create_model_iwpodnet(backboneName='Original'):
	#
	#  Creates additonal layers to discriminate the tasks of detection and
	#  localization. Can freeze common layers and train specialized layers
	#  separately
	#
	
	# input_layer = Input(shape=(None,None,3),name='input')
	
	# x = conv_batch(input_layer, 16, 3)
	# x = conv_batch(x, 16, 3)
	# x = MaxPooling2D(pool_size=(2,2))(x)
	# x = conv_batch(x, 32, 3)
	# x = res_block(x, 32)
	# x = MaxPooling2D(pool_size=(2,2))(x)
	# x = conv_batch(x, 64, 3)
	# x = res_block(x,64)
	# x = res_block(x,64)
	# x = MaxPooling2D(pool_size=(2,2))(x)
	# x = conv_batch(x, 64, 3)
	# x = res_block(x,64)
	# x = res_block(x,64)
	# x = MaxPooling2D(pool_size=(2,2))(x)
	# x = conv_batch(x, 128, 3)
	# x = res_block(x,128)
	# x = res_block(x,128)
	# x = res_block(x,128)
	# x = res_block(x,128)
	# x = build_head(x)
	model = buildModel(backboneName=backboneName)
	model.summary()
	return model


if __name__ == '__main__':

	model = create_model_iwpodnet()
	print ('Finished')

