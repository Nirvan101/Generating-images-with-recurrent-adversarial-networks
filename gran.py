from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout, concatenate, Add
from keras.models import Model,Sequential
from keras import backend as K
from PIL import Image
import os
from keras.callbacks import ModelCheckpoint
import random,numpy as np
import cv2

'''
input: hct,ct,z
output: hct,ct
'''
def generator():
    inp_hct = Input((128,))
    inp_ct = Input((64,64,3))
    inp_z = Input((128,))
    inp = concatenate(  [inp_hct, inp_z]   )
    x = Dense(1024, input_dim=256, activation='relu')( inp )
    x = Dense( 128 * 8 * 8 , activation='relu')(x)
    x = Reshape( (8,8,128) )(x) 
    x = Conv2D(128, (2,2), padding='same' ,strides=1,activation='relu')(x)  
    x = UpSampling2D((2, 2))(x)           #( 16, 16, 128)
    x = Conv2D(128, (2,2), padding='same', strides=1,activation='relu')(x)
    x = UpSampling2D((2, 2))(x)         #( 32, 32, 128)
    x = Conv2D(64, (2,2), padding='same', strides=1,activation='relu')(x)
    x = UpSampling2D((2, 2))(x)         #( 64, 64, 128)
    ct = Conv2D(3, (2,2), padding='same', strides=1,activation='relu')(x)   #(64, 64, 3)
    
    
    x = Conv2D(64, (4, 4), strides = 1, activation='relu', padding='same')(inp_ct)   #(64,64)
    x = MaxPooling2D((2, 2), padding='same')(x)     #(32,32)
    x = Conv2D(64, (2, 2), strides = 1, activation='relu', padding='same')(x)   #(32,32)
    x = MaxPooling2D((2, 2), padding='same')(x)     #(16,16)
    x = Conv2D(128, (2, 2), strides = 1, activation='relu', padding='same')(x)   #(16,16)
    x = MaxPooling2D((2, 2), padding='same')(x)     #(8,8,128)
    x = Flatten()(x)  
    x = Dense(128 * 8 * 8 ,activation='relu')(x)
    x = Dense(1024 ,activation='relu')(x)
    hct = Dense(128 ,activation='tanh')(x)
    
    return Model( [inp_hct,inp_ct,inp_z] , [hct,ct] )



'''
input: ct
output: prob value that ct is real
'''
def discriminator(): 
    inp = Input((64,64,3))
    x = Conv2D(64, (4, 4), strides = 1, activation='relu', padding='same')(inp)   #(64,64)
    x = MaxPooling2D((2, 2), padding='same')(x)     #(32,32)
    x = Conv2D(64, (2, 2), strides = 1, activation='relu', padding='same')(x)   #(32,32)
    x = MaxPooling2D((2, 2), padding='same')(x)     #(16,16)
    x = Conv2D(128, (2, 2), strides = 1, activation='relu', padding='same')(x)   #(16,16)
    x = MaxPooling2D((2, 2), padding='same')(x)     #(8,8,128)
    x = Flatten()(x)  
    x = Dense(128 * 8 * 8 ,activation='relu')(x)
    x = Dense(1024 ,activation='relu')(x)
    x = Dense(128 ,activation='relu')(x)
    x = Dense(64 ,activation='relu')(x)
    x = Dense(1 ,activation='sigmoid')(x)
    
    return Model( inp , x )


#to train generator and discriminator together
def gen_on_dis(gen,dis):
    inp_c = Input((64,64,3))
    
    add = Add()([gen.output[1] , inp_c])
    
    out = dis(add) 
    return Model( [gen.input[0] , gen.input[1], gen.input[2] , inp_c] ,out )


#applies recurrence over the generator
def gen_predict(gen,z,hct,ct):    
    C = []
        
    num_steps = 20    
    for i in range(num_steps):
        t = [ np.array(hct) ,  np.array(ct) ,  np.array(z) ]
        
        #t = K.expand_dims(t,-1)
        
        [hct,ct] = gen.predict( t )
        C.append(ct)
        
    C = np.array(C)           # (20,batch_size,64,64,3)
    
    C = C.sum(0)              # (batch_size,64,64,3)
    
    return [C,hct,ct]    


gen = generator()
gen.compile(loss='binary_crossentropy',optimizer='adam')
gen.summary()

dis = discriminator()
dis.compile(loss='binary_crossentropy',optimizer='adam')


g_on_d = gen_on_dis(gen,dis)
g_on_d.compile(loss='binary_crossentropy',optimizer='adam')
g_on_d.summary()



path = './images/'
images = os.listdir(path)

datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    
batches = datagen.flow_from_directory(path ,target_size = (64,64),batch_size = 32, class_mode=None)

num_epochs = 100
num_steps = 100
batch_size = 32

epochnum = 1

#train batch-wise
#batch_size = 32 images
for _ in range(num_epochs):
    
    print()
    print('--- Epoch # '+str(epochnum))
    epochnum += 1
    
    X = np.array( batches.next() )            # (batch_size,64,64,3)
    y = np.array( [1]*len(X) )                # (batch_size,)
    
    dis.trainable = True
    dis.train_on_batch(X,y)
    
    
    z = np.random.rand( len(X),128 )           # (batch_size,128)
    hct = np.random.rand( len(X),128 )         # (batch_size,128)
    ct = X                                     # (batch_size,64,64,3)
    
    print('z shape = '+str(z.shape))
    print('ct shape = '+str(ct.shape))
    print('hct shape = '+str(hct.shape))
    print()
    
    [C,hct,ct] = gen_predict(gen,z,hct,ct)
     
    y = np.array( [0]*C.shape[0] )             # (batch_size,)
    dis.train_on_batch( np.array(C) , y )
    
    dis.trainable = False
    g_on_d.train_on_batch([ np.array(hct), np.array(ct) , np.array(z) , np.array(C)] , np.array([1]*len(X)) )

        
#predict
z = np.random.rand( batch_size,128 )
hct = np.random.rand(batch_size,128 )
ct = batches.next()    
[C,hct,ct] = gen_predict(gen,z,hct,ct)
    
#C is the list of output images
#C = (batch_size,64,64,3)



#train image-wise
#batch_size = 1 
'''
for _ in range(num_epochs):
    for i in images:
        
        X = cv2.imread(path + str(i) )
        X = cv2.resize(X, (64,64))
        y = np.array( [1] )
        
        dis.trainable = True
        dis.train_on_batch(np.array( [X] ),y)
        
        z = np.random.rand( 128 )
        hct = np.random.rand( 128 )
        
        ct = X
        [C,hct,ct] = gen_predict(gen,z,hct,ct)
        
        y = np.array( [0] )
        dis.train_on_batch(np.array([C]),y)
        
        dis.trainable = False
        g_on_d.train_on_batch([[hct,ct,z,C]] , np.array([1]) )
'''

