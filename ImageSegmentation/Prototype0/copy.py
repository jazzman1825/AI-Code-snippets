import pickle 
import numpy as np
import matplotlib.pyplot as plt
#from playsound import playsound
from tensorflow import keras

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

# Let's load the data:

print("Ladataan tiedot...")
opetus_ja_vastetiedot=pickle.load( open( "ImageSegmentation/Data/opetustiedot.p", "rb" ) )
print("...valmista!")


# And illustrate its shape:

syotekuva_tensori=opetus_ja_vastetiedot[0]
vastekuva_tensori=opetus_ja_vastetiedot[1]

print("Syötekuvatensorin koko: ",syotekuva_tensori.shape)
print("Vastekuvatensorin koko: ",vastekuva_tensori.shape)

# And some visualization, too:

""" kuva1=plt.figure()
kuva1.suptitle("Alkuperäiset syöte- ja vastekuvahistogrammit")
akse1=kuva1.add_subplot(2,1,1)
histov=akse1.hist(syotekuva_tensori.flatten(),200)
histo2v=np.array(histov,dtype=object)
akse1.set_ylim((0,(np.max(histo2v[0][1:]))))

akse2=kuva1.add_subplot(2,1,2)
histo=akse2.hist(vastekuva_tensori.flatten(),200)
histo2=np.array(histo,dtype=object)
akse2.set_ylim((0,(np.max(histo2[0][1:]))))

plt.show()
quit() """

# Let's do the basic scaling ... 

#skaalaus...
syotekuva_tensori=syotekuva_tensori/255
vastekuva_tensori=vastekuva_tensori/255

# And shape tricks ... the batch dimension... 

#lisataan dimensio...
syotekuva_tensori=np.expand_dims(syotekuva_tensori,axis=3)
vastekuva_tensori=np.expand_dims(vastekuva_tensori,axis=3)

""" print("Syötekuvatensorin koko after expand_dims: ",syotekuva_tensori.shape)
print("Vastekuvatensorin koko after expand_dims: ",vastekuva_tensori.shape)
print(syotekuva_tensori[0].shape)
 """
# For the model we need to define the input and output shape...


kuvakoko=(128,128,1) #syotekuva_tensori[0].shape

IMG_HEIGHT=kuvakoko[0]
IMG_WIDTH=kuvakoko[1]
IMG_CHANNELS=kuvakoko[2]

# The model ... remember a lot of approaches can be found from literature ... and data sources

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

# Contracting Path
c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
p4 = MaxPooling2D((2, 2))(c4)

c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)

# Expansive Path    
u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = Concatenate()([u6, c4])
c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)

u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = Concatenate()([u7, c3])
c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)

u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = Concatenate()([u8, c2])
c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)

u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = Concatenate()([u9, c1])
c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
keras.utils.plot_model(model, show_shapes=True)

#näppituntumaa tarvittaessa...

#print(model(np.zeros((1,128,128,1))).shape)
#print(model(np.expand_dims(np.expand_dims(syotekuva_tensori[0],axis=0),axis=3)))

filepath = "terasmalli.h5"

# Configurations for the training...

earlystopper = EarlyStopping(patience=5, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

callbacks_list = [earlystopper, checkpoint]

# Let's divide the training and validation data sets...

opetusjoukko=[]
validointijoukko=[]
for i in range(200): #syotekuva_tensori):
    if np.random.rand()<0.2:
        opetusjoukko.append(i)
        #pippurointi, hox käytä vain jos tiedät mitä tarkoittaa...
        if np.random.rand()<0.0:
            vastekuva_tensori[i,50:60,50:60]=1
    else:
        validointijoukko.append(i)
        #pippurointi, hox käytä vain jos tiedät mitä tarkoittaa...
        #if np.random.rand()>0.5:
            #vastekuva_tensori[i,50:60,50:60]=1

opetusjoukko=np.array(opetusjoukko)
validointijoukko=np.array(validointijoukko)

print("Opetusjoukko N=",len(opetusjoukko)," ja validointijoukko N=",len(validointijoukko))
print("Opetusjoukon vastekuvien keskiarvo:",np.mean(vastekuva_tensori[opetusjoukko]))
print("Validointijoukon vastekuvien keskiarvo",np.mean(vastekuva_tensori[validointijoukko]))

#painotus voi olla tärkeää...
#class_weights={0:1,1:10}
sample_weights=np.zeros((128,128,2))
sample_weights[:,:,0]=0.1
sample_weights[:,:,1]=0.9

# Remember to take care about computation load ... (let's discuss...)

history = model.fit(
    syotekuva_tensori[opetusjoukko],vastekuva_tensori[opetusjoukko],
    validation_data=(syotekuva_tensori[validointijoukko],vastekuva_tensori[validointijoukko]),
    batch_size=16, epochs=20,
    callbacks=callbacks_list
#    sample_weight=sample_weights
    )

print(history.history)

suorituskykyfig=plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.draw()
plt.pause(3)

fig=plt.figure()
sub1=fig.add_subplot(2,3,1)
sub2=fig.add_subplot(2,3,2)
sub3=fig.add_subplot(2,3,3)
sub_ero=fig.add_subplot(2,3,5)

#HOX tässä voit ottaa "toisen laatiman..." mallin ja testata sitä ... ja toisinpäin..."

#tallennetaan malli...
model.save('terasmies')

#ladataan painokertoimet...
model.load_weights('terasmalli.h5')

#Testataan mallia, valitaan yksi kuva ja tulostetaan sen päälle vaste...
for testikuva_numero in range(5):
    print("Testikuvan jrj numero...",testikuva_numero)
    syotekuva=np.expand_dims(np.expand_dims(syotekuva_tensori[testikuva_numero],axis=0),axis=3)
    testitulos=model.predict(syotekuva)

    #kasitellaan testitulos kynnystysta kayttaen:
    #testitulos_np0=np.array(testitulos)
    testitulos_np_k=(testitulos>=0.5).astype(np.uint8)

    #muunnetaan testitulos samaan shapeen kuin syotekuva...
    testitulos_np=testitulos_np_k[0,:,:,0]
    #plt.hist(testitulos_np.flatten())
    #plt.draw()
    #plt.pause(0.1)
    #print(testitulos)
    #testitulos_np[50:60,50:60]=255

    #haetaan testituloksesta maskiarvot suoraan matriisiin...
    herkkyys=30
    print("Testitulos-matriisin summa:",np.sum(testitulos_np))

    if 1: #np.sum(testitulos_np)>herkkyys:
    #if np.sum(testitulos_np)>herkkyys:

        indeksit=np.where(testitulos_np>herkkyys)
        print("Testituloksen muoto: ",testitulos_np.shape)

        #merkitään tämä jotenkin kuvaan...
        yhdistelmakuva_pohja=np.copy(syotekuva_tensori[testikuva_numero,:,:,0])
        print("Pohjakuvan muoto: ",yhdistelmakuva_pohja.shape)
        #print(indeksit)
        lapinakyvyys=0.3
        yhdistelmakuva_pohja[indeksit]=lapinakyvyys*yhdistelmakuva_pohja[indeksit]+(1-lapinakyvyys)*255


        #sub1=fig.add_subplot(2,3,1)
        sub1.cla()
        sub1.imshow(np.squeeze(syotekuva_tensori[testikuva_numero,:,:,0]),cmap='gray')
        sub1.set_xticks([])
        sub1.tick_params(labelsize=5)

        sub1.title.set_text('Test Image')

        #sub2=fig.add_subplot(2,3,2)
        sub2.cla()
        sub2.imshow(np.squeeze(vastekuva_tensori[testikuva_numero,:,:,0]),cmap='gray')
        #sub2.matshow(np.squeeze(vastekuva_tensori[testikuva_numero,:,:,0]))
        sub2.set_xticks([])
        sub2.tick_params(labelsize=5)

        sub2.title.set_text('True Mask')

        #sub3=fig.add_subplot(2,3,3)
        sub3.cla()
        sub3.imshow(testitulos_np,cmap='gray')
        sub3.set_xticks([])
        sub3.tick_params(labelsize=5)

        sub3.title.set_text('Predicted Mask')

        #Tutkitaan kuinka hyvin ennuste ja todellinen vastaavat toisiaan...
        gt=np.squeeze(vastekuva_tensori[testikuva_numero,:,:,0])
        seg=np.squeeze(testitulos_np)

        k=1
        # viite: https://stackoverflow.com/questions/31273652/how-to-calculate-dice-coefficient-for-measuring-accuracy-of-image-segmentation-i
        ero_dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg) + np.sum(gt))
        #Huom! Tulee not-a-number jos sekä seg että gt on nollia...

        #playsound('klik.wav')
        #lasketaan
        sub_ero.cla()
        sub_ero.text(0.5,0.5,str(ero_dice))


        plt.draw()
        plt.pause(0.1)
