Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 180, 180, 3)]     0         
                                                                 
 conv2d (Conv2D)             (None, 178, 178, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2  (None, 89, 89, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 87, 87, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 43, 43, 64)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 41, 41, 128)       73856     
                                                                 
 flatten (Flatten)           (None, 215168)            0         
                                                                 
 dropout (Dropout)           (None, 215168)            0         
                                                                 
 dense (Dense)               (None, 1)                 215169    
                                                                 
=================================================================
Total params: 308417 (1.18 MB)
Trainable params: 308417 (1.18 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________