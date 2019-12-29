import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow import keras
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
x=np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0],dtype='float')
y=np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0],dtype='float')
def build_model():
    model=tf.keras.Sequential([
        # tf.keras.layers.Dense(1,input_shape=[1]),
        tf.keras.layers.Dense(32,activation='relu',input_shape=[1]),
        tf.keras.layers.Dense(1),
    ])
    return model
model=build_model()
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit(x,y,epochs=500)
print(model.predict([6.5]))


def export_model(saver,model,input_node_names,output_node_name):
    tf.train.write_graph(K.get_session().graph_def,'out','Reg_Test'+'_graph.pbtxt')
    saver.save(K.get_session(),'out/'+'Reg_Test'+'.chkp')
    freeze_graph.freeze_graph('out/'+'Reg_Test'+'_graph.pbtxt',None,False,'out/'+'Reg_Test'+'.chkp',output_node_name,"save/restore_all","save/Const:0",'out/frozen_'+'Reg_Test'+'.pb',True,""
                              )
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + 'Reg_Test' + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + 'Reg_Test' + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_2/Softmax")