# coding: utf-8

# In[ ]:

import tensorflow as tf
import sys
import os

import data_utils
from model_seq2seq import CaptionGenerator
# Environment Parameters

#mode = 'train'
#mode = 'load_train'
#mode = 'predict'
mode = 'load_predict'

model_name = 'models/hw2_special'
prediction_name = "Output1113v1.txt"


#train_path = "./MLDS_hw2_data/training_data/feat/"
#test_path = "./MLDS_hw2_data/testing_data/feat/"
#label_train_path = "./MLDS_hw2_data/training_label.json"
#
#peer_review_path = "./MLDS_hw2_data/testing_data/feat/"


#test_path = sys.argv[1] + 'testing_data/feat/'#directory which contains testing video feature files
#result_file = "outputv1.txt"#result txt file
#    
#
#peer_review_path = sys.argv[1] + 'peer_review/feat/'
#peer_review_result_file = sys.argv[3]

train_path = os.path.join( sys.argv[1],'training_data/feat/' )
label_train_path = os.path.join( sys.argv[1],'training_label.json')
test_path = os.path.join( sys.argv[1],'testing_data/feat/' )#directory which contains testing video feature files
result_file = sys.argv[2]#result txt file
    
peer_review_path = os.path.join( sys.argv[1],'peer_review/feat/' )
peer_review_result_file = sys.argv[3]


iter_epochs = None
# iter_epochs = range(99, 1000, 100)


training_max_time_steps = 40
word_encoding_threshold = 1
random_every_epoch = True
shuffle_training_data = True
save_per_epoch = 100

num_units = 256
num_layers = 2
x_embedding_size = 4096

use_dropout = True
output_keep_prob = 0.5 if 'train' in mode else 1.0
use_residual = True 
projection_using_bias = False
attention_type = 'Luong'
beam_width = 3
max_to_keep = 20

start_id = 1
end_id = 2

epochs = 200
batch_size = 50

use_attention = False
use_beamsearch = False


# In[ ]:
y_vocab_size = 6059


vocab_dict, label_dict = data_utils.vocab_process(label_train_path)
#y_train, y_seq_len = data_utils.set_captions(vocab_dict, label_dict)

if 'train' in mode:
    x_feats_train, x_seq_train, x_ids_train = data_utils.load_data(train_path)
    
elif 'predict' in mode:
    x_feats_test, x_seq_test, x_ids_test = data_utils.load_data(test_path)
    x_feats_peer, x_seq_peer, x_ids_peer = data_utils.load_data(peer_review_path)


## In[ ]:
#
##MODEL
## <S2VT> Sequence to Sequence - Video to Text without CNN
#
model = CaptionGenerator(num_units, num_layers, y_vocab_size, x_embedding_size,training_max_time_steps,
                         start_id, end_id, output_keep_prob, use_dropout, use_attention, attention_type,
                         projection_using_bias, use_residual, use_beamsearch, beam_width=3)


learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(model.loss)    

#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)


# In[ ]:

import numpy as np

saver = tf.train.Saver(max_to_keep=max_to_keep)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
print('Session started')


if 'train' in mode:
    if 'load' in mode:
        saver.restore(sess, '{}.ckpt'.format(model_name))
    
    best_loss_val = 1e10


    for epoch in range(epochs):
        if random_every_epoch:
            y_train, y_seq_train = data_utils.set_captions(vocab_dict, label_dict, x_ids_train)
        count = 0
        loss_val_sum = 0.0
#        for x_, y_, x_seq_len_, y_seq_len_ in msvd.next_batch(batch_size):
        for x_, y_, x_seq_len_, y_seq_len_ in data_utils.next_batch(x_feats_train,y_train,x_seq_train,y_seq_train,batch_size):
            if shuffle_training_data:
                p = np.random.permutation(x_.shape[0])
                x_, y_, x_seq_len_, y_seq_len_ = x_[p], y_[p], x_seq_len_[p], y_seq_len_[p]
            _, loss_val, preds = sess.run(
                [train_op, model.loss, model.training_id], 
                feed_dict={model.x: x_, 
                           model.y: y_, 
                           model.x_seq_len: x_seq_len_,
                           model.y_seq_len: y_seq_len_, 
                           model.y_max_seq_len: np.full(y_seq_len_.size, 
                                                  training_max_time_steps, 
                                                  dtype=np.int32)})
            count += 1
            loss_val_sum += loss_val
#             print('Epoch {}, train_loss_val: {}.'.format(epoch, loss_val))
#             predictions.print(preds, False, True, '=> {}')
            if loss_val < best_loss_val:
                best_loss_val = loss_val
#                 print('Model saved.')
#                 saver.save(sess, '{}.ckpt'.format(model_name))
            if save_per_epoch and (epoch+1) % save_per_epoch == 0:
                print('Model saved.')
                saver.save(sess, '{}_epoch_{}.ckpt'.format(model_name, epoch))
        loss_val_avg = loss_val_sum / count
        
        #Show the first 5 preds in training mode
        train_sen = []
        for psa in preds:
            sa = []
            for wa in psa:
#                   idx = np.argmax(w)
                for i_voca in range(0, len(vocab_dict)):
                    if (wa == i_voca):
                        vocaba = vocab_dict[i_voca]                    
                if vocaba == '<eos>':
                    break
                sa.append(vocaba)
            train_sen.append(' '.join(sa))
        print('The first 5 training preds: \n')
        for i_preda in range(0,5):
            print('{}\n'.format(train_sen[i_preda]))
               
        
#        predictions.print(preds[:5], False, True, '=> {}')
        print('Epoch {}, average_train_loss_val: {}.'.format(epoch, loss_val_avg))

    print('Finished model training. The best train_loss_val: {}.'.format(best_loss_val))
    
elif 'predict' in mode:
    if 'load' in mode:
        saver.restore(sess, '{}.ckpt'.format(model_name))

    #TESTING
    for x_, x_seq_len_, x_id in data_utils.testing_data(x_feats_test,x_seq_test,x_ids_test):
        preds = sess.run(
                model.predicting_id, 
                feed_dict={model.x: x_, 
                           model.x_seq_len: x_seq_len_})
        
    sentencesss = []
    for ps in preds:
        s = []
        for w in ps:
#                idx = np.argmax(w)
            for i_voc in range(0, len(vocab_dict)):
                if (w == i_voc):
                    vocab = vocab_dict[i_voc]                    
#                vocab = self.vocab_processor._reverse_mapping[idx]
            if vocab == '<eos>':
                break
            s.append(vocab)
        sentencesss.append(' '.join(s))
    print('The first 5 predictions: \n')
#        for pred in sentencesss:
    for i_pred in range(0,5):
        print('{}\n'.format(sentencesss[i_pred]))

            
    with open(result_file, "w") as text_file:
        for i_resu in range(0,len(sentencesss)):
            text_file.write(str(x_id[i_resu])+","+str(sentencesss[i_resu])) 
            text_file.write('\n')
 

           
#    #PEER REVIEW
    for x_, x_seq_len_, x_id in data_utils.testing_data(x_feats_peer,x_seq_peer,x_ids_peer):
        preds = sess.run(
                model.predicting_id, 
                feed_dict={model.x: x_, 
                           model.x_seq_len: x_seq_len_})
        
    sentencesss = []
    for ps in preds:
        s = []
        for w in ps:
#                idx = np.argmax(w)
            for i_voc in range(0, len(vocab_dict)):
                if (w == i_voc):
                    vocab = vocab_dict[i_voc]                    
#                vocab = self.vocab_processor._reverse_mapping[idx]
            if vocab == '<eos>':
                break
            s.append(vocab)
        sentencesss.append(' '.join(s))
            
    with open(peer_review_result_file, "w") as text_file:
        for i_resu in range(0,len(sentencesss)):
            text_file.write(str(x_id[i_resu])+","+str(sentencesss[i_resu])) 
            text_file.write('\n')        
        

    print('Finished predicting ya.')



