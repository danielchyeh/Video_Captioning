import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
import json

from keras.preprocessing import sequence
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from os import listdir
import data_prepro


data_path = sys.argv[1]
test_output_file = sys.argv[2]
model_file = './models/model.ckpt'

#peer_review_output_file = sys.argv[3]


class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step
                 ,n_caption_lstm_step):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step
        

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb') # (token_unique, 1000)
        
        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False) # c_state, m_state are concatenated along the column axis 
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W') # (4096, 1000)
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')
        # attention variables
        self.attention_z = tf.Variable(tf.random_uniform([self.batch_size,self.lstm2.state_size],-0.1,0.1), name="attention_z")
        self.attention_W = tf.Variable(tf.random_uniform([self.lstm1.state_size,self.lstm2.state_size],-0.1,0.1),name="attention_W")
    
        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W') # (1000, n_words)

        self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_train_model(self):
        #set up the input placeholders
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image]) # (batch, 80, 4096)
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1]) 
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1]) # (batch_size, max_length+1)

        video_flat = tf.reshape(video, [-1, self.dim_image]) 
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])
        #print("lstm1 sate size,",self.lstm1.state_size)
        #print("lstm2 sate size,",self.lstm2.state_size) # 2*hidden size 
        state1 = tf.zeros([self.batch_size, self.lstm1.state_size]) # initial state
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size]) # initial state
        padding = tf.zeros([self.batch_size, self.dim_hidden]) # (batch, 1000)

        probs = []
        loss = 0.0

        ##############################  Encoding Stage ##################################
        context_padding = tf.zeros([self.batch_size, self.lstm2.state_size]) #(batch_size, 2000)
        h_list = []
        for i in range(0, self.n_video_lstm_step): # n_vedio_lstm_step = 80
            with tf.variable_scope("LSTM1", reuse= (i!=0)):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)
                h_list.append(state1)
            with tf.variable_scope("LSTM2", reuse=(i!=0)):
                output2, state2 = self.lstm2(tf.concat( [padding, output1, context_padding] ,1), state2)
        #print(np.shape(h_list))
        h_list = tf.stack(h_list,axis=1) 
        #print(np.shape(h_list)) # (64, 80, 2000)

        ############################# Decoding Stage ######################################
        for i in range(0, self.n_caption_lstm_step): ## Phase 2 => only generate captions
            if i==0:
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
            
            with tf.variable_scope("LSTM1",reuse= True):
                output1, state1 = self.lstm1(padding, state1)
                
                
            with tf.variable_scope("LSTM2", reuse= True):
                ##### attention ####
                context = []
                if i == 0:
                    new_z = self.attention_z

                h_list_flat = tf.reshape(h_list,[-1,self.lstm1.state_size])
                htmp = tf.matmul(h_list_flat,self.attention_W) # for matmul operation (5120,2000)
                hW = tf.reshape(htmp,[self.batch_size, self.n_video_lstm_step,self.lstm2.state_size])
                for x in range(0,self.batch_size):
                    x_alpha = tf.reduce_sum(tf.multiply(hW[x,:,:],new_z[x,:]),axis=1)
                    x_alpha = tf.nn.softmax(x_alpha)
                    x_alpha = tf.expand_dims(x_alpha,1)
                    x_new_z = tf.reduce_sum(tf.multiply(x_alpha,h_list[x,:,:]),axis=0)
                    context.append(x_new_z) 
                context = tf.stack(context)
                #print("context shape", context.shape)
                #with tf.variable_scope("LSTM2", reuse= True):
                #print(output1.shape) # (64,1000)
                output2, state2 = self.lstm2(tf.concat([current_embed, output1, context], 1), state2)
                new_z = state2
        
            labels = tf.expand_dims(caption[:, i+1], 1) # (batch, max_length, 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # (batch_size, 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b) #probability of each word
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels= onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]
                      
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss = loss + current_loss

        return loss, video, video_mask, caption, caption_mask, probs



    def build_generator(self):

        # batch_size = 1 during testing mode
        context_padding = tf.zeros([1, self.lstm2.state_size])
        h_list = []
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image]) # (80, 4096)
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(0, self.n_video_lstm_step):
            with tf.variable_scope("LSTM1", reuse=(i!=0)):
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)
                h_list.append(state1)

            with tf.variable_scope("LSTM2", reuse=(i!=0)):
                output2, state2 = self.lstm2(tf.concat([padding, output1, context_padding], 1), state2)
        h_list = tf.stack(h_list,axis=1) 
        for i in range(0, self.n_caption_lstm_step):
            if i == 0:
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

            with tf.variable_scope("LSTM1", reuse=True):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2", reuse=True):
                context = []
                if i == 0:
                    new_z = self.attention_z
                h_list_flat = tf.reshape(h_list,[-1,self.lstm1.state_size])
                htmp = tf.matmul(h_list_flat,self.attention_W)
                hW = tf.reshape(htmp, [1, self.n_video_lstm_step,self.lstm1.state_size])
                for x in range(0,1): # only one sample 
                    x_alpha = tf.reduce_sum(tf.multiply(hW[x,:,:],new_z[x,:]),axis=1)
                    x_alpha = tf.nn.softmax(x_alpha)
                    x_alpha = tf.expand_dims(x_alpha,1)
                    x_new_z = tf.reduce_sum(tf.multiply(x_alpha,h_list[x,:,:]),axis=0)
                    context.append(x_new_z)
                context = tf.stack(context)
                output2, state2 = self.lstm2(tf.concat([current_embed, output1,context],1), state2)
                new_z = state2
            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds

    def save_model(self, sess, model_file):
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir('model')
        saver = tf.train.Saver()
        saver.save(sess, model_file)
        return


    def restore_model(self, sess, model_file):
        if os.path.isdir(os.path.dirname(model_file)):
            saver = tf.train.Saver()
            saver.restore(sess, model_file)
        return


dim_image = 4096
dim_hidden= 256

n_video_lstm_step = 80
n_caption_lstm_step = 15
n_frame_step = 80

n_epochs = 1000
batch_size = 32
data_number = 1450
learning_rate = 0.001

N_iter = 4000
display_step = 20

#mode: 0 for train, 1 for test.
mode = 1
#reload model: 0 off / 1 on
reload = 1

#input training data
train_label_path = data_path+"training_label.json"
vocab_dict, label_dict = data_prepro.vocab_process(train_label_path) #training labels
x_feats_train, x_seq_train, x_ids_train = data_prepro.load_data(data_path+"training_data/feat/") #training dicts
y_train, y_seq_train = data_prepro.set_captions(vocab_dict, label_dict, x_ids_train)


if mode == 0:
    print("training mode on")  

    model = Video_Caption_Generator(
                dim_image=dim_image,
                n_words=len(vocab_dict),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_lstm_steps=n_frame_step,
                n_video_lstm_step=n_video_lstm_step,
                n_caption_lstm_step=n_caption_lstm_step)
    
    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_train_model()
    tf_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)

    init = tf.global_variables_initializer()
    #sess = tf.InteractiveSession()

    with tf.Session() as sess:
        sess.run(init)
        if reload == 1:
            model.restore_model(sess, model_file)
            print("sucessfully restore the model")

        step = 0 #initialize the training step 
        print("start training")     
        while step < N_iter:
            
            batch_x, batch_y, idx = data_prepro.next_batch(x_feats_train,y_train,batch_size,data_number)
            
            y = np.full((batch_size, n_caption_lstm_step+1), 1)
            y_mask = np.zeros(y.shape, dtype=np.float32)
            for i, caption in enumerate(batch_y):
                y[i,:len(caption)] = caption
                y_mask[i, :len(caption)] = 1
                
            if batch_x.shape[1] == n_video_lstm_step:
                mask_x = np.ones((batch_x.shape[0], batch_x.shape[1]), dtype=np.float32)
            
            sess.run(tf_optimizer, feed_dict={
                    tf_video:batch_x,
                    tf_video_mask:mask_x,
                    tf_caption:batch_y, 
                    tf_caption_mask:y_mask
                    })
    
    
            if step % display_step == 0:
                loss= sess.run(tf_loss, feed_dict={
                        tf_video:batch_x,
                        tf_video_mask:mask_x,
                        tf_caption:batch_y, 
                        tf_caption_mask: y_mask
                        })
                print(str(step) + '/' + str(N_iter) + ' step, loss = ' + str(loss))
                
                model.save_model(sess, model_file)
                print("model saved")
                
            step += 1
            
        model.save_model(sess, model_file)


else:        
    print("testing mode on")

    model = Video_Caption_Generator(
                dim_image=dim_image,
                n_words=len(vocab_dict),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_lstm_steps=n_frame_step,
                n_video_lstm_step=n_video_lstm_step,
                n_caption_lstm_step=n_caption_lstm_step)
    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    sess = tf.InteractiveSession()
    print("start to restore")
    saver = tf.train.Saver()
    saver.restore(sess,model_file)
    print("restore success")

    test_folder_path = data_path+"testing_data/feat/"
    test_path = listdir(test_folder_path)
    test_features = [ (file[:-4],np.load(test_folder_path + file)) for file in test_path]

    test_feature_dict = {}
    for test_tuple in test_features:
        test_feature_dict[test_tuple[0]] = test_tuple[1]
    
    with open(data_path+"testing_id.txt","r") as f:
        test_id = [line.strip() for line in f.readlines()]


    test_sentences = []

    for idx in test_id:
        video_feat = test_feature_dict[idx]
        video_feat = video_feat.reshape(1,80,4096)
        print(video_feat.shape)
        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        print(generated_word_index)
        #print(len(generated_word_index))

        sen_s = []
        single_s = []
        for w in generated_word_index:
            for i_voc in range(0,len(vocab_dict)):
                if(w == i_voc):
                    gen_vocab = vocab_dict[i_voc]
            if gen_vocab == '<eos>' or gen_vocab == '<pad>':
                break
            single_s.append(gen_vocab)
        
        sen_s = ' '.join(single_s)
        if sen_s[0] == ' ':
            sen_s = sen_s[1:]
        test_sentences.append(sen_s)

        print('Test id: {}'.format(idx))
        print('Generated Caption: {}\n'.format(sen_s))


    submit = pd.DataFrame(np.array([test_id,test_sentences]).T)
    submit.to_csv(test_output_file,index = False,  header=False)



    #peerreview part
    peer_flag = 0
    if peer_flag == 1:#test

        peerreview_folder_path = data_path+"peerreview/feat/"
        peerreview_path = listdir(peerreview_folder_path)
        peerreview_features = [ (file[:-4],np.load(peerreview_folder_path + file)) for file in peerreview_path]

        peerreview_feature_dict = {}
        for peerreview_tuple in peerreview_features:
            peerreview_feature_dict[peerreview_tuple[0]] = peerreview_tuple[1]
        
        with open(data_path+"peerreview_id.txt","r") as f:
            test_id = [line.strip() for line in f.readlines()]


        test_sentences = []

        for idx in test_id:
            video_feat = test_feature_dict[idx]
            video_feat = video_feat.reshape(1,80,4096)
            print(video_feat.shape)
            if video_feat.shape[1] == n_frame_step:
                video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

            generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
            print(generated_word_index)
            print(len(generated_word_index))

            sen_s = []
            single_s = []
            for w in generated_word_index:
                for i_voc in range(0,len(vocab_dict)):
                    if(w == i_voc):
                        gen_vocab = vocab_dict[i_voc]
                if gen_vocab == '<eos>':
                    break
                single_s.append(gen_vocab)
            
            sen_s = ' '.join(single_s)
            if sen_s[0] == ' ':
                sen_s = sen_s[1:]
            test_sentences.append(sen_s)

            print(idx)
            print('Generated Caption: {}\n'.format(sen_s))


        submit = pd.DataFrame(np.array([test_id,test_sentences]).T)
        submit.to_csv(test_output_file,index = False,  header=False)
