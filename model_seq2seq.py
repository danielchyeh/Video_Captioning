#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:29:21 2017

@author: cbel
"""
import tensorflow as tf
from tensorflow.python.layers.core import Dense


"""
Basic attention-based caption generator using tensorflow dynamic_rnn_decoder API
"""

class CaptionGenerator(object):
    def __init__(self, num_units, num_layers, y_vocab_size, x_embedding_size, training_max_time_steps,
                 start_id, end_id, output_keep_prob, use_dropout, use_attention, attention_type,
                 projection_using_bias, use_residual, use_beamsearch, beam_width=3):
    
        self.x = tf.placeholder(tf.float32, [None, 80, x_embedding_size], name='x')
        self.y = tf.placeholder(tf.int32, [None, training_max_time_steps + 1], name='y')
        self.x_seq_len = tf.placeholder(tf.int32, [None], name='x_seq_len')
        self.y_seq_len = tf.placeholder(tf.int32, [None], name='y_seq_len')
        self.y_max_seq_len = tf.placeholder(tf.int32, [None], name='y_max_seq_len')
        self.model_batch_size = tf.shape(self.x)[0]
        
        def rnn_cell():
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            if use_dropout:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
            if use_residual:
                cell = tf.nn.rnn_cell.ResidualWrapper(cell)
            return cell
        
#        with tf.name_scope("encoder"):
        # Encoder

        input_projection_layer = tf.layers.dense(
                inputs=self.x, 
                units=num_units, 
                use_bias=projection_using_bias
                )

            # encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                [rnn_cell() for _ in range(num_layers)])

            # initial_state = tf.zeros([tf.size(x_seq_len), encoder_cell.state_size])

        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                cell=encoder_cell, 
                #     inputs=x, 
                inputs=input_projection_layer,
                #     initial_state=initial_state, 
                sequence_length=self.x_seq_len, 
                dtype=tf.float32)
            
#        with tf.name_scope("decoder"):   
             
        ## Decoder for training
            
        self.embedding = tf.Variable(
            #     tf.truncated_normal([y_vocab_size, num_units], mean=0.0, stddev=0.1), 
            tf.random_uniform([y_vocab_size, num_units], -0.1, 0.1), 
            dtype=tf.float32)
            
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.y[:, :-1])
            
        output_projection_layer = Dense(
            y_vocab_size, 
            use_bias=projection_using_bias
            )
            
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=self.encoder_inputs_embedded,
            sequence_length=self.y_max_seq_len, 
            # although we don't want to feed <eos> into the decoder, still setting seq_len to be max here
            # later it will be filtered out by masks in the loss calculating state
            time_major=False)
            
        # decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
            
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [rnn_cell() for _ in range(num_layers)])
            
        if use_attention:
            ## Attention model
            if attention_type == 'Bahdanau':
                training_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=num_units, 
                    memory=self.encoder_outputs, 
                    memory_sequence_length=self.x_seq_len)
            else:
                training_attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    num_units=num_units, 
                    memory=self.encoder_outputs, 
                    memory_sequence_length=self.x_seq_len)
            
            training_attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell, 
                attention_mechanism=training_attention_mechanism, 
                attention_layer_size=num_units)
            
            training_attention_state = training_attention_cell.zero_state(
                self.model_batch_size, tf.float32).clone(cell_state=self.encoder_state)
            
        if not use_attention:
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, 
                initial_state=self.encoder_state,
                helper=training_helper, 
                output_layer=output_projection_layer)
            
            
            
        training_maximum_iterations = tf.round(tf.reduce_max(training_max_time_steps))
            
        training_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=training_decoder, 
            maximum_iterations=training_maximum_iterations)
            
        # epsilon = tf.constant(value=1e-50, shape=[1])
        # logits = tf.add(outputs.rnn_output, epsilon)
        self.training_logits = training_outputs.rnn_output
        self.training_id = training_outputs.sample_id
            
            
        
        # Decoder for predicting
            
#        tag_bos = 1 #<bos>
#        tag_eos = 2 #<eos>
        start_ids = tf.fill([self.model_batch_size], start_id)
            
        if not use_beamsearch:
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=self.embedding, 
                start_tokens=start_ids, 
                end_token=end_id)
            
        if use_beamsearch:
            # Beam Search tile
            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(self.encoder_outputs, multiplier=beam_width)
            tiled_x_seq_len = tf.contrib.seq2seq.tile_batch(self.x_seq_len, multiplier=beam_width)
            tiled_encoder_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, multiplier=beam_width)
            
        if use_attention:
            if not use_beamsearch:
                ## Attention model
                if attention_type == 'Bahdanau':
                    predicting_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                        num_units=num_units, 
                        memory=self.encoder_outputs, 
                        memory_sequence_length=self.x_seq_len)
                else:
                    predicting_attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                        num_units=num_units, 
                        memory=self.encoder_outputs, 
                        memory_sequence_length=self.x_seq_len)
            else:
                ## Attention model
                if attention_type == 'Bahdanau':
                    predicting_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                        num_units=num_units, 
                        memory=tiled_encoder_outputs, 
                        memory_sequence_length=tiled_x_seq_len)
                else:
                    predicting_attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                        num_units=num_units, 
                        memory=tiled_encoder_outputs, 
                        memory_sequence_length=tiled_x_seq_len)
            
            
        if not use_beamsearch:
            if not use_attention:
                predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell, 
                    initial_state=self.encoder_state, 
                    helper=predicting_helper, 
                    output_layer=output_projection_layer)
            
            
            
            
        predicting_maximum_iterations = tf.round(tf.reduce_max(training_max_time_steps) * 2)
            
        predicting_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=predicting_decoder, 
            impute_finished=False, 
            maximum_iterations=predicting_maximum_iterations)
            
        if not use_beamsearch:
            self.predicting_id = predicting_outputs.sample_id
        else:
            self.predicting_id = predicting_outputs.predicted_ids[:, :, 0]
             
             
             
        # Loss and Optimizer
#        with tf.name_scope("Loss"):
        targets = self.y[:, 1:]
            
        masks = tf.sequence_mask(self.y_seq_len, training_max_time_steps, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.training_logits, 
            targets=targets, 
            weights=masks, 
            average_across_timesteps=False, 
            average_across_batch=True)
        self.loss = tf.reduce_sum(loss)


