# questions: train_loss, projection layer, target weight


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import tensorflow as tf
import numpy as np


def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.

    Args:
      X: [m,n,k]
      W: [k,l]

    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


def process_decoder_input(target_data, batch_size):
    """
    Preprocess target data for encoding
    :return: Preprocessed target data
    """
    # get '<GO>' id
    go_id = 0
    
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    filled_go_id = tf.fill([batch_size, 1], go_id)
    after_concat = tf.concat( [filled_go_id, after_slice], 1)
    
    return after_concat


def MakeFancyRNNCell(H, keep_prob, num_layers=1):
    """Make a fancy RNN cell.

    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.

    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers

    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cells = []
    for _ in range(num_layers):
        cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0)
        #cell = tf.nn.rnn_cell.DropoutWrapper(
        #  cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        cells.append(cell)

    return tf.nn.rnn_cell.MultiRNNCell(cells)


# Decorator-foo to avoid indentation hell.
# Decorating a function as:
# @with_self_graph
# def foo(self, ...):
#     # do tensorflow stuff
#
# Makes it behave as if it were written:
# def foo(self, ...):
#     with self.graph.as_default():
#         # do tensorflow stuff
#
# We hope this will save you some indentation, and make things a bit less
# error-prone.
def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper


class RNNLM(object):
    def __init__(self, graph=None, *args, **kwargs):
        """Init function.

        This function just stores hyperparameters. You'll do all the real graph
        construction in the Build*Graph() functions below.

        Args:
          V: vocabulary size
          H: hidden state dimension = embedding size
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, V, H, softmax_ns=200, num_layers=1, batch_size=100):
        # Model structure; these need to be fixed for a given model.
        self.V = V
        self.H = H
        self.num_layers = num_layers
        self.batch_size_ = batch_size


        # Training hyperparameters; these can be changed with feed_dict,
        # and you may want to do so during training.
        with tf.name_scope("Training_Parameters"):
            # Number of samples for sampled softmax.
            self.softmax_ns = softmax_ns

            self.learning_rate_ = tf.placeholder(tf.float32, [], name="learning_rate")

            # For gradient clipping, if you use it.
            # Due to a bug in TensorFlow, this needs to be an ordinary python
            # constant instead of a tf.constant.
            self.max_grad_norm_ = 1.0

            self.use_dropout_ = tf.placeholder_with_default(
                False, [], name="use_dropout")

            # If use_dropout is fed as 'True', this will have value 0.5.
            self.dropout_keep_prob_ = tf.cond(
                self.use_dropout_,
                lambda: tf.constant(0.5),
                lambda: tf.constant(1.0),
                name="dropout_keep_prob")

            # Dummy for use later.
            self.no_op_ = tf.no_op()


    @with_self_graph
    def BuildCoreGraph(self):
        """Construct the core RNNLM graph, needed for any use of the model.

        This should include:
        - Placeholders for input tensors (encoder_inputs_, initial_h_, decoder_outputs_)
        - Variables for model parameters
        - Tensors representing various intermediate states
        - A Tensor for the final state (final_h_)
        - A Tensor for the output logits (logits_), i.e. the un-normalized argument
          of the softmax(...) function in the output layer.
        - A scalar loss function (loss_)

        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).

        """
        # Input ids, with dynamic shape depending on input. Sourse input words
        # Should be shape [batch_size, max_encoder_time]
        self.encoder_inputs_ = tf.placeholder(tf.int32, [None, None], name="encoder_inputs")  # Batch_size * sentence length

#         # target input words??
#         # Should be shape [batch_size, max_decoder_time] and contain integer word indices.
#         self.decoder_inputs_ = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")  # Batch_size * sentence length
        
        # target output words, these are decoder_inputs shifted to the left by one time step with an 
        # end-of-sentence tag appended on the right
        # Should be shape [batch_size, max_decoder_time] and contain integer word indices.
        self.decoder_outputs_ = tf.placeholder(tf.int32, [None, None], name="decoder_outputs")  # Batch_size * sentence length

        # Get dynamic shape info from inputs
        #with tf.name_scope("batch_size"):
        #self.batch_size_ = tf.shape(self.encoder_inputs_)[0]  #  batch_size is set to # of rows in the input 
        
        #with tf.name_scope("max_encoder_time"):
        self.max_encoder_time_ = tf.shape(self.encoder_inputs_)[1]  # max_encoder_time is the # of cols of encode input. 
            
        #with tf.name_scope("max_decoder_time"):
        self.max_decoder_time_ = tf.shape(self.decoder_inputs_)[1]  # max_decoder_time is the # of cols of decode input. 

        # Get sequence length from encoder_inputs_.
        # TL;DR: pass this to dynamic_rnn.
        # This will be a vector with elements ns[i] = len(input_w_[i])
        # You can override this in feed_dict if you want to have different-length
        # sequences in the same batch, although you shouldn't need to for this
        # assignment.
        self.ns_ = tf.tile([self.max_encoder_time_], [self.batch_size_, ], name="ns")

        self.decoder_lengths_ = tf.reduce_sum(tf.to_int32(tf.not_equal(self.decoder_outputs_, 1)), 1)  # 1D vector (reduced to "1" dimension)
#         self.decoder_lengths_= tf.placeholder(tf.int32, [None], name="decoder_lengths")
#         self.max_target_length = tf.reduce_max(self.decoder_lengths_) # the length of the longest sentence from the source input data, used for GreedyEmbeddingHelper


        self.embedding_ = tf.Variable(tf.random_uniform([self.V, self.H],-1, 1), name="embedding")       

        #   print(self.encoder_emb_inp_.get_shape())


        # Construct RNN/LSTM cell and recurrent layer.
        with tf.name_scope("Encoder_Layer"):
            # Maps a sequence of symbols to a sequence of embeddings
            #input ids:[batch_size, doc_length]; return: tensor of [batch_size, doc_length, embed_dim] with embedded sequences
            self.encoder_emb_inp_ = tf.contrib.layers.embed_sequence(self.encoder_inputs_, 
                                             vocab_size=self.V, 
                                             embed_dim=self.H)
            
            self.encoder_cell_ = MakeFancyRNNCell(H=self.H, keep_prob=0, num_layers=self.num_layers)           
            self.encoder_initial_h_ = self.encoder_cell_.zero_state(self.batch_size_, dtype=tf.float32)
            
            #   encoder_outputs: [batch_size, max_encoder_time, H]
            #   encoder_final: [batch_size, H]
            self.encoder_outputs_, self.encoder_final_h_ = tf.nn.dynamic_rnn(
                                           cell=self.encoder_cell_, inputs=self.encoder_emb_inp_,
                                           initial_state=self.encoder_initial_h_)
#                                            sequence_length=self.source_sequence_length_)
        
        # decode training and inference share parameters and variables, should in the same variable scope
        # the number of RNN layers in the decoder model has to be equal to the number of RNN layers in the encoder model
        with tf.name_scope("Decoder_Layer"):      
            self.tmp_dec_inp = process_decoder_input(self.decoder_inputs_,self.batch_size_)  #2nd argument needs to be the batch size, matching the decoder_initial_h_ size.

            # decoder training and inference share the same embedding parameters
            self.decoder_emb_inp_ = tf.nn.embedding_lookup(self.embedding_, self.tmp_dec_inp)

            self.decoder_cell_ = MakeFancyRNNCell(H=self.H, keep_prob=self.dropout_keep_prob_, num_layers=self.num_layers)
#             self.decoder_cell_ = tf.nn.rnn_cell.BasicLSTMCell(self.H)
            self.decoder_initial_h_ = self.decoder_cell_.zero_state(self.batch_size_, dtype=tf.float32)  # This needs to match dimention of the self.tmp_dec_inp that is also used in the helper. 


            # Helper, pass the embedded input to the basic decoder
            self.helper_ = tf.contrib.seq2seq.TrainingHelper(
                self.decoder_emb_inp_, 
                self.decoder_lengths_
                ) # what's decoder sequence length, 
            
            # Decoder, accessing to the source information through initializing it with the last hidden state of the encoder

            # basic decoder model, connecting the decoder RNN layers and the input prepared by TrainingHelper
#             self.decoder_ = tf.contrib.seq2seq.BasicDecoder(
#                 self.decoder_cell_, self.helper_, self.decoder_initial_h_ )
            
            self.decoder_ = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell_, helper=self.helper_, 
                            initial_state=encoder_state, output_layer=self.output_layer_ )
            
            # Dynamic decoding, returns (final_outputs, final_state, final_sequence_lengths)
            self.outputs_, _ ,_ = tf.contrib.seq2seq.dynamic_decode(self.decoder_,impute_finished=True) 
                                                                    #maximum_iterations=max_summary_length
            self.logits_ = self.outputs_.rnn_output
        
            # output layer, turn the top hidden states to logit vectors of dimension V
            self.output_layer_ = tf.layers.Dense(self.V, use_bias=False) 
            
            
        # create an inference process in decoding layer
      
            self.decoder_cell_inf_ = MakeFancyRNNCell(H=self.H, keep_prob=self.dropout_keep_prob_, num_layers=self.num_layers)
#             self.decoder_cell_inf_ = tf.nn.rnn_cell.BasicLSTMCell(self.H)

            # provide the decode embedding parameter from training to inference helper
            self.helper_inf_ = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.decoder_emb_inp_, 
                tf.fill([self.batch_size],self.start_of_sequence_id), # id of Go
                self.end_of_sequence_id # if of EOS
                )
            
            # basic decoder model, connecting the decoder RNN layers and the input prepared by TrainingHelper
#             self.decoder_inf_ = tf.contrib.seq2seq.BasicDecoder(
#                 self.decoder_cell_inf_, self.helper_inf_, encoder_state, output_layer)
            
            self.decoder_inf_ = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cell_, helper=self.helper_, 
                            initial_state=encoder_state, output_layer=self.output_layer_)
            
            # Dynamic decoding, returns (final_outputs, final_state, final_sequence_lengths)
            self.outputs_inf_, _ ,_ = tf.contrib.seq2seq.dynamic_decode(self.decoder_inf_,impute_finished=True) 
                                                                    #maximum_iterations=max_target_sequence_length           
            self.logits_inf_ = self.outputs_inf_.sample_id

#         # Output layer, only computer logits here # I think i need to use projection layer above
#         # logits,[batch_size, max_time, V].
#         with tf.name_scope("Output_Layer"):
#             self.W_out_ = tf.Variable(tf.random_uniform([self.H,self.V], -1.0, 1.0), name="W_out") #hidden_size, V
#             self.b_out_ = tf.Variable(tf.zeros([self.V], dtype=tf.float32), name="b_out")
#             #self.logits_ = tf.reshape(tf.add(matmul3d(self.outputs_, self.W_out_),self.b_out_, name="logits"),
#             #                          [self.batch_size_,self.max_decoder_time_,self.V])


        self.masks_ = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        # Loss computation (true loss, for prediction)
        with tf.name_scope("Cost_Function"): 
            # use weighted softmax cross entropy loss function
            self.loss_ = tf.contrib.seq2seq.sequence_loss(self.logits_, self.decoder_outputs_, self.masks_)
            
#             # Full softmax loss, for scoring
#             self.per_example_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_outputs_, logits=self.logits_,
#                                                                                name="per_example_loss")
#             self.loss_ = tf.reduce_mean(self.per_example_loss_, name="loss")

#             # target_weights is a zero-one matrix of the same size as decoder_outputs. It masks padding positions outside of the target
#             # sequence lengths with values 0   
# #             self.loss_ = (tf.reduce_sum(self.per_example_loss_ * self.target_weights_) / self.batch_size_)

      

    @with_self_graph
    def BuildTrainGraph(self):
        """Construct the training ops.

        You should define:
        - train_loss_ : sampled softmax loss, for training
        - train_step_ : a training op that can be called once per batch

        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).
        """

        # Define approximate loss function.
        # Note: self.softmax_ns (i.e. k=200) is already defined; use that as the
        # number of samples.
            # Loss computation (sampled, for training)
              
        
#         with tf.name_scope("Training_Loss"):
#             self.per_example_train_loss_ = tf.nn.sampled_softmax_loss(weights=tf.transpose(self.W_out_), biases=self.b_out_,
#                                                      labels=tf.expand_dims(tf.reshape(self.logits_,[-1,]), 1), 
#                                                      inputs=tf.reshape(self.logits_,
#                                                      [self.batch_size_*self.max_decoder_time_,self.H]),
#                                                      num_sampled=self.softmax_ns, num_classes=self.V,
#                                                      name="per_example_sampled_softmax_loss")
#             self.train_loss_ = tf.reduce_mean(self.per_example_train_loss_, name="sampled_softmax_loss")


        # Define optimizer and training op
        with tf.name_scope("Training"):
            # calculate and clip gradients
#             self.tvars_ = tf.trainable_variables()
#             self.grads_, _ = tf.clip_by_global_norm(tf.gradients(self.train_loss_, self.tvars_), self.max_grad_norm_)
            
#             # optimization
#             self.optimizer_ = tf.train.AdagradOptimizer(self.learning_rate_)
# #             self.optimizer_ = tf.train.AdamOptimizer(self.learning_rate_)
#             self.train_step_ = self.optimizer_.apply_gradients(zip(self.grads_, self.tvars_)
#                                                                ,global_step=tf.train.get_or_create_global_step())
    
            self.optimizer_ = tf.train.AdamOptimizer(self.learning_rate_)
            self.capped_gradients_ = [(tf.clip_by_value(grad, -1.,1.), var) for grad, var in gradients if grad is not None]
            self.train_step_ = self.optimizer_.apply_gradients(self.capped_gradients_)

        # Initializer step
        init_ = tf.global_variables_initializer()


    @with_self_graph
    def BuildSamplerGraph(self):
        """Construct the sampling ops.

        You should define pred_samples_ to be a Tensor of integer indices for
        sampled predictions for each batch element, at each timestep.

        Hint: use tf.multinomial, along with a couple of calls to tf.reshape
        """
        # Replace with a Tensor of shape [batch_size, max_time, num_samples = 1]
        self.pred_samples_ = None

        #### YOUR CODE HERE ####
        with tf.name_scope("Prediction"):
            self.pred_proba_ = tf.nn.softmax(self.logits_, name="pred_proba")
            self.pred_max_ = tf.argmax(self.logits_, 1, name="pred_max")
            self.pred_samples_ = tf.reshape(tf.multinomial(tf.reshape(self.logits_,[self.batch_size_*self.max_time_,self.V]), 1, name="pred_samples"),[self.batch_size_,self.max_time_,1])
   
        #### END(YOUR CODE) ####


