import torch
from torch import nn
import torchvision
#from residual_block import ResidualBlock
from torch.nn.utils.weight_norm import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, features_dim, decoder_dim, attention_dim,dropout=0.5):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.features_att = weight_norm(nn.Linear(features_dim, attention_dim))  # linear layer to transform encoded image
        self.decoder_att_1 = weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.decoder_att_2 = weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.full_att = weight_norm(nn.Linear(attention_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, decoder_hidden_1,decoder_hidden_2):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
 
        att1 = self.features_att(image_features)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att_1(decoder_hidden_1)  # (batch_size, attention_dim)
        att3 = self.decoder_att_2(decoder_hidden_2)  # (batch_size, attention_dim )
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1) + att3.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)

        alpha = self.softmax(att)  # (batch_size, num_pixels)

        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class Output(nn.Module):
    def __init__(self,decoder_dims,encoder_dims,output_dims,vocab_size):
        super(Output, self).__init__()

        self.lang_linear = weight_norm(nn.Linear(decoder_dims,output_dims))
        self.att_linear = weight_norm(nn.Linear(decoder_dims,output_dims))
        #self.fc_linear = weight_norm(nn.Linear(encoder_dims,output_dims))
        self.relu = nn.ReLU()
        self.fc = weight_norm(nn.Linear(output_dims*2, vocab_size))  # linear layer to find scores over vocabulary
        self.dropout = nn.Dropout(p=0.5)
        #self.softmax = nn.Softmax(dim=1)
    
    def forward(self,h1,h2):
        l1 = self.lang_linear(h1)
        l2 = self.att_linear(h2)
        #l3 = self.fc_linear(fc.squeeze(1))
        out = self.relu(torch.cat([l1,l2],dim=-1))
        out = self.dropout(out)
        out = self.fc(out)
       
        #out = self.softmax(out)
         
        return out

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, features_dim=2048, dropout=0.5, output_dim = 512,encoder_dim=2048):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()
         
        self.features_dim = features_dim 
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(features_dim, decoder_dim, attention_dim)  # attention network
        self.output = Output(decoder_dim,features_dim,output_dim,vocab_size)
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        #self.image_features_embed = weight_norm(nn.Linear(encoder_dim,embed_dim))     
        #self.image_fc_embed = weight_norm(nn.Linear(encoder_dim,embed_dim))     
        self.dropout = nn.Dropout(p=self.dropout)
        self.language_lstm = nn.LSTMCell(embed_dim +decoder_dim+features_dim, decoder_dim, bias=True)  # decoding LSTMCell

        self.attention_lstm = nn.LSTMCell(features_dim, decoder_dim,bias=True)

        #self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        #self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        #self.f_beta =weight_norm(nn.Linear(decoder_dim, encoder_dim))  # linear layer to create a sigmoid-activated gate
        
        self.fc = weight_norm(nn.Linear(decoder_dim,vocab_size))
        #self.sigmoid = nn.Sigmoid()
        #self.fc_linear = nn.Linear(output_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        #self.fc.bias.data.fill_(0)
        #self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        #mean_encoder_out = encoder_out.mean(dim=1)
        #h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        #c = self.init_c(mean_encoder_out)
        h = torch.zeros(batch_size,self.decoder_dim).to(device)
        c = torch.zeros(batch_size,self.decoder_dim).to(device)
        return h, c

    def forward(self, img_att, img_fc, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        image_features = torch.cat([img_att,img_fc],dim=1)
        #print(encoder_out.size)
        batch_size = image_features.size(0)
        #encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        num_pixels = image_features.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        image_fc = img_fc[sort_ind]
        
        encoded_captions = encoded_captions[sort_ind]
       
        #fc = fc.squeeze(1)
       
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        #image_features = self.image_features_embed(image_features)
        #print(image_features.size())
        #image_fc = self.image_fc_embed(image_fc)
        #print(image_fc.size())
        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        predictions_1 = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            #down lstm(input word and previous word)           
            h1,c1 = self.language_lstm(torch.cat([embeddings[:batch_size_t,t,:],h2[:batch_size_t],image_fc.squeeze(1)[:batch_size_t]],dim=1),(h1[:batch_size_t],c1[:batch_size_t]))
            
            #h1,c1 = self.language_lstm(embeddings[:batch_size_t,t,:],(h1[:batch_size_t],c1[:batch_size_t]))
            preds1 = self.fc(self.dropout(h1)) 

            # attention network 
            attention_weighted_encoding, alpha = self.attention(image_features[:batch_size_t],h1[:batch_size_t],h2[:batch_size_t])

            #gate = self.sigmoid(self.f_beta(h2[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            #attention_weighted_encoding = gate * attention_weighted_encoding
            #print(attention_weighted_encoding.size())
            # top lstm 
            #h2, c2 = self.attention_lstm(
            #    torch.cat([h1[:batch_size_t], attention_weighted_encoding], dim=1),
            #    (h2[:batch_size_t], c2[:batch_size_t]))  # (batch_size_t, decoder_dim)
            h2, c2 = self.attention_lstm(attention_weighted_encoding[:batch_size_t], (h2[:batch_size_t], c2[:batch_size_t]))
            #print(h2.size())
            out = self.output(h1[:batch_size_t],h2[:batch_size_t])
           
            predictions[:batch_size_t, t, :] = out
            predictions_1[:batch_size_t, t, :] = preds1
            alphas[:batch_size_t, t, :] = alpha

        return predictions, predictions_1, encoded_captions, decode_lengths, alphas, sort_ind
