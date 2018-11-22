import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)
        
        self.fc_out = nn.Linear(hidden_size, vocab_size)
    
    
    def forward(self, features, captions):
        
        ''' Add dim to features
                current shape of features = [batch_size, fc_out]
                    ex: features.shape: torch.Size([10, 256])
                    
                expected shape of features= [batch_size, 1, fc_out]
        '''
        features = features.unsqueeze(1)
        
        ''' do not predict the word after <end> token
                current shape of captions: [batch_size, num of words in a sentence]
                    ex: captions.shape: torch.Size([10, 13])
        '''
        captions = captions[:, :-1]
            
        # word embedding layer for vocab
        cap_embed = self.embed(captions)
        
        assert cap_embed.size(-1) == features.size(-1), 'Error: Size mismatched.'
         
        # combine features and cap_embed as a word sequence
        x = torch.cat((features, cap_embed), dim=1)
        
        # initial hidden 
        hc = self.init_hidden(features.size(0))
        
        x, hc = self.lstm(x, hc)
        
        x = self.fc_out(x)
        
        return x
    
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        
        return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass
