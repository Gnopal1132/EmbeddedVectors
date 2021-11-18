import torch
from pathlib import Path
import os
import pickle
from preprocessor import CorpusExtraction, Preprocessing
import numpy as np


# Loading the input
def load_input(path):
    file = open(path, 'rb')
    result = pickle.load(file)
    file.close()
    return result


class EmbeddedVectors:
    def __init__(self, vectorizer_path, model_weight_path, model_graph_path):
        self.vec_path = vectorizer_path
        self.graph_path = model_graph_path
        self.weight_path = model_weight_path
        self.model = self.load_model(self.weight_path, self.graph_path)

        assert Path(self.vec_path).exists(), 'vectorizer.pickle missing!'

        assert Path(self.weight_path).exists(), 'model.ckpt missing!'

        assert Path(self.graph_path).exists(), 'model_graph.pickle is missing!'

    @staticmethod
    def load_model(path_w, path_g):
        file = open(path_g, 'rb')
        model = pickle.load(file)
        file.close()

        return model.load_from_checkpoint(path_w)

    @torch.no_grad()
    def generate_embedding(self, input_frame, vocabsize=100, threshold=0.70):
        self.model.eval()
        corpus_generator = CorpusExtraction(input_frame)
        vocab_size = vocabsize
        threshold = threshold

        # Corpus is a DataFrame
        _, corpus, _ = corpus_generator.return_corpus_with_proportion(vocab_size=vocab_size,
                                                                      create_vocab=False,
                                                                      threshold=threshold)
        preprocessed_corpus = np.array(Preprocessing(corpus.values), dtype=np.float64)

        # Embdedding matrix shape = (number of topics,Dimension of the embedding space)
        # define the matrix containing the topic embeddings,
        embedding_matrix = self.model.alphas.weight

        # theta: """Returns paramters of the variational distribution for \theta.
        #         input: bows
        #                 batch of bag-of-words...tensor of shape bsz x V
        #         output: mu_theta, log_sigma_theta
        #         """

        theta, _ = self.model.encode(torch.from_numpy(preprocessed_corpus).float())
        softmax = torch.nn.Softmax(dim=-1)
        dist_theta = softmax(theta)

        return torch.einsum('ij,jk -> ik', [dist_theta, embedding_matrix])


# Driver Code
frame = load_input(os.path.join(os.curdir, 'tweet.pickle'))  # Loading the dataframe
vec_path = os.path.join(os.curdir, 'vectorizer.pickle')
model_path = os.path.join(os.curdir, 'model.ckpt')
model_graph = os.path.join(os.curdir, 'model_graph.pickle')
vec = EmbeddedVectors(vectorizer_path=vec_path, model_graph_path=model_graph,
                      model_weight_path=model_path)

# Hyperparameters to set
vocab_size = 1000
threshold = 0.70
print(vec.generate_embedding(input_frame=frame, vocabsize=vocab_size, threshold=threshold))
