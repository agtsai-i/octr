import random
import numpy as np
import math
from scipy.special import digamma

# random crap that should be factored out later
lu=2
lv=2
su=0.5
sv=8
a0=0.5
b0=0.45

class OnlineCTR(object):    
    def __init__(self, num_topics, num_users, vocab_size):
        self.K = num_topics
        self.U = num_users
        self.T = vocab_size
        self.J = 4   # number of iterations for Gibbs sampling
        self.JB = 1  # number of burn in iterations
        
    def initialize(self, corpus):
        '''
        Initialize necessary matrices, given a corpus
        corpus: list of document vectors, where each document vector is a list of word ids
        
        corkus.random01() is equivalent to random.random()
        # https://stackoverflow.com/questions/43077030/mersenne-twister-in-python
        words -> words assigned to topics -> topic counts per document
        corpus -> Z -> CDK
        
        Note the two alternative weight calculations in update_z_joint() !!
        
        has to be a better OO way of doing this
        rather than index values passed everywhere       
        '''
        self.corpus = corpus
        self.V = len(corpus)                              # number of documents        
        self.Z = [np.zeros(len(doc)) for doc in corpus]   # topic assignments for each word in doc         
        self.CDK = np.zeros([self.V, self.K])             # count-document-topic: for each document, number of words assigned to each topic
        
        # Assign words to random topics, and initialize
        for doc_index in range(self.V):
            for word_index in range(len(self.corpus[doc_index])):
                random_topic = random.choice(range(self.K))
                self.Z[doc_index][word_index] = random_topic
                self.CDK[doc_index][random_topic] += 1
        
        # initialize the mean vectors
        # and just the diagonals of the covariance matrices
        # how the F is it that these are all initialized, and yet saves mem?
        self.U_mean = np.zeros([self.U, self.K])
        self.U_cov = np.zeros([self.U, self.K])
        self.V_mean = np.zeros([self.V, self.K])
        self.V_cov = np.zeros([self.V, self.K])
        
        # initialize to random values
        # np.random.multivariate_normal would probably be smarter
        rows, cols = self.U_mean.shape
        for r in range(rows):
            for c in range(cols):
                self.U_cov[r][c] = su * su
                self.U_mean[r][c] = np.random.random()

        rows, cols = self.V_mean.shape
        for r in range(rows):
            for c in range(cols):
                self.V_cov[r][c] = sv * sv
                self.V_mean[r][c] = np.random.rand()                
        
        # topic vectors
        self.phi = np.zeros([self.K, self.T])
        self.stat_phi = np.zeros([self.K, self.T])
        self.phi_sum = np.zeros(self.K)
        self.stat_phi_list_k = []
        self.stat_phi_list_t = []
        
        # parallels phi
        # maybe initial values aren't right
        # https://github.com/kzhai/PyLDA/blob/master/hybrid.py
        self.gamma = 1 * np.random.gamma(100., 1./100., (self.K, self.T))
        
    def feed_one(self, user_index, item_index, rating):
        '''
        Given a rating,
        update everything
        '''        
        user_vector = self.U_mean[user_index, :]
        item_vector = self.V_mean[item_index, :]
        rating_hat = np.dot(user_vector, item_vector)
        delta = rating - rating_hat
        self.update_topic_modeling(item_index)
        self.update_user_and_item_vectors(user_index, item_index, rating, delta)
        
    def update_topic_modeling(self, item_index):
        # Gibbs sampling!
        # Given one document, pretend the entire corpus consists of this one document
        # and update the probabilities accordingly (I think)
        for j in range(self.J):
            self.update_z_joint(item_index)            
            if j < self.JB:
                continue
            else:
                self.collect_phi(j == self.JB, item_index)
        self.calc_phi(j - self.JB)
        
    def update_z_joint(self, item_index):
        document = self.corpus[item_index]
        weights = np.zeros(self.K)
        N = len(document)
        cdk = self.CDK[item_index]
        
        for word_index in range(N):
            word_id = document[word_index]
            word_k = self.Z[item_index][word_index]
            cdk[word_k] -= 1
            
            for k in range(self.K):
                if k == 0:
                    cul = 0
                else:
                    cul = weights[k-1]
                weights[k] = (
                    cul + (cdk[k] + a0)
                    
                    # strategy 1 variational optimal distribution
                    # references: 
                    # https://github.com/blei-lab/onlineldavb/blob/master/onlineldavb.py#L34
                    # https://github.com/blei-lab/onlineldavb/blob/master/onlineldavb.py#L135 
                    # https://arxiv.org/pdf/1601.00670.pdf  (eq 59)
                    # https://arxiv.org/pdf/1206.7051.pdf (eq 29)
                    # BUT WHERE IS GAMMA UPDATED? Does it need to be updated?
                    * math.exp(digamma(b0+self.gamma[k][word_id])-digamma(b0*self.T+self.gamma[k,:].sum()))
                    
                    # strategy 2 approximation that does not require digamma()
                    # No idea where this comes from
                    # * (b0 + self.phi[k][word_id])/(b0 * self.T + self.phi_sum[k])
                    * math.exp(0.5/sv/sv/(float(N)*(2.0*self.V_mean[item_index][k]-(1.0+2.0*cdk[k])/float(N))))
                    )
            # abnormality tests
            
            # wtf is happening here?
            # find a weight that is that is biggest so far (?)
            # this whole weight update section is just weird
            sel = weights[self.K-1] * np.random.random()
            seli = 0
            while(weights[seli] < sel):
                seli += 1
            self.Z[item_index][word_index] = seli
            cdk[word_k] += 1
            
    def collect_phi(self, reset, item_index):
        if reset:
            self.stat_phi = np.zeros([self.K, self.T])
            self.stat_phi = np.zeros([self.K, self.T])
            self.stat_phi_list_k = []
            self.stat_phi_list_t = []
            
        document = self.corpus[item_index]    
        z = self.Z[item_index]
        for i in range(len(document)):
            k = z[i]            
            word_id = document[i]
            if self.stat_phi[k][word_id] == 0:
                self.stat_phi_list_k.append(k)
                self.stat_phi_list_t.append(word_id)
            self.stat_phi[k][word_id] += 1
            
    def calc_phi(self, rounds):
        for k, t in zip(self.stat_phi_list_k, self.stat_phi_list_t):
            self.phi[k][t] += self.stat_phi[k][t] / float(rounds)
            self.phi_sum[k] += self.stat_phi[k][t] / float(rounds)
            
    def update_user_and_item_vectors(self, user_index, item_index, rating, delta):
        new_u_mean = np.zeros(self.K)
        new_u_cov = np.zeros(self.K)
        new_v_mean = np.zeros(self.K)
        new_v_cov = np.zeros(self.K)
                
        denomi_u = lu * lu
        denomi_v = lu * lu
        
        u_mean = self.U_mean[user_index]
        u_cov = self.U_cov[user_index]
        v_mean = self.V_mean[item_index]
        v_cov = self.V_cov[item_index]
        cdk = self.CDK[item_index]

        for k in range(self.K):
            denomi_u += v_mean[k] * u_cov[k] * v_mean[k]
            denomi_v += u_mean[k] * v_cov[k] * u_mean[k]
            
        for k in range(self.K):
            new_u_cov[k] = 1.0 / (1.0 / u_cov[k] + v_mean[k] * v_mean[k] / lv / lv)
            new_u_mean[k] = delta / denomi_u * u_cov[k] * v_mean[k]
            new_v_cov[k] = 1.0 / (1.0 / v_cov[k] + 1.0 / sv / sv + u_mean[k] * u_mean[k] / lv / lv)
            
        v_cov_mix = np.zeros(self.K)
        for k in range(self.K):
            v_cov_mix[k] = 1.0 / (1.0 / v_cov[k] + 1.0 / sv / sv)
            
        denomi_mix = 1.0
        nomi_op1 = 0.0
        nomi_op2 = 0.0
        for k in range(self.K): 
            nomi_op1 += u_mean[k] * v_cov_mix[k] / v_cov[k] * v_mean[k]
            nomi_op2 += u_mean[k] * v_cov_mix[k] / sv / sv * cdk[k] / len(self.corpus[item_index])
            denomi_mix += u_mean[k] * v_cov_mix[k] / lv / lv * u_mean[k]
        mult = (nomi_op1 + 1. * nomi_op2 - rating) / denomi_mix

        for k in range(self.K):
            v_mean[k] = (v_cov_mix[k] / v_cov[k] * v_mean[k] +
                         1.0 * v_cov_mix[k] / sv / sv * cdk[k] / len(self.corpus[item_index]) -
                         v_cov_mix[k] / lu / lu * u_mean[k] * mult)
            u_cov[k] = new_u_cov[k]
            u_mean[k] += new_u_mean[k]  # oh this + is interesting
            v_cov[k] = new_v_cov[k]
