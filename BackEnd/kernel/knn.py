import torch as th


def randomize_tensor(tensor):
    return tensor[th.randperm(len(tensor))]

def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
    
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = th.pow(x - y, p).sum(2)
    
    return dist

class NN():

    def __init__(self, X = None, Y = None, p = 2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = distance_matrix(x, self.train_pts, self.p) ** (1/self.p)
        labels = th.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):

    def __init__(self, X = None, Y = None, k = 3, p = 2,d = 1e-3):
        self.k = k
        self.d = d
        super().__init__(X, Y, p)
    
    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = distance_matrix(x, self.train_pts, self.p) ** (1/self.p)

        value,indice = dist.topk(self.k, largest=False)

        mask = value<self.d
        #value = value[mask]
        #indice = indice[mask]

        votes = self.train_label[indice]

        winner = th.zeros(votes.size(0), dtype=votes.dtype, device=votes.device) -1
        count = th.zeros(votes.size(0), dtype=votes.dtype, device=votes.device)

        for lab in self.unique_labels:
            vote_count = th.logical_and((votes == lab),mask).sum(1)
            who = vote_count > count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner

if __name__ == '__main__':
    import torch
    result1 = torch.rand(3,10)
    label = torch.Tensor(list(range(len(result1)))).long().cuda()
    cluster = KNN(result1.cpu(),label.cpu(),k=2,p=2,d=1e-3)
    result2 = cluster(result1+1e-12)
    print(result2)
