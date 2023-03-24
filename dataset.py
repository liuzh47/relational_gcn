import os
import numpy as np

from dgl.data.utils import extract_archive, download
from dgl.data.knowledge_graph import KnowledgeGraphDataset, build_knowledge_graph

class WN11Dataset(KnowledgeGraphDataset):
    def __init__(self, reverse=True, raw_dir=None, force_reload=False,
                 verbose=True, transform=None):
        name = 'wordnet11'
        super(WN11Dataset, self).__init__(name, reverse, raw_dir,
                                          force_reload, verbose, transform)

    def __getitem__(self, idx):
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, WN18Dataset has only one graph object

        Return
        -------
        :class:`dgl.DGLGraph`

            The graph contains

            - ``edata['e_type']``: edge relation type
            - ``edata['train_edge_mask']``: positive training edge mask
            - ``edata['val_edge_mask']``: positive validation edge mask
            - ``edata['test_edge_mask']``: positive testing edge mask
            - ``edata['train_mask']``: training edge set mask (include reversed training edges)
            - ``edata['val_mask']``: validation edge set mask (include reversed validation edges)
            - ``edata['test_mask']``: testing edge set mask (include reversed testing edges)
            - ``ndata['ntype']``: node type. All 0 in this dataset
        """
        return super(WN11Dataset, self).__getitem__(idx)
        
    def __len__(self):
        r"""The number of graphs in the dataset."""
        return super(WN11Dataset, self).__len__()
        
    def download(self):
        zip_path = os.path.join(self.raw_dir, self.name + '.zip')
        url = "https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/wordnet11.zip"
        download(url, path=zip_path)
        extract_archive(zip_path, self.raw_path)
        
    def process(self):
        """
        The original knowledge base is stored in triplets.
        This function will parse these triplets and build the DGLGraph.
        """
        root_path = self.raw_path
        train_path = os.path.join(root_path, self.name, 'train.txt')
        valid_path = os.path.join(root_path, self.name, 'dev.txt')
        test_path = os.path.join(root_path, self.name, 'test.txt')
        
        entity_dict = {}
        relation_dict = {}
        entity_cnt = 0
        relation_cnt = 0
        
        for triplet in _read_triplets(train_path):
            s = triplet[0]
            r = triplet[1]
            o = triplet[2]
            if s not in entity_dict.keys():
                entity_dict[s] = entity_cnt
                entity_cnt += 1
            
            if o not in entity_dict.keys():
                entity_dict[o] = entity_cnt
                entity_cnt += 1
                
            if r not in relation_dict.keys():
                relation_dict[r] = relation_cnt
                relation_cnt += 1
        
        train = np.asarray(_read_triplets_as_list(train_path, entity_dict, relation_dict))
        valid = np.asarray(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
        test = np.asarray(_read_triplets_as_list(test_path, entity_dict, relation_dict))
        num_nodes = len(entity_dict)
        num_rels = len(relation_dict)
        if self.verbose:
            print("# entities: {}".format(num_nodes))
            print("# relations: {}".format(num_rels))
            print("# training edges: {}".format(train.shape[0]))
            print("# validation edges: {}".format(valid.shape[0]))
            print("# testing edges: {}".format(test.shape[0]))

        # for compatability
        self._train = train
        self._valid = valid
        self._test = test

        self._num_nodes = num_nodes
        self._num_rels = num_rels
        # build graph
        g, data = build_knowledge_graph(num_nodes, num_rels, train, valid, test, reverse=self.reverse)
        etype, ntype, train_edge_mask, valid_edge_mask, test_edge_mask, train_mask, val_mask, test_mask = data
        g.edata['train_edge_mask'] = train_edge_mask
        g.edata['valid_edge_mask'] = valid_edge_mask
        g.edata['test_edge_mask'] = test_edge_mask
        g.edata['train_mask'] = train_mask
        g.edata['val_mask'] = val_mask
        g.edata['test_mask'] = test_mask
        g.edata['etype'] = etype
        g.ndata['ntype'] = ntype
        self._g = g
        
def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line


def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        if triplet[0] in entity_dict and triplet[1] in relation_dict and triplet[2] in entity_dict:
            s = entity_dict[triplet[0]]
            r = relation_dict[triplet[1]]
            o = entity_dict[triplet[2]]
            l.append([s, r, o])
    return l

