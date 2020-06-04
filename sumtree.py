import random


class _Node():
    _idx = 0
    def __init__(self, value, children=[], active=True):
        self.parent = None
        self.children = children
        for child in self.children:
            child.parent = self
        self.value = value
        if not self.children:
            self.idx = _Node._idx
            _Node._idx += 1
        self.active = active
        self.experience = None

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class PositiveSumTree():
    def __init__(self, maxlen):
        self._pointer = 0
        _Node._idx = 0
        self.leafs = [_Node(0, active=False) for _ in range(maxlen)]
        self._pointer = 0
        self.size = 0
        self.maxlen = maxlen

        def _layer(array):
            layer = []
            if len(array) == 1:
                self.root = array[0]
                return
            for child1, child2 in zip(array[0::2], array[1::2]):
                layer.append(_Node(child1.value+child2.value, children=[child1, child2],
                                  active=child1.active or child2.active))
            if len(array) % 2:
                layer.append(_Node(array[-1].value, children=[array[-1]], active=array[-1].active))
            _layer(layer)
        _layer(self.leafs)

    def retrieve(self, value):
        def _retrieve(node, value):
            if not node.children:
                return node
            if len(node.children) == 1 or node.children[0].value >= value or not node.children[1].active:
                return _retrieve(node.children[0], value)
            else:
                return _retrieve(node.children[1], value - node.children[0].value)
        return _retrieve(self.root, value)

    def update_leaf(self, leaf, value):
        assert value >= 0
        leaf = self.leafs[leaf.idx]
        def _inc(node, value):
            node.active = True
            node.value += float(value)
            if node.parent:
                _inc(node.parent, value)
        _inc(leaf.parent, value-leaf.value)
        leaf.value = float(value)

    def sample(self, k):
        step = self.root.value / k
        values = [random.uniform(i * step, i * step + step) for i in range(1, k+1)]
        return [self.retrieve(value) for value in values]

    def append(self, value, experience):
        assert value >= 0
        self.update_leaf(self.leafs[self._pointer], value)
        self.leafs[self._pointer].experience = experience
        self.leafs[self._pointer].active = True
        self._pointer += 1
        self._pointer %= len(self.leafs)
        self.size = min(self.size + 1, self.maxlen)


if __name__ == '__main__':
    import time

    prev_time = time.time()
    maxlen = 1000
    tree = PositiveSumTree(maxlen=maxlen)
    print(f'Initialiazed sum tree with {maxlen} nodes in {time.time() - prev_time} secs')

    print(f'Size of the tree is {tree.size}')
    n_append = 1000
    prev_time = time.time()
    for _ in range(n_append):
        tree.append(random.random(), random.random())
    print(f'Appended {n_append} nodes to the tree in {time.time() - prev_time} secs')
    assert tree.size == maxlen
    print(f'Size of the tree is {tree.size}')

    prev_time = time.time()
    node = tree.retrieve(1)
    print(f'Retrieved value in {time.time() - prev_time} secs')

    prev_time = time.time()
    n = 32
    nodes = tree.sample(n)
    print(f'Sampled {n} nodes in {time.time() - prev_time} secs')

    prev_time = time.time()
    map(tree.update_leaf, zip(nodes, [random.random() for _ in range(n)]))
    print(f'Updated sampled nodes in {time.time() - prev_time} secs')

    print(f'Cumilative Sum: {tree.root.value}')
