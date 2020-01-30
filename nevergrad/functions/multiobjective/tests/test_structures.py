from nevergrad.functions.multiobjective.hypervolume import VectorNode, VectorLinkedList


def test_initialize():
    dim = 4
    node = VectorNode(dim)

    assert node.coordinate is None
    for entry in node.next:
        assert entry is None
    for entry in node.prev:
        assert  entry is None

    assert list(node.area) == [0.0] * dim
    assert list(node.volume) == [0.0] * dim


def test_initialize():
    dim = 4
    multilist = VectorLinkedList(dimension=dim)

    assert dim == multilist.dimension
    assert isinstance(multilist.sentinel, VectorNode)
    assert len(multilist.sentinel.prev) == 4
    assert len(multilist.sentinel.next) == 4
    assert len(multilist) == 4

    for d in range(dim):
        assert multilist.sentinel is multilist.sentinel.next[d]
        assert multilist.sentinel is multilist.sentinel.prev[d]

    assert len(multilist.sentinel.next) == len(multilist.sentinel.prev)
    assert len(multilist.sentinel.next) == len(multilist.sentinel.next[0].next)


def test_append():
    dim = 4
    multilist = VectorLinkedList(dimension=dim)

    new_node = VectorNode(dim)
    multilist.append(new_node, 0)

    for i in range(1, dim):
        assert new_node.next[i] is None
        assert new_node.prev[i] is None
        assert multilist.sentinel.next[i] is multilist.sentinel
        assert multilist.sentinel.prev[i] is multilist.sentinel

    assert new_node.next[0] is multilist.sentinel
    assert new_node.prev[0] is multilist.sentinel
    assert multilist.sentinel.next[0] is new_node
    assert multilist.sentinel.prev[0] is new_node

    another_node = VectorNode(dim)
    multilist.append(another_node, 0)
    for i in range(1, dim):
        assert new_node.next[i] is None
        assert new_node.prev[i] is None
        assert multilist.sentinel.next[i] is multilist.sentinel
        assert multilist.sentinel.prev[i] is multilist.sentinel

    assert new_node.next[0] is another_node
    assert new_node.prev[0] is multilist.sentinel
    assert multilist.sentinel.next[0] is new_node
    assert multilist.sentinel.prev[0] is another_node


def test_pop():
    dim = 4
    multilist = VectorLinkedList(dimension=dim)

    new_node = VectorNode(dim)
    multilist.append(new_node, 0)

    popped_node = multilist.pop(new_node, 0+1)
    assert popped_node is new_node
    assert new_node.next[0] is multilist.sentinel
    assert new_node.prev[0] is multilist.sentinel
    for i in range(dim):
        assert multilist.sentinel.next[i] is multilist.sentinel
        assert multilist.sentinel.prev[i] is multilist.sentinel
