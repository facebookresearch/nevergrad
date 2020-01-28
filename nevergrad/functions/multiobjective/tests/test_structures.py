from unittest import TestCase

from nevergrad.functions.multiobjective.hypervolume import VectorNode, VectorLinkedList


class TestNode(TestCase):
    def test_initialize(self):
        dim = 4
        node = VectorNode(dim)

        self.assertIsNone(node.coordinate)
        for entry in node.next:
            self.assertTrue(entry is None)
        for entry in node.prev:
            self.assertTrue(entry is None)

        self.assertListEqual(list(node.area), [0.0] * dim)
        self.assertListEqual(list(node.volume), [0.0] * dim)


class TestMultiList(TestCase):
    def setUp(self) -> None:
        self.dim = 4
        self.multilist = VectorLinkedList(dimension=self.dim)

    def test_initialize(self):
        self.assertEqual(self.dim, self.multilist.dimension)
        self.assertIsInstance(self.multilist.sentinel, VectorNode)
        self.assertEqual(4, len(self.multilist.sentinel.prev))
        self.assertEqual(4, len(self.multilist.sentinel.next))
        self.assertEqual(4, len(self.multilist))

        for d in range(self.dim):
            self.assertIs(
                self.multilist.sentinel, self.multilist.sentinel.next[d]
            )
            self.assertIs(
                self.multilist.sentinel, self.multilist.sentinel.prev[d]
            )

        self.assertEqual(
            len(self.multilist.sentinel.next),
            len(self.multilist.sentinel.prev),
        )
        self.assertEqual(
            len(self.multilist.sentinel.next),
            len(self.multilist.sentinel.next[0].next),
        )

    def test_append(self):
        new_node = VectorNode(self.dim)
        self.multilist.append(new_node, 0)

        for i in range(1, self.dim):
            self.assertIsNone(new_node.next[i])
            self.assertIsNone(new_node.prev[i])
            self.assertIs(
                self.multilist.sentinel.next[i], self.multilist.sentinel
            )
            self.assertIs(
                self.multilist.sentinel.prev[i], self.multilist.sentinel
            )
        self.assertIs(new_node.next[0], self.multilist.sentinel)
        self.assertIs(new_node.prev[0], self.multilist.sentinel)
        self.assertIs(self.multilist.sentinel.next[0], new_node)
        self.assertIs(self.multilist.sentinel.prev[0], new_node)

        another_node = VectorNode(self.dim)
        self.multilist.append(another_node, 0)
        for i in range(1, self.dim):
            self.assertIsNone(new_node.next[i])
            self.assertIsNone(new_node.prev[i])
            self.assertIs(
                self.multilist.sentinel.next[i], self.multilist.sentinel
            )
            self.assertIs(
                self.multilist.sentinel.prev[i], self.multilist.sentinel
            )
        self.assertIs(new_node.next[0], another_node)
        self.assertIs(new_node.prev[0], self.multilist.sentinel)
        self.assertIs(self.multilist.sentinel.next[0], new_node)
        self.assertIs(self.multilist.sentinel.prev[0], another_node)

    def test_pop(self):
        new_node = VectorNode(self.dim)
        self.multilist.append(new_node, 0)

        popped_node = self.multilist.pop(new_node, 0+1)
        self.assertIs(popped_node, new_node)
        self.assertIs(new_node.next[0], self.multilist.sentinel)
        self.assertIs(new_node.prev[0], self.multilist.sentinel)
        for i in range(self.dim):
            self.assertIs(
                self.multilist.sentinel.next[i], self.multilist.sentinel
            )
            self.assertIs(
                self.multilist.sentinel.prev[i], self.multilist.sentinel
            )
