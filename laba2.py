from collections import deque
import matplotlib.pyplot as plt
import random
import numpy as np
import sys
sys.setrecursionlimit(100001)

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BinarySearchTree:
    def __init__(self):
        self.root = None

    # Вставка
    def insert(self, key):
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert_recursive(self.root, key)

    def _insert_recursive(self, node, key):
        if key < node.key:
            if node.left is None:
                node.left = Node(key)
            else:
                self._insert_recursive(node.left, key)
        elif key > node.key:
            if node.right is None:
                node.right = Node(key)
            else:
                self._insert_recursive(node.right, key)

    # Поиск
    def search(self, key):
        return self._search_recursive(self.root, key)

    def _search_recursive(self, node, key):
        if node is None or node.key == key:
            return node
        if key < node.key:
            return self._search_recursive(node.left, key)
        else:
            return self._search_recursive(node.right, key)

    # Удаление
    def delete(self, key):
        self.root = self._delete_recursive(self.root, key)

    def _delete_recursive(self, node, key):
        if node is None:
            return node
        if key < node.key:
            node.left = self._delete_recursive(node.left, key)
        elif key > node.key:
            node.right = self._delete_recursive(node.right, key)
        else:
            # Узел с одним или без потомков
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            # Узел с двумя потомками
            min_larger_node = self._find_min(node.right)
            node.key = min_larger_node.key
            node.right = self._delete_recursive(node.right, min_larger_node.key)
        return node

    def _find_min(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def calculate_height(self, node):
        if node is None:
            return -1
        left_height = self.calculate_height(node.left)
        right_height = self.calculate_height(node.right)
        return max(left_height, right_height) + 1

    def print_tree(self, node, level=0, prefix="Root: "):
        if node is not None:
            # Рекурсивно обходим правое поддерево
            self.print_tree(node.right, level + 1, "R----")
            # Печатаем текущий узел с отступами
            print(" " * (level * 6) + prefix + str(node.key))
            # Рекурсивно обходим левое поддерево
            self.print_tree(node.left, level + 1, "L----")





class AVLNode:
    def __init__(self, key, height=1, left=None, right=None):
        self.key = key
        self.height = height
        self.left = left
        self.right = right


class AVLTree:
    def __init__(self):
        self.root = None

    def get_height(self, node):
        return node.height if node else 0

    def get_balance(self, node):
        return self.get_height(node.left) - self.get_height(node.right) if node else 0

    def update_height(self, node):
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))

    def rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self.update_height(y)
        self.update_height(x)
        return x

    def rotate_left(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self.update_height(x)
        self.update_height(y)
        return y

    def balance(self, node):
        balance_factor = self.get_balance(node)
        if balance_factor > 1:
            if self.get_balance(node.left) < 0:
                node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        if balance_factor < -1:
            if self.get_balance(node.right) > 0:
                node.right = self.rotate_right(node.right)
            return self.rotate_left(node)
        return node

    def insert(self, node, key):
        if not node:
            return AVLNode(key)
        if key < node.key:
            node.left = self.insert(node.left, key)
        elif key > node.key:
            node.right = self.insert(node.right, key)
        else:
            return node  # No duplicates allowed
        self.update_height(node)
        return self.balance(node)

    def delete(self, node, key):
        if not node:
            return node
        if key < node.key:
            node.left = self.delete(node.left, key)
        elif key > node.key:
            node.right = self.delete(node.right, key)
        else:
            if not node.left:
                return node.right
            if not node.right:
                return node.left
            temp = self.get_min_value_node(node.right)
            node.key = temp.key
            node.right = self.delete(node.right, temp.key)
        self.update_height(node)
        return self.balance(node)

    def get_min_value_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current

    def search(self, node, key):
        if not node or node.key == key:
            return node
        if key < node.key:
            return self.search(node.left, key)
        return self.search(node.right, key)

    def insert_key(self, key):
        self.root = self.insert(self.root, key)

    def delete_key(self, key):
        self.root = self.delete(self.root, key)

    def search_key(self, key):
        return self.search(self.root, key)

    def print_tree(self, node, level=0, prefix="Root: "):
        if node is not None:
            # Рекурсивно обходим правое поддерево
            self.print_tree(node.right, level + 1, "R----")
            # Печатаем текущий узел с отступами
            print(" " * (level * 6) + prefix + str(node.key))
            # Рекурсивно обходим левое поддерево
            self.print_tree(node.left, level + 1, "L----")

    def calculate_height(self, node):
        if node is None:
            return -1
        left_height = self.calculate_height(node.left)
        right_height = self.calculate_height(node.right)
        return max(left_height, right_height) + 1



class RedBlackNode:
    def __init__(self, key, color="red", left=None, right=None, parent=None):
        self.key = key
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent


class RedBlackTree:
    def __init__(self):
        self.NIL = RedBlackNode(key=None, color="black")
        self.root = self.NIL

    def get_height(self, node):
        if node is self.NIL:
            return -1  # Пустое поддерево имеет высоту -1
        left_height = self.calculate_height(node.left)
        right_height = self.calculate_height(node.right)

        return max(left_height, right_height) + 1

    def search(self, node, key):
        if node == self.NIL or node.key == key:
            return node
        if key < node.key:
            return self.search(node.left, key)
        return self.search(node.right, key)

    def search_key(self, key):
        return self.search(self.root, key)

    def rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def rotate_right(self, y):
        x = y.left
        y.left = x.right
        if x.right != self.NIL:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x

    def insert(self, key):
        new_node = RedBlackNode(key, color="red", left=self.NIL, right=self.NIL)
        parent = None
        current = self.root

        while current != self.NIL:
            parent = current
            if new_node.key < current.key:
                current = current.left
            else:
                current = current.right

        new_node.parent = parent
        if parent is None:
            self.root = new_node
        elif new_node.key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

        self.fix_insert(new_node)

    def fix_insert(self, node):
        while node != self.root and node.parent.color == "red":
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == "red":
                    node.parent.color = "black"
                    uncle.color = "black"
                    node.parent.parent.color = "red"
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self.rotate_left(node)
                    # Case 3: Node is left child
                    node.parent.color = "black"
                    node.parent.parent.color = "red"
                    self.rotate_right(node.parent.parent)
            else:  # Symmetric to the above
                uncle = node.parent.parent.left
                if uncle.color == "red":
                    node.parent.color = "black"
                    uncle.color = "black"
                    node.parent.parent.color = "red"
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self.rotate_right(node)
                    # Case 3
                    node.parent.color = "black"
                    node.parent.parent.color = "red"
                    self.rotate_left(node.parent.parent)
        self.root.color = "black"

    def transplant(self, u, v):
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def delete(self, key):
        node = self.search(self.root, key)
        if node == self.NIL:
            return
        y = node
        y_original_color = y.color
        if node.left == self.NIL:
            x = node.right
            self.transplant(node, node.right)
        elif node.right == self.NIL:
            x = node.left
            self.transplant(node, node.left)
        else:
            y = self.minimum(node.right)
            y_original_color = y.color
            x = y.right
            if y.parent == node:
                x.parent = y
            else:
                self.transplant(y, y.right)
                y.right = node.right
                y.right.parent = y
            self.transplant(node, y)
            y.left = node.left
            y.left.parent = y
            y.color = node.color
        if y_original_color == "black":
            self.fix_delete(x)

    def fix_delete(self, x):
        while x != self.root and x.color == "black":
            if x == x.parent.left:
                sibling = x.parent.right
                if sibling.color == "red":
                    sibling.color = "black"
                    x.parent.color = "red"
                    self.rotate_left(x.parent)
                    sibling = x.parent.right
                if sibling.left.color == "black" and sibling.right.color == "black":
                    sibling.color = "red"
                    x = x.parent
                else:
                    if sibling.right.color == "black":
                        sibling.left.color = "black"
                        sibling.color = "red"
                        self.rotate_right(sibling)
                        sibling = x.parent.right

                    sibling.color = x.parent.color
                    x.parent.color = "black"
                    sibling.right.color = "black"
                    self.rotate_left(x.parent)
                    x = self.root
            else:
                sibling = x.parent.left
                if sibling.color == "red":
                    sibling.color = "black"
                    x.parent.color = "red"
                    self.rotate_right(x.parent)
                    sibling = x.parent.left
                if sibling.right.color == "black" and sibling.left.color == "black":
                    sibling.color = "red"
                    x = x.parent
                else:
                    if sibling.left.color == "black":
                        sibling.right.color = "black"
                        sibling.color = "red"
                        self.rotate_left(sibling)
                        sibling = x.parent.left
                    sibling.color = x.parent.color
                    x.parent.color = "black"
                    sibling.left.color = "black"
                    self.rotate_right(x.parent)
                    x = self.root
        x.color = "black"

    def minimum(self, node):
        while node.left != self.NIL:
            node = node.left
        return node

    def print_tree(self, node, level=0, prefix="Root: "):
        if node != self.NIL:
            self.print_tree(node.right, level + 1, "R----")
            print(" " * (level * 6) + prefix + f"{node.key} ({node.color})")
            self.print_tree(node.left, level + 1, "L----")

    def calculate_height(self, node):

        if node is self.NIL:
            return -1  # Пустое поддерево имеет высоту -1
        left_height = self.calculate_height(node.left)
        right_height = self.calculate_height(node.right)

        return max(left_height, right_height) + 1


def find_tree_dependence():
    n = []
    h_avl = []
    h_rb = []
    h_bst = []

    bst = BinarySearchTree()

    for i in range(1, 101):

        for _ in range(100):
            while True:
                num = random.randint(0, 10000)
                if not bst.search(num):
                    bst.insert(num)
                    break

        height = bst.calculate_height(bst.root)

        h_bst.append(height)



    for i in range(1, 101):
        keys = list(range(1, 100 * i + 1))

        # AVL
        avl = AVLTree()
        for key in keys:
            avl.insert_key(key)
        h_avl.append(avl.calculate_height(avl.root))


        rb = RedBlackTree()
        for key in keys:
            rb.insert(key)
        h_rb.append(rb.calculate_height(rb.root))

        n.append(100 * i)


    coefficients1 = np.polyfit(n, h_bst, 2)
    coefficients2 = np.polyfit(n, h_avl, 2)
    coefficients3 = np.polyfit(n, h_rb, 2)



    polynomial1 = np.poly1d(coefficients1)
    polynomial2 = np.poly1d(coefficients2)
    polynomial3 = np.poly1d(coefficients3)

    equation1 = f"h = {coefficients1[0]:.10f}n² + {coefficients1[1]:.6f}n + {coefficients1[2]:.2f}"
    equation2 = f"h = {coefficients2[0]:.10f}n² + {coefficients2[1]:.6f}n + {coefficients2[2]:.2f}"
    equation3 = f"h = {coefficients3[0]:.10f}n² + {coefficients3[1]:.6f}n + {coefficients3[2]:.2f}"



    x_fit = np.linspace(min(n), max(n), 500)

    y_fit1 = polynomial1(x_fit)
    y_fit2 = polynomial2(x_fit)
    y_fit3 = polynomial3(x_fit)


    plt.figure(1)
    plt.plot(x_fit, y_fit1, color="blue", label=f"Регрессионный полином {equation1}", linestyle=':')
    plt.step(n, np.log2(n), label='Зависимость h = log(n)', color='purple', linestyle='--', where='post')
    plt.step(n, h_bst, label='Зависимость высоты от ключей (BST)', color='brown')
    plt.xlabel('Количество ключей')
    plt.ylabel('Высота дерева')
    plt.title('Зависимость высоты BST от количества ключей')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    plt.figure(2)
    plt.plot(x_fit, y_fit2, color="blue", label=f"Регрессионный полином {equation2}", linestyle=':')
    plt.step(n, h_avl, label='AVL (Зависимость высоты от ключей)', color='green', where='post')
    plt.step(n, np.log2(n), label='Зависимость h = log(n)', color='purple', linestyle='--', where='post')
    plt.xlabel('Количество ключей')
    plt.ylabel('Высота дерева')
    plt.title('Зависимость высоты AVL-дерева от количества ключей')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    plt.figure(3)
    plt.plot(x_fit, y_fit3, color="blue", label=f"Регрессионный полином {equation3}", linestyle=':')
    plt.step(n, h_rb, label='Красно-черное дерево (Зависимость высоты от ключей)', color='red', where='post')
    plt.step(n, np.log2(n), label='Зависимость h = log(n)', color='purple', linestyle='--', where='post')
    plt.xlabel('Количество ключей')
    plt.ylabel('Высота дерева')
    plt.title('Зависимость высоты Красно-черного дерева от количества ключей')
    plt.grid(True)
    plt.legend()
    plt.show()



def pre_order(root): #прямой
    if root:
        print(root.key, end=" ")
        pre_order(root.left)
        pre_order(root.right)


def in_order(root): #симметричный
    if root:
        in_order(root.left)
        print(root.key, end=" ")
        in_order(root.right)

def post_order(root): #обратный
    if root:
        post_order(root.left)
        post_order(root.right)
        print(root.key, end=" ")

def level_order(root): # в ширину
    if not root:
        return
    queue = deque([root])
    while queue:
        node = queue.popleft()
        print(node.key, end=" ")
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)







################################################################################

find_tree_dependence()

################################################################################

tree = BinarySearchTree()

for i in range(0, 10):
    tree.insert(random.randint(1,1000))

tree.print_tree(tree.root)

print("Прямой обход: ")
pre_order(tree.root)
print('\n')

print("Симметричный обход: ")
in_order(tree.root)
print('\n')

print("Обратный обход: ")
post_order(tree.root)
print('\n')

print("Обход в ширину: ")
level_order(tree.root)
print('\n')




################################################################################

tree = AVLTree()

for i in range(0, 10):
    tree.insert_key(random.randint(1,1000))

tree.print_tree(tree.root)

print("Прямой обход: ")
pre_order(tree.root)
print('\n')

print("Симметричный обход: ")
in_order(tree.root)
print('\n')

print("Обратный обход: ")
post_order(tree.root)
print('\n')

print("Обход в ширину: ")
level_order(tree.root)
print('\n')

################################################################################
tree = RedBlackTree()

for i in range(0, 10):
    tree.insert(random.randint(1,1000))

tree.print_tree(tree.root)

print("Прямой обход: ")
pre_order(tree.root)
print('\n')

print("Симметричный обход: ")
in_order(tree.root)
print('\n')

print("Обратный обход: ")
post_order(tree.root)
print('\n')

print("Обход в ширину: ")
level_order(tree.root)
print('\n')