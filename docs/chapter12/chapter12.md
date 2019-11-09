111. [Minimum Depth of Binary Tree](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/solution/er-cha-shu-de-zui-xiao-shen-du-by-leetcode/)

```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if root == None:
            return 0

        if root.left == None or root.right == None:
            return self.minDepth(root.left) + self.minDepth(root.right) + 1

        return min(self.minDepth(root.right), self.minDepth(root.left)) + 1
```
- 注意叶子结点的定义:没有子节点的节点

112. [Path Sum](https://leetcode-cn.com/problems/path-sum/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-26/)
```python
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        if not root.left and not root.right and root.val == sum:
            return True
        sum -= root.val

        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)
```

113. [Path Sum II](https://leetcode-cn.com/problems/path-sum-ii/solution/lu-jing-zong-he-iipython-by-fei-ben-de-cai-zhu-uc4/)
```python
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        stack = []
        if not root:
            return []

        def help(root, sum, tmp):
            if not root:
                return []
            if not root.left and not root.right and sum - root.val == 0:
                tmp += [root.val]
                stack.append(tmp)
            sum -= root.val
            help(root.left, sum, tmp + [root.val])
            help(root.right, sum, tmp + [root.val])

        help(root, sum, [])
        return stack
```


114. [Flatten Binary Tree to Linked List](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/submissions/)
```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        while root is not None:
            if root.left is not None:   # 如果左子树不为空,那么找到左子树最右节点
                pre_rigth = root.left
                while pre_rigth.right is not None:
                    pre_rigth = pre_rigth.right
                pre_rigth.right = root.right # 将左子树的最右节点的指向root的右孩子
                root.right = root.left
                root.left = None
            root = root.right   # 继续下一个节点
```
- 递归的将root的左子树链接到右子树上

115. [Distinct Subsequences](https://leetcode-cn.com/problems/distinct-subsequences/)
```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        l1, l2 = len(s) + 1, len(t) + 1
        cur = [0] * l2
        cur[0] = 1
        for i in range(1, l1):
            pre = cur[:]
            for j in range(1, l2):
                cur[j] = pre[j] + pre[j - 1] * (s[i - 1] == t[j - 1])
        return cur[-1]
```


116. [Populating Next Right Pointers in Each Node](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/)
```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if root and root.left:
            root.left.next = root.right

            if root.next:
                root.right.next = root.next.left

            self.connect(root.left)
            self.connect(root.right)

        return root
```
- 任意一次递归，只需要考虑子节点的 next 属性：
    1. 将左子节点连接到右子节点
    2. 将右子节点连接到 root.next 的左子节点
    3. 递归左右节点

117. [Populating Next Right Pointers in Each Node II](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node-ii/)
```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if root and (root.left or root.right):
            if root.left and root.right:
                root.left.next = root.right
            node = root.right or root.left
            head = root.next
            while head and not (head.left or head.right):
                head = head.next
            node.next = head and (head.left or head.right)

            self.connect(root.right)
            self.connect(root.left)

        return root
```
- 任意一次递归,设置子节点的next属性有三种情况:
    1. 没有子节点:直接返回
    2. 一个子节点：将这个子节点的 next 属性设置为同层的下一个节点，即为 root.next 的最左边的一个节点，如果 root.next 没有子节点，则考虑 root.next.next，依次类推
    3. 两个子节点:左子节点指向右子节点，然后右子节点同第二种情况的做法

118. [Pascal's Triangle](https://leetcode-cn.com/problems/pascals-triangle/)
```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        r = [[1]]
        for i in range(1, numRows):
            r.append([1] + [sum(r[-1][j:j + 2]) for j in range(i)])
        return numRows and r or []
```

119. [Pascal's Triangle II](https://leetcode-cn.com/problems/pascals-triangle-ii/)

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        r = [1]
        for i in range(1, rowIndex + 1):
            r = [1] + [sum(r[j:j + 2]) for j in range(i)]
        return r
```


120. [Triangle](https://leetcode-cn.com/problems/triangle/)
```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle:
            return
        res = triangle[-1]  # 保存每行结果

        # 自底向上，从倒数第二行开始遍历
        for i in range(len(triangle) - 2, -1, -1):
            for j in range(len(triangle[i])):
                res[j] = min(res[j], res[j + 1]) + triangle[i][j]
        return res[0]
```