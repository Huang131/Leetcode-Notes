92. [Reverse Linked List II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)
```python
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        if m == n:
            return head

        dummyNode = ListNode(-1)
        dummyNode.next = head
        pre = dummyNode

        for i in range(m - 1):
            pre = pre.next

        reverse = None
        cur = pre.next
        for i in range(n - m + 1):
            next = cur.next
            cur.next = reverse
            reverse = cur
            cur = next

        pre.next.next = cur
        pre.next = reverse

        return dummyNode.next
```

93. [Restore IP Addresses](https://leetcode-cn.com/problems/restore-ip-addresses/solution/bao-li-he-hui-su-by-powcai/)
```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        self.dfs(s, 0, "", res)
        return res

    def dfs(self, s, index, path, res):
        if index == 4:
            if not s:
                res.append(path[:-1])
            return
        for i in range(1, 4):
            if i <= len(s):  # i要小于s的长度
                if i == 1:
                    self.dfs(s[i:], index + 1, path + s[:i] + ".", res)
                elif i == 2 and s[0] != "0":  # 选择两个数字时,不能以0开头
                    self.dfs(s[i:], index + 1, path + s[:i] + ".", res)
                elif i == 3 and s[0] != "0" and int(s[:3]) <= 255:
                    self.dfs(s[i:], index + 1, path + s[:i] + ".", res)
```


94. [Binary Tree Inorder Traversal](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)
```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        r = []
        stack = []

        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                root = stack.pop()
                r.append(root.val)
                root = root.right
        return r
```

95. [Unique Binary Search Trees II](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/solution/zi-ding-xiang-xia-by-powcai/)
```python
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        def node(val, left, right):
            node = TreeNode(val)
            node.left = left
            node.right = right
            return node

        def trees(first, last):
            return [
                node(root, left, right) for root in range(first, last + 1)
                for left in trees(first, root - 1)
                for right in trees(root + 1, last)
            ] or [None]

        if n == 0:
            return []
        return trees(1, n)
```
- 递归，root遍历取完1到n，左子树为[1,root-1]的组合可能，右子树为[root+1,n]的组合可能


96. [Unique Binary Search Trees](https://leetcode-cn.com/problems/unique-binary-search-trees/)
```python
class Solution:
    def numTrees(self, n: int) -> int:
        res = [0] * (n + 1)
        res[0] = 1
        for i in range(1, n + 1):
            for j in range(i):
                res[i] += res[j] * res[i - 1 - j]
        return res[n]
```
- 动态规划，遍历1-n，个数为左右子树的乘积
```python
def numTrees(self, n):
    return math.factorial(2*n)/(math.factorial(n)*math.factorial(n+1))
```
- 公式法，[卡特兰数](https://baike.baidu.com/item/catalan/7605685?fr=aladdin) $C_0=1,C_{n+1} = \frac{2(2n+1)}{n+2}C_n$




97. [Interleaving String](https://leetcode-cn.com/problems/interleaving-string/)
```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        r, c, l = len(s1), len(s2), len(s3)
        if r + c != l:  # 长度不匹配
            return False

        dp = [True for _ in range(c + 1)]

        for j in range(1, c + 1):
            dp[j] = dp[j - 1] and s2[j - 1] == s3[j - 1]

        for i in range(1, r + 1):
            dp[0] = (dp[0] and s1[i - 1] == s3[i - 1])
            for j in range(1, c + 1):
                dp[j] = (dp[j] and s1[i - 1] == s3[i - 1 + j]) or (
                    dp[j - 1] and s2[j - 1] == s3[i - 1 + j])

        return dp[-1]
```

98. [Validate Binary Search Tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)
```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        output = []
        self.inOrder(root, output)

        return all([a > b for a, b in zip(output[1:], output)])

    def inOrder(self, root, output):
        if root is None:
            return

        self.inOrder(root.left, output)
        output.append(root.val)
        self.inOrder(root.right, output)
```
- 中序遍历结果为升序



99. [Recover Binary Search Tree](https://leetcode-cn.com/problems/recover-binary-search-tree/)
```python
class Solution:
    def __init__(self):
        self.res = []

    def recoverTree(self, root):
        self.mid(root)
        node1 = None
        node2 = None
        for i in range(len(self.res) - 1):
            if self.res[i].val > self.res[i + 1].val and node1 == None:
                node1 = self.res[i]
                node2 = self.res[i + 1]
            elif self.res[i].val > self.res[i + 1].val and node1 != None:
                node2 = self.res[i + 1]

        node1.val, node2.val = node2.val, node1.val

    def mid(self, root):
        if root is not None:
            self.mid(root.left)
            self.res.append(root)
            self.mid(root.right)
```
- 中序遍历，如果有一个降序对，交换这两个node；若有两个降序对，说明第一对的前一个node和第二对的后一个node需要交换。

100. [Same Tree](https://leetcode-cn.com/problems/same-tree/)
```python
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if p and q and p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(
                p.right, q.right)
        else:
            return p is q
```