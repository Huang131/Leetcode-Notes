101. [Symmetric Tree](https://leetcode-cn.com/problems/symmetric-tree/)
```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def isSym(L, R):
            if not L and not R: return True
            if L and R and L.val == R.val:
                return isSym(L.left, R.right) and isSym(L.right, R.left)
            return False

        return isSym(root, root)
```
102. [Binary Tree Level Order Traversal](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)
```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:

        ans, level = [], [root]
        while root and level:
            ans.append([node.val for node in level])
            level = [kid for n in level for kid in (n.left, n.right) if kid]
        return ans
```
103. [Binary Tree Zigzag Level Order Traversal](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)
```python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        ans, level = [], [root]
        while root and level:
            ans.append([node.val for node in level])
            level = [kid for n in level for kid in (n.left, n.right) if kid]
        ans = [k if i % 2 == 0 else k[::-1] for i, k in enumerate(ans)]
        return ans
```
- 层次遍历，对结果进行列表推导，反转奇数层


105. [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not inorder:
            return None
        root = TreeNode(preorder[0])
        n = inorder.index(root.val)

        root.left = self.buildTree(preorder[1:n + 1], inorder[:n])
        root.right = self.buildTree(preorder[n + 1:], inorder[n + 1:])

        return root
```




107. [Binary Tree Level Order Traversal II](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/)
```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:

        ans, level = [], [root]
        while root and level:
            ans.append([node.val for node in level])
            level = [kid for n in level for kid in (n.left, n.right) if kid]
        return ans[::-1]
```

108. [Convert Sorted Array to Binary Search Tree](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/solution/dfsdi-gui-er-fen-fa-by-chencyudel/)
```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if not nums:
            return None

        mid = len(nums) // 2

        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid + 1:])

        return root
```

- 平衡二叉搜索树：每个节点的左右子树都高度差在1以内，每个节点左子树小于右子树
- 每个节点当做根节点的时候，左子树形成的数组一定比它小，右子树形成的数组一定比他大


109. [Convert Sorted List to Binary Search Tree](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/solution/you-xu-lian-biao-zhuan-huan-er-cha-sou-suo-shu-pyt/)
```python
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        if head is None:
            return
        node_arr = []
        while head:
            node_arr.append(head.val)
            head = head.next

        def buildBST(nums):
            if len(nums) == 0:
                return
            mid = len(nums) // 2
            root = TreeNode(nums[mid])

            root.left = buildBST(nums[:mid])
            root.right = buildBST(nums[mid + 1:])
            return root

        return buildBST(node_arr)
```
- 链表转化为数组后,采用108题的做法

110. [Balanced Binary Tree](https://leetcode-cn.com/problems/balanced-binary-tree/)
```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def check(root):
            if root is None:
                return 0
            left = check(root.left)
            right = check(root.right)
            if left == -1 or right == -1 or abs(left - right) > 1:
                return -1
            return 1 + max(left, right)

        return check(root) != -1
```