149. [Max Points on a Line](https://leetcode-cn.com/problems/max-points-on-a-line/)
```python
from math import gcd
from collections import Counter


class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        if len(points) < 2:
            return len(points)
        points = [tuple(x) for x in points]
        P = Counter(points)

        def slop(p1, p2):  # 求出两点之间的斜率
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            if dx == 0:
                return (0, 1)
            if dy == 0:
                return (1, 0)
            if dx < 0:
                dx = -dx
                dy = -dy
            g = gcd(dx, dy)  # 最大公约数
            return (dx // g, dy // g)

        lines = [Counter() for _ in range(len(points))]
        for i in range(1, len(points)):
            for j in range(i):
                if points[j] == points[i]:
                    continue
                k = slop(points[j], points[i])
                # 记录过该点以k为斜率的直线个数
                lines[i][k] += 1 
                lines[j][k] += 1
        ans = 0
        for i, l in enumerate(lines):
            ans = max(ans, max(l.values(), default=0) + P[points[i]])
        return ans
```

1.   [Best Time to Buy and Sell Stock II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        return sum(b - a for a, b in zip(prices, prices[1:]) if b > a)
```
- 把连续上升拆分成多个上升,当明天的价格大于今天的价格就会产生利润，循环内计算的就是每次变化获取到的利润。

1.   [Best Time to Buy and Sell Stock III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        n = len(prices)
        # 状态转移方程 dp[k][i]:到底i天经过k次交易得到的最大利润
        dp = [[0] * n for _ in range(3)]
        for k in range(1, 3):  # k 为交易次数
            pre_max = -prices[0]  # 处理边界情况
            for i in range(1, n):  # i 为交易天数
                pre_max = max(pre_max, dp[k - 1][i - 1] - prices[i])
                dp[k][i] = max(dp[k][i - 1], prices[i] + pre_max)
        return dp[-1][-1]
```


1.   [Binary Tree Maximum Path Sum](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)
```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        def maxend(node):
            if not node:
                return 0
            left = maxend(node.left)
            right = maxend(node.right)
            self.max = max(self.max, left + node.val + right)
            return max(node.val + max(left, right), 0)

        self.max = float('-inf')
        maxend(root)
        return self.max
```


1.     [Valid Palindrome](https://leetcode-cn.com/problems/valid-palindrome/)
```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        while l < r:
            while l < r and not s[l].isalnum():
                l += 1
            while l < r and not s[r].isalnum():
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        return True
```

1.   [Word Ladder II](https://leetcode-cn.com/problems/word-ladder-ii/solution/bfs-level-order-traverse-by-matrix95/)
```python
import collections
import string
class Solution:
    def findLadders(self, beginWord: str, endWord: str,
                    wordList: List[str]) -> List[List[str]]:
        wordList = set(wordList)  # 转化为set实现O(1)的in判断
        if endWord not in wordList:
            return []
        level = {beginWord}
        parents = collections.defaultdict(set)
        while level and endWord not in parents:
            next_level = collections.defaultdict(set)
            for node in level:
                for char in string.ascii_lowercase:
                    for i in range(len(beginWord)):
                        n = node[:i] + char + node[i + 1:]
                        if n in wordList and n not in parents:
                            next_level[n].add(node)
            level = next_level
            parents.update(next_level)
        res = [[endWord]]
        while res and res[0][0] != beginWord:
            res = [[p] + r for r in res for p in parents[r[0]]]
        return res
```
- 注意要把wordList转化为set,实现O(1)的in判断,否者超时
- 递归输出res


1.   [Word Ladder](https://leetcode-cn.com/problems/word-ladder/solution/dan-ci-jie-long-by-leetcode/)
```python
from collections import deque
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        def construct_dict(word_list):
            d = {}
            for word in word_list:
                for i in range(len(word)):
                    s = word[:i] + "_" + word[i + 1:]
                    d[s] = d.get(s, []) + [word]
            return d

        def bfs_words(begin, end, dict_words):
            queue, visited = deque([(begin, 1)]), set()
            while queue:
                word, steps = queue.popleft()
                if word not in visited:
                    visited.add(word)
                    if word == end:
                        return steps
                    for i in range(len(word)):
                        s = word[:i] + "_" + word[i + 1:]
                        neigh_words = dict_words.get(s, [])
                        for neigh in neigh_words:
                            if neigh not in visited:
                                queue.append((neigh, steps + 1))
            return 0

        d = construct_dict(wordList or set([beginWord, endWord]))
        return bfs_words(beginWord, endWord, d)
```
- 将问题抽象在一个无向无权图中，每个单词作为节点，差距只有一个字母的两个单词之间连一条边。问题变成找到从起点到终点的最短路径
- 算法中最重要的步骤是找出相邻的节点，也就是只差一个字母的两个单词。为了快速的找到这些相邻节点，我们对给定的 wordList 做一个预处理，将单词中的某个字母用 '-' 代替

1.     [Longest Consecutive Sequence](https://leetcode-cn.com/problems/longest-consecutive-sequence/)
```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        used = {x: False for x in nums}
        longest = 0
        for i in used:
            if used[i] == False:
                curr, lenright = i + 1, 0
                while curr in used:
                    lenright += 1
                    used[curr] = True
                    curr += 1
                curr, lenleft = i - 1, 0
                while curr in used:
                    lenleft += 1
                    used[curr] = True
                    curr -= 1
                longest = max(longest, lenleft + 1 + lenright)
        return longest
```

1.   [Sum Root to Leaf Numbers](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)
```python
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        if not root:
            return 0
        stack, res = [(root, root.val)], 0
        while stack:
            node, val = stack.pop()
            if node:
                if not node.left and not node.right:
                    res += val
                if node.left:
                    stack.append((node.left, val * 10 + node.left.val))
                if node.right:
                    stack.append((node.right, val * 10 + node.right.val))
        return res
```


1.   [Surrounded Regions](https://leetcode-cn.com/problems/surrounded-regions/)
```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not any(board):
            return

        m, n = len(board), len(board[0])
        save = [
            ij for k in range(m + n) for ij in ((0, k), (m - 1, k), (k, 0), (k, n - 1))
        ]
        # 遍历边界坐标
        while save:
            i, j = save.pop()
            if 0 <= i < m and 0 <= j < n and board[i][j] == "O":
                board[i][j] = "S"
                save += (i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)

        # 把'S'转化为'O',其他字符统一变换成'X'
        board[:] = [["XO"[c == "S"] for c in row] for row in board]
```

1.   [Palindrome Partitioning](https://leetcode-cn.com/problems/palindrome-partitioning/)
```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        self.dfs(s, [], res)
        return res

    def dfs(self, s, path, res):
        if not s:
            res.append(path)
            return
        for i in range(1, len(s) + 1):
            if self.isPal(s[:i]):
                self.dfs(s[i:], path + [s[:i]], res)

    def isPal(self, s):
        return s == s[::-1]
```
- 分治,将大问题分解为小问题。在遍历切割字符串的过程中,递归求的回文串。

1.   [Palindrome Partitioning II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-3-8/)
```python
class Solution:
    def minCut(self, s: str) -> int:
        cut = [x for x in range(-1, len(s))]
        for i in range(0, len(s)):
            for j in range(i, len(s)):
                if s[i:j] == s[j:i:-1]:
                    cut[j + 1] = min(cut[j + 1], cut[i] + 1)
        return cut[-1]
```


1.     [Gas Station](https://leetcode-cn.com/problems/gas-station/)
```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total = 0  # 判断整个数组是否有解
        sum = 0  # 判断当前指针的有效性
        j = -1
        for i in range(len(gas)):
            sum += gas[i] - cost[i]
            total += gas[i] - cost[i]
            if sum < 0:
                j = i
                sum = 0
        return j + 1 if total >= 0 else -1
```


1.     [Candy](https://leetcode-cn.com/problems/candy/)
```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        res = [1] * n

        # 从左到右
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                res[i] = res[i - 1] + 1
        # 从右到左
        for i in range(n - 1, 0, -1):
            if ratings[i - 1] > ratings[i]:
                res[i - 1] = max(res[i - 1], res[i] + 1)

        return sum(res)
```

1.     [Single Number](https://leetcode-cn.com/problems/single-number/submissions/)
```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        x = 0
        for k in nums:
            x ^= k
        return x
```

1.     [Single Number II](https://leetcode-cn.com/problems/single-number-ii/)
```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return (3 * sum(set(nums)) - sum(nums)) // 2
```



1.     [Copy List with Random Pointer](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)
```python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        d, node = {None: None}, head
        while node:
            d[node] = Node(node.val, None, None)
            node = node.next
        node = head
        while node:
            d[node].next = d[node.next]
            d[node].random = d[node.random]
            node = node.next
        return d[head]
```
- 难点在于random可能指向还未创建的节点
     1. 通过字典记录对应的节点,第二次遍历添加next和random指向 
     2. 或者通过在原链表上添加节点,最后拆分的方法完成题目要求
   

1.     [Linked List Cycle](https://leetcode-cn.com/problems/linked-list-cycle/)
```python
class Solution(object):
    def hasCycle(self, head):
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True
        return False
```
- 快慢指针
```python  
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        while head and head.val != None:
            head.val, head = None, head.next

        return head != None
```
1.     [Linked List Cycle II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)
```python
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        s = {None}
        while head not in s:
            s.add(head)
            head = head.next
        return head
```

```python
class Solution(object):
    def detectCycle(self, head):
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                break
            else:
                return None
        while head is not slow:
            head = head.next
            slow = slow.next
        return head
```
- 设环的起始节点为 E，快慢指针从 head 出发，快指针速度为 2，设相交节点为 X，head 到 E 的距离为 H，E 到 X 的距离为 D，环的长度为 L，那么有：快指针走过的距离等于慢指针走过的距离加快指针多走的距离（多走了 n 圈的 L） 2(H + D) = H + D + nL，因此可以推出 H = nL - D，这意味着如果我们让俩个慢指针一个从 head 出发，一个从 X 出发的话，他们一定会在节点 E 相遇

1.     [Reorder List](https://leetcode-cn.com/problems/reorder-list/)
```python
class Solution:
    def _splitList(self, head):
        fast = head
        slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next
            fast = fast.next

        middle = slow.next
        slow.next = None

        return head, middle

    def _reverseList(self, head):
        last = None
        currentNode = head

        while currentNode:
            nextNode = currentNode.next
            currentNode.next = last
            last = currentNode
            currentNode = nextNode

        return last

    def _mergeLists(self, a, b):

        tail = a
        head = a

        a = a.next
        while b:
            tail.next = b
            tail = tail.next
            b = b.next
            if a:
                a, b = b, a

        return head

    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head or not head.next:
            return

        a, b = self._splitList(head)
        b = self._reverseList(b)
        head = self._mergeLists(a, b)
```

133. [Clone Graph](https://leetcode-cn.com/problems/clone-graph/solution/)
```python
import copy
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        return copy.deepcopy(node)
```
140. [Word Break II](https://leetcode-cn.com/problems/word-break-ii/)
```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        memo = {len(s): ['']}

        def sentences(i):
            if i not in memo:
                memo[i] = [
                    s[i:j] + (tail and ' ' + tail)
                    for j in range(i + 1,
                                   len(s) + 1) if s[i:j] in wordDict
                    for tail in sentences(j)
                ]
            return memo[i]

        return sentences(0)
```

1.     [Binary Tree Preorder Traversal](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)
```python
def preorderTraversal(self, root):
    ret = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            ret.append(node.val)
            stack.append(node.right)
            stack.append(node.left)
    return ret
```
- 用栈模拟递归
```python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        return root and sum(
            ([root.val],
             *map(self.preorderTraversal, [root.left, root.right])), []) or []
```
- 使用map对左右孩子分别调用,sum对list进行相加操作


1.     [Binary Tree Postorder Traversal](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)
```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        r, stack = [], root and [root] or []
        while stack:
            root = stack.pop()
            r.append(root.val)
            stack += root.left and [root.left] or []
            stack += root.right and [root.right] or []
        return r[::-1]
```
1.       [LRU Cache](https://leetcode-cn.com/problems/lru-cache/)
```python
class LRUCache:
    def __init__(self, capacity: int):
        self.od, self.cap = collections.OrderedDict(), capacity

    def get(self, key: int) -> int:
        if key not in self.od:
            return -1
        self.od.move_to_end(key)
        return self.od[key]

    def put(self, key: int, value: int) -> None:
        if key in self.od:
            del self.od[key]
        elif len(self.od) == self.cap:
            self.od.popitem(False)  # 先进先出
        self.od[key] = value
```

1.   [Insertion Sort List](https://leetcode-cn.com/problems/insertion-sort-list/solution/jia-ge-tailsu-du-jiu-kuai-liao-by-powcai/)
```python
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        p = dummy = ListNode(0)
        cur = dummy.next = head
        while cur and cur.next:
            val = cur.next.val
            if cur.val < val:
                cur = cur.next
                continue
            if p.next.val > val:
                p = dummy
            while p.next.val < val:
                p = p.next
            new = cur.next
            cur.next = new.next
            new.next = p.next
            p.next = new
        return dummy.next
```

1.   [Sort List](https://leetcode-cn.com/problems/sort-list/)
```python
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        def merge(h1, h2):
            dummy = tail = ListNode(None)
            while h1 and h2:
                if h1.val < h2.val:
                    tail.next, tail, h1 = h1, h1, h1.next
                else:
                    tail.next, tail, h2 = h2, h2, h2.next
            tail.next = h1 or h2
            return dummy.next

        if not head or not head.next:
            return head
        pre, slow, fast = None, head, head
        while fast and fast.next:
            pre, slow, fast = slow, slow.next, fast.next.next
        pre.next = None

        return merge(*map(self.sortList, (head, slow)))
```


1.       [Evaluate Reverse Polish Notation](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)
```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for k, v in enumerate(tokens):
            if v in '+-*/':
                b, a = stack.pop(), stack.pop()
                v = eval('a' + v + 'b')
            stack.append(int(v))
        return stack.pop()
```
- 用栈模拟求解步骤
```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        t, f = tokens.pop(), self.evalRPN
        if t in '+-*/':
            b, a = f(tokens), f(tokens)
            t = eval('a' + t + 'b')
        return int(t)
```
- 递归地返回左右表达式操作后结果。eval 函数将字符串看作代码得到输出值


1.     [Two Sum II - Input array is sorted](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/)
```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        first, last = 0, len(numbers) - 1
        while numbers[first] + numbers[last] != target:
            if numbers[first] + numbers[last] > target:
                last -= 1
            else:
                first += 1
        return [first + 1, last + 1]

```

1.     [Array Partition I](https://leetcode-cn.com/problems/array-partition-i/)
```python
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        return sum(sorted(nums)[::2])
```