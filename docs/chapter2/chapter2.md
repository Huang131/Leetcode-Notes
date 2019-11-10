11 [Container With Most Water](https://leetcode-cn.com/problems/container-with-most-water/)

>给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
说明：你不能倾斜容器，且 n 的值至少为 2。
<br>
示例:
```
输入: [1,8,6,2,5,4,8,3,7]
输出: 49
```

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        res, left, right = 0, 0, len(height) - 1
        while left < right:
            res, left, right = (max(res, height[left] * (right - left)),left + 1,right)
                                 if height[left] < height[right] else (
                                    max(res, height[right] * (right - left)),
                                    left, right - 1)
        return res
```
- 双指针，从两端向中间遍历，用res存贮最大面积，较短指针移向较长指针


12 [Integer to Roman](https://leetcode-cn.com/problems/integer-to-roman/)
>罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。

```
字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```
例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

```
    I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
    X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
    C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
```
给定一个整数，将其转为罗马数字。输入确保在 1 到 3999 的范围内。
<br>
示例:
```
输入: 3
输出: "III"

输入: 4
输出: "IV"

输入: 58
输出: "LVIII"
解释: L = 50, V = 5, III = 3.
```

```python

class Solution:
    def intToRoman(self, num: int) -> str:
        M = ["", "M", "MM", "MMM"] #[0,1000,2000,3000]
        C = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"] # [0,100,200,300,400,500,600,700,800,900]
        X = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"] # [0,10,20,30,40,50,60,70,80,90]
        I = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"] # [0,1,2,3,4,5,6,7,8,9]

        return M[num // 1000] + C[(num % 1000) // 100] + X[(num % 100) //
                                                           10] + I[num % 10]
```

13 [Roman to Integer](https://leetcode-cn.com/problems/roman-to-integer/?utm_source=LCUS&utm_medium=ip_redirect_q_uns&utm_campaign=transfer2china)

>给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。
<br>
示例:

```
输入: "LVIII"
输出: 58
解释: L = 50, V= 5, III = 3.

输入: "MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        d = {
            'I': 1,
            'IV': 3,
            'V': 5,
            'IX': 8,
            'X': 10,
            'XL': 30,
            'L': 50,
            'XC': 80,
            'C': 100,
            'CD': 300,
            'D': 500,
            'CM': 800,
            'M': 1000
        }
        r = d[s[0]]
        for i in range(1, len(s)):
            r += d.get(s[i - 1:i + 1], d[s[i]])
        return r
```
- 构建一个字典记录所有罗马数字子串,长度为2的子串记录的值(实际值-子串左边的罗马数字的值)
- 遍历s,判断当前位置和前一个位置是否在字典内,如果在就记录值,不在就直接记录当前位置字符对应值
- 例如CD为400,先遍历到C记录为100,在遍历到CD,记录为300。相加,正好为正确值400

14 [Longest Common Prefix](https://leetcode-cn.com/problems/longest-common-prefix/)

>编写一个函数来查找字符串数组中的最长公共前缀。所有输入只包含小写字母 a-z 。
如果不存在公共前缀，返回空字符串 ""。
<br>
示例:

```
输入: ["flower","flow","flight"]
输出: "fl"

输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。
```

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        r = [len(set(c)) == 1 for c in zip(*strs)] + [0]
        return strs[0][:r.index(0)] if strs else ''
```

- 用set()函数去重判断是否为公共前缀,0作为标志位。True 和 False 被解释为 1 和 0

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        import os
        return os.path.commonprefix(strs)
```
- os中存在库函数求公共前缀
```python
def commonprefix(m):
    if not m: return ''
    if not isinstance(m[0], (list, tuple)):
        m = tuple(map(os.fspath, m))
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1     
```
- commonprefix()函数通过max(),min()计算出ascii码最大,最小的字符串进行比较。如果s1和s2有共同前缀,其他字符串都有。如果s1和s2没有,其它有也没用

15 [3Sum](https://leetcode-cn.com/problems/3sum)

>给定一个包含 n 个整数的数组`nums`，判断`nums`中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。
注意：答案中不可以包含重复的三元组。
<br>
示例:

```
例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums, r = sorted(nums), set()
        # 对前n-2个非重复数的下标进行遍历
        for i in [ i for i in range(len(nums) - 2) if i < 1 or nums[i] > nums[i - 1] ]:
            # 字典d保存第三个数大小和索引
            d = {-(nums[i] + n): j for j, n in enumerate(nums[i + 1:])}
            r.update([(nums[i], n, -nums[i] - n) for j, n in enumerate(nums[i + 1:])
                                                            if n in d and d[n] > j])
        return list(map(list, r))
```
- 时间复杂度：O(N^2)
- sort避免重复,节省计算。使得输出结果都是升序,利用set排除一些排除一些相同结果。
- 用字典记录{需要的值:当前索引},字典会记录比较大的那个索引,用d[n]>j来避免重复选择一个元素
- (nums[i], n, -nums[i] - n)保证列表升序


16 [3Sum Closest](https://leetcode-cn.com/problems/3sum-closest/)
>给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。
<br>
示例:

```
例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.
与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
```

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        result = nums[0] + nums[1] + nums[2]
        # 遍历前n-2个数
        for i in range(len(nums) - 2):
            # 双指针
            j, k = i + 1, len(nums) - 1
            while j < k:
                cur_sum = nums[i] + nums[j] + nums[k]
                if cur_sum == target:
                    return cur_sum
                # cur_sum 更接近目标
                if abs(cur_sum - target) < abs(result - target):
                    result = cur_sum

                if cur_sum < target:
                    j += 1
                elif cur_sum > target:
                    k -= 1

        return result
```
- sort排序后,双指针遍历

17 [Letter Combinations of a Phone Number](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

>给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
<br>
示例:

```
输入："23"
输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        from itertools import product
        l = '- - abc def ghi jkl mno pqrs tuv wxyz'.split()
        return [''.join(c) for c in product(*[l[int(i)] for i in digits])] if digits else []
```
- product(A,B)函数,返回A和B中的元素组成的笛卡尔积的元组


18 [4Sum](https://leetcode-cn.com/problems/4sum)

>给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。
注意：
答案中不可以包含重复的四元组。
<br>
示例:

```
给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。

满足要求的四元组集合为：
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]
```

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        import collections
        from itertools import combinations as com
        # 将两个索引组合存入l中
        dic, l = collections.defaultdict(list), [*com(range(len(nums)), 2)]
        # 将剩下的两数之和作为索引存入dic
        for a, b in l:
            dic[target - nums[a] - nums[b]].append((a, b))
        # 如果nums[c]+nums[d]存在,从字典中取出对应a,b索引
        r = [(*ab, c, d) for c, d in l for ab in dic[nums[c] + nums[d]]]
        return [
            *set(tuple(sorted(nums[i] for i in t)) for t in r if len(set(t)) == 4)
        ]
```
- 与2Sum相似,先将总和与任意两数之和的差存入字典,再获得其余任意两个数字,寻找匹配值
- combinations(iterable, r) 创建一个迭代器,返回iterable中所有长度为r的子序列。combinations(range(4), 3) --> (0,1,2), (0,1,3), (0,2,3), (1,2,3)

19 [Remove Nth Node From End of List](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list)

>给定一个链表，删除链表的倒数第`n`个节点，并且返回链表的头结点。
<br>
示例:
```
给定一个链表: 1->2->3->4->5, 和 n = 2.
当删除了倒数第二个节点后，链表变为 1->2->3->5.
```

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        p = q = head
        for i in range(n):
            q = q.next
        if not q:
            return head.next
        while q.next:
            q = q.next
            p = p.next
        p.next = p.next.next
        return head
```
- 双指针,q先走n步,然后p和q一起走,直到q走到尾节点

20 [Valid Parentheses](https://leetcode-cn.com/problems/valid-parentheses/)

>给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
有效字符串需满足：
```
    左括号必须用相同类型的右括号闭合。
    左括号必须以正确的顺序闭合。
```
注意空字符串可被认为是有效字符串。
<br>
示例:

```
输入: "()[]{}"
输出: true

输入: "([)]"
输出: false
```

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        d = {'(': ')', '[': ']', '{': '}'}
        for k in s:
            if k in '{([':
                stack.append(k)
            elif not stack or d[stack.pop()] != k:
                    return False
        return not stack
```
- 不断删除有效括号直到不能删除