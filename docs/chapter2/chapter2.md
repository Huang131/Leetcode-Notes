1.  [Container With Most Water](https://leetcode-cn.com/problems/container-with-most-water/solution/sheng-zui-duo-shui-de-rong-qi-by-leetcode/)
```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        res, left, right = 0, 0, len(height) - 1
        while left < right:
            res, left, right = (max(res, height[left] * (right - left)),
                                left + 1,
                                right) if height[left] < height[right] else (
                                    max(res, height[right] * (right - left)),
                                    left, right - 1)
        return res
```
- 双指针，从两端向中间遍历，用res存贮最大面积，较短指针移向较长指针


12. [Integer to Roman](https://leetcode-cn.com/problems/integer-to-roman/)
```python
# 字符          数值
# I             1
# V             5
# X             10
# L             50
# C             100
# D             500
# M             1000

class Solution:
    def intToRoman(self, num: int) -> str:
        M = ["", "M", "MM", "MMM"] #[0,1000,2000,3000]
        C = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"] # [0,100,200,300,400,500,600,700,800,900]
        X = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"] # [0,10,20,30,40,50,60,70,80,90]
        I = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"] # [0,1,2,3,4,5,6,7,8,9]

        return M[num // 1000] + C[(num % 1000) // 100] + X[(num % 100) //
                                                           10] + I[num % 10]
```

13. [Roman to Integer](https://leetcode-cn.com/problems/roman-to-integer/?utm_source=LCUS&utm_medium=ip_redirect_q_uns&utm_campaign=transfer2china)
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

14. [Longest Common Prefix](https://leetcode-cn.com/problems/longest-common-prefix/)
```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        r = [len(set(c)) == 1 for c in zip(*strs)] + [0]
        return strs[0][:r.index(0)] if strs else ''
```
- 用set()函数去重判断是否为公共前缀,0作为标志位
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

15.  [3Sum](https://leetcode-cn.com/problems/3sum)
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums, r = sorted(nums), set()
        # 对前n-2个非重复数的下标进行遍历
        for i in [
                i for i in range(len(nums) - 2)
                if i < 1 or nums[i] > nums[i - 1]
        ]:
            # 字典d保存第三个数大小和索引
            d = {-(nums[i] + n): j for j, n in enumerate(nums[i + 1:])}
            r.update([(nums[i], n, -nums[i] - n) for j, n in enumerate(nums[i + 1:])
                                                            if n in d and d[n] > j])
        return list(map(list, r))
```
- sort避免重复,使得输出结果都是升序,用字典记录{需要的值:当前索引},字典会记录比较大的那个索引,用d[n]>j来避免重复选择一个元素,(nums[i], n, -nums[i] - n)保证列表升序


17. [Letter Combinations of a Phone Number](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)
```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        from itertools import product
        l = '- - abc def ghi jkl mno pqrs tuv wxyz'.split()
        return [''.join(c)
                for c in product(*[l[int(i)]
                                   for i in digits])] if digits else []
```



18. [4Sum](https://leetcode-cn.com/problems/4sum)
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
            *set(
                tuple(sorted(nums[i] for i in t))
                for t in r if len(set(t)) == 4)
        ]
```
- 与2Sum相似,先将总和与任意两数之和的差存入字典,再获得其余任意两个数字,寻找匹配值
- combinations(iterable, r) 创建一个迭代器,返回iterable中所有长度为r的子序列。combinations(range(4), 3) --> (0,1,2), (0,1,3), (0,2,3), (1,2,3)

19. [Remove Nth Node From End of List](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list)
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

20. [Valid Parentheses](https://leetcode-cn.com/problems/valid-parentheses/)
```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        d = {'(': ')', '[': ']', '{': '}'}
        for k in s:
            if k in '{([':
                stack.append(k)
            else:
                if not stack or d[stack.pop()] != k:
                    return False
        return not stack
```