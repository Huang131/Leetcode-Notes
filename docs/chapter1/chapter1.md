1. [Two Sum](https://leetcode-cn.com/problems/two-sum)

>给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那两个整数，并返回他们的数组下标。
你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。
<br>
示例:
```
给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
```

python代码
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, n in enumerate(nums):
            if n in d: return [d[n], i]
            d[target - n] = i
```
- 用字典记录 { 需要的值:当前索引 } 时间复杂度:O(n)


1. [Add Two Number](https://leetcode-cn.com/problems/add-two-numbers)
```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode, carry=0) -> ListNode:
        if not (l1 or l2):
            return ListNode(1) if carry else None
        l1, l2 = l1 or ListNode(0), l2 or ListNode(0)
        val = l1.val + l2.val + carry
        l1.val, l1.next = val % 10, self.addTwoNumbers(l1.next,l2.next,al > 9)
        return l1
```
- 用carry记录进位



3. [Longest Substring Without Repeating Characters](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters)
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start = maxLength = 0
        useChar = {}  # 保存非重复字符串
        for i, ch in enumerate(s):
            if ch in useChar and start <= useChar[ch]:
                start = useChar[ch] + 1  # start 移动到字典重复字符所在的后一位
            else:
                maxLength = max(maxLength, i - start + 1)
            useChar[ch] = i  # 当前遍历字符所在索引存入字典

        return maxLength
```
- 滑动窗口问题

4. [MedianofTwoSortedArrays](https://leetcode-cn.com/problems/median-of-two-sorted-arrays)
```python
class Solution:
    def findKth(self, nums1: List[int], nums2: List[int], k: int):

        # 边界条件
        if len(nums1) > len(nums2):
            return self.findKth(nums2, nums1, k)
        elif len(nums1) == 0:
            return nums2[k - 1]
        elif k == 1:
            return min(nums1[0], nums2[0])

        first = min(k // 2, len(nums1))
        second = k - first

        if nums1[first - 1] < nums2[second - 1]:
            return self.findKth(nums1[first:], nums2, k - first)
        elif nums1[first - 1] > nums2[second - 1]:
            return self.findKth(nums1, nums2[second:], k - second)
        else:
            return nums1[first - 1]

    def findMedianSortedArrays(self, nums1: List[int],nums2: List[int]) -> float:

        length = len(nums1) + len(nums2)

        if (length & 0x1):
            return self.findKth(nums1, nums2, length // 2 + 1)
        else:
            return ((self.findKth(nums1, nums2, length // 2) +
                     self.findKth(nums1, nums2, length // 2 + 1)) / 2)
```
- 从K入手,利用有序这个条件,每次排除一半的元素。假设A和B的元素个数都大于K/2,因为K的奇偶性不影响结论,假设K为偶数。将A[k/2-1]与B[k/2-1]进行比较：
    * A[k/2-1] == B[k/2-1]  直接返回A[k/2-1]或者B[k/2-1]
    * A[k/2-1]  < B[k/2-1]  删除A中的k/2个元素
    * A[k/2-1]  > B[k/2-1]  删除B中的k/2个元素
- 函数终止条件：
    * 当A或B是空时:直接返回B[k-1]或A[k-1]
    * 当k=1时,返回min(A[0],B[0])
    * 当A[k/2-1] == B[k/2-1]时,返回A[k/2-1]或B[k/2-1]

5. [Longest Palindromic Substring](https://leetcode-cn.com/problems/longest-palindromic-substring/)
```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        r = ''
        for i, j in [(i, j) for i in range(len(s)) for j in (0, 1)]:
            while i > -1 and i + j < len(s) and s[i] == s[i + j]:
                i, j = i - 1, j + 2
            r = max(r, s[i + 1:i + j], key=len)
        return '' if not s else r
```
- 字符串的中心可能是一个字符也可能是两个字符,例如字符串abcbd,回文子串bcb,中心字符为c。字符串abccbd,回文子串bccb,中心字符为cc。所以j的取值为0或1。
- i 遍历字符串中的每一个字符,通过j的取值判断回文子串的中心字符取值情况。j为0时,子串假设为一个中心字符。j为1时，子串假设为两个中心字符。
- r保存每次确定中心字符情况后的最大子串


6. [ZigZag Conversion](https://leetcode-cn.com/problems/zigzag-conversion/)
```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows >= len(s):
            return s
        L = [''] * numRows
        index, step = 0, 1
        for x in s:
            L[index] += x
            if index == 0:
                step = 1  # 正向
            elif index == numRows - 1:
                step = -1  # 反向
            index += step
        return ''.join(L)
```
- step控制前进方向,最后把numRowsa行字符串合并

7. [Reverse Integer](https://leetcode-cn.com/problems/reverse-integer/solution)
```python
class Solution:
    def reverse(self, x: int) -> int:
        # x // max(1, abs(x)) 相当于cmp函数
        r = x // max(1, abs(x)) * int(str(abs(x))[::-1])
        return r if r.bit_length() < 32 or r == -2**31 else 0
```

- x // max(1, abs(x))意味着 0：x为0， 1：x为正， -1：x为负，相当于被废弃的函数cmp
- [::-1]代表序列反转
- 2^31 和 -2^31 的比特数为32，其中正负号占用了一位
- 32位整数范围 [−2^31, 2^31 − 1] 中正数范围小一个是因为0的存在


8. [String to Integer(atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi)
```python
class Solution:
    def myAtoi(self, str: str) -> int:
        return max(
            min(int(*re.findall('^[\+\-]?\d+', str.lstrip())), 2**31 - 1),
            -2**31)
```
- 使用正则表达式 ^：匹配字符串开头，[\+\-]：代表一个+字符或-字符，?：前面一个字符可有可无，\d：一个数字，+：前面一个字符的一个或多个，\D：一个非数字字符，*：前面一个字符的0个或多个
- max(min(数字, 2 ** 31 - 1), -2 ** 31) 用来防止结果越界

9. [Palindrome Number](https://leetcode-cn.com/problems/palindrome-number/)
```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        return str(x) == str(x)[::-1]
```

10. [Regular Expression Matching](https://leetcode-cn.com/problems/regular-expression-matching/)
```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if s == p:
            return True
        if len(p) > 1 and p[1] == '*':  # 下一个字符为*
            if s and (s[0] == p[0] or p[0] == '.'):
                return self.isMatch(s, p[2:]) or self.isMatch(s[1:], p)
            else:
                return self.isMatch(s, p[2:])
        elif s and p and (s[0] == p[0] or p[0] == '.'):
            return self.isMatch(s[1:], p[1:])
        return False
```
- '.' 匹配任意单个字符, '*' 匹配零个或多个前面的那一个元素
- 当模式中第二个字符是'*'时：
    -  如果字符串第一个字符跟模式第一个字符不匹配，则模式后移2个字符，继续匹配。如果字符串第一个字符跟模式第一个字符匹配，可以有3种匹配方式：
        1. 模式后移2位字符,即模式前两位被忽略（*匹配0个字符）
        2. 字符串后移1字符,模式不变,*可以匹配多位
        3. 字符后移1字符,模式后移2字符
    发现情况3可以被情况1和情况2包含,即执行一次情况2,再执行一次情况1。所以情况3不用判断
- 当模式中第二个字符不是'*'时:
    - 如果字符串第一个字符和模式中的第一个字符相匹配,字符串和模式都后移一个字符。
    - 如果字符串第一个字符和模式中的第一个字符不匹配,直接返回False
