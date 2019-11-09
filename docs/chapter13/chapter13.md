121. []()



122. [Best Time to Buy and Sell Stock](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit, min_price = 0, float('inf')
        for price in prices:
            max_profit, min_price = max(max_profit, price - min_price), min(
                min_price, price)
        return max_profit
```

139. [Word Break](https://leetcode-cn.com/problems/word-break/)
```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        ok = [True]
        for i in range(1, len(s) + 1):
            ok += any(ok[j] and s[j:i] in wordDict for j in range(i)),
        return ok[-1]
```