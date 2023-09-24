import numpy as np


def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i =0
        if len(nums1)> 0 and  len(nums2)>0:    
            while i < m:
                
                if nums1[i]> nums2[0]:
                    nums1 = nums1[:i]+[nums2[0]]+nums1[i:m+n-1]
                    nums2 = nums2[1:]      
                i+=1 
            
            nums1 = nums1[:m+n - len(nums2)]+nums2   
        elif len(nums1)==0: 
            nums1 = nums2            
        return nums1     

def merge_1(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    
    i = 0 
    
    if len(nums1)> 0 and  len(nums2)>0: 
        while i < m+n:
            
            if nums1[i] > nums2[0]:
                
                nums1.insert(i,nums2[0])
                nums2.pop(0)
                nums1.pop(m+n-1)
            
            i+=1
        
        j = len(nums2)
        
        for k in range(j):
            nums1.pop(m-n-1)
            
        nums1.extend(nums2)
        
    elif len(nums1)==0: 
            nums1.extend(nums2)   
         
   
def removeDuplicates(nums: list[int]) -> int:

    unique = []
    i = 0
    while i < len(nums):

        if nums[i] not in unique:
            unique.append(nums[i])
        else :
            nums.pop(i)
            i-=1
        i+=1




def removeDuplicates2(nums: list()):
    
 
    counts = {key :0 for key in set(nums)}
    
    i =0
    while i < len(nums):
        
        counts[nums[i]]+=1
        if counts[nums[i]]>2:
            nums.pop(i)
            i-=1
        i+=1
        


def majorityElement(nums: list[int]) -> int:

    threshold = int(len(nums)/2)

    counts = [i for i in set(nums) if nums.count(i) > threshold]

    return counts[0]



def rotate(nums: list[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        l = nums.copy()
        n = len(nums)
        for i in range(n):
            
            nums[(i+k)%n] = l[i]



# Uses Two pointers PTR

def maxProfit(prices: list[int]) -> int:  #Time Limit exceeded

    l, r = 0,1
    profit = 0
    
    while r < len(prices):
        
        if prices[r]< prices[l]:
            l = r
        
        if profit < prices[r] - prices[l]:
            
            profit = prices[r] - prices[l]
        
        r+=1

    return profit


def maxProfit2(prices:list[int]):

    l,r = 0,1

    profits =0
    profit = 0

    while r < len(prices):

        if prices[r]< prices[l]:
            # profits+= profit
            # profit = 0
            l = r
        if profit < prices[r]-prices[l]:
            profit = prices[r]-prices[l]
            profits+= profit
            profit = 0
            l = r
        
        r+=1
    
    return profits
                

def canJump( nums: list[int])->bool:
    
    goal = len(nums) -1
    n = len(nums)-1
    canJ = False
    
    for i in range(1,n+1):
        
        if nums[n-i] >= goal - n+i:
            goal = n-i
    
    
    if goal ==0:
        canJ = True     
        
        
        
    return canJ



def jump(nums: list[int]) -> int:
    

    jumpcount =0
    r = len(nums)-1

    while r>0:
        
        for j in range(1,r+1):

            if nums[r-j] >= j:
                best = r-j
        
        r = best
        jumpcount +=1
                
        
    return jumpcount



def hindex(citations:list[int]) ->int:
    

    h = 0
    l = len(citations)
        
    for i in range(1,l+1):
        
        satisfied = [j for j in citations if j >= i]
        
        if len(satisfied)> h:
            h = i
            
    return h

          
class RandomizedSet:

    def __init__(self):
        
        self.set = []

    def insert(self, val: int) -> bool:
        
        if val == []:
            return
        if val in self.set:
            return False
        else:
            self.set.extend(val)
            return True
        

    def remove(self, val: int) -> bool:
        
        if val == []:
            return
        if val[0] in self.set:
            self.set.pop(self.set.index(val[0]))
            return True
        else:
            return False
        

    def getRandom(self) -> int:
        

        return random.choice(self.set)


ran = RandomizedSet()

import random

class RandomizedSet_1:
    def __init__(self):
        
        self.numMap = {}
        self.list = []
        
    def insert(self, val: int) -> bool:
        
        if val == None:
            return
        if val in self.numMap:
            return False
        else:
            self.numMap[val] = len(self.list)
            self.list.append(val)
            return True
        

    def remove(self, val: int) -> bool:
        
        if val == None:
            return
        if val in self.numMap:
            idx = self.numMap[val]
            lastVal = self.list[-1]
            self.list[idx] = lastVal
            self.list.pop()
            del self.numMap[val]
            
            return True
        else:
            return False
        

    def getRandom(self) -> int:
        
        return random.choice(self.list)

import math
def productExceptSelf(nums):
    
    prod = math.prod(nums)

    answer = [int(prod*(nums[i]**-1)) if nums[i]!=0 else math.prod(nums[:i])*math.prod(nums[i+1:]) for i in range(len(nums))]
        
    
    return answer

def canCompleteCircuit(gas,cost): # Another Greedy problem
    
    if sum(gas)<sum(cost):
        return -1
    
    
    total = 0
    res = 0
    
    for i in range(len(gas)):
        
        total += (gas[i]- cost[i])
        
        if total < 0:
            total = 0
            res = i +1             
    
    return res



def candy(ratings):
    
    nKids = len(ratings)
    
    candies = [1 for _ in range(nKids)]
    
    # First Take care of the extremities
    if nKids == 1:
        return 1
    if nKids == 2:
        if ratings[0]> ratings[1]:
            candies[0] = 2
        if ratings[1]> ratings[0]:
            candies[1] = 2
    else:    
        
        left = candies.copy()
        right = candies.copy()
        
        for i in range(1,nKids-1):

            if ratings[i]>ratings[i-1]:
                left[i] = left[i-1]+1
            if ratings[nKids-i-1]> ratings[nKids-i]:
                right[nKids-i-1] = right[nKids-i]+1

        candies = [max(left[i],right[i]) for i in range(nKids)]
        
        if ratings[0]> ratings[1]:
            candies[0] = candies[1]+1
        if ratings[-1]> ratings[-2]:
            candies[-1] = candies[-2]+1 
            
    return (candies,sum(candies))       



def trap(height):
    
    x = len(height)
    
    if x<=2:
        return 0
    else:
        
        elevation = [0 for _ in range(x)]
        
        left = elevation.copy()
        right = elevation.copy()
        
        maxl = height[0]
        maxr = height[x-1]
        for i in range(x):
            
            if height[i]>maxl:
                maxl = height[i]
            left[i] = maxl
            
            if height[x-i-1]>maxr:
                maxr = height[x-i-1]
            right[x-i-1] = maxr    

        trapped = [min(left[i],right[i]) - height[i] for i in range(x)]
        
        pass
    return sum(trapped)
              
        
        
def romanToInt(s: str) -> int:
    
    x = len(s)

    if x ==0:
        return 0
    else :

        admitted =['I','V','X','L','C','D','M']

        values ={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}

        tot =0
        i =0

        while i < x-1:
            
            if admitted.index(s[i])<admitted.index(s[i+1]):
                tot += (values[s[i+1]]-values[s[i]])
                i+=2
            else:
                tot += values[s[i]]
                i +=1
        if i == x-1:
            tot += values[s[x-1]]

    return tot


def intToRoman(num: int) -> str:
    

    div =[1000,900,500,400,100,90,50,40,10,9,5,4,1]

    values ={1:'I',4:'IV',5:'V',9:'IX',10:'X',40:'XL',50:'L',90:'XC',100:'C',400:'CD',500:'D',900:'CM',1000:'M'}

    roman = ''
    
    
    for j in range(len(div)):
        
        if num//div[j]>0:
            
            count = num//div[j]
                
            roman += (values[div[j]]*count)
            
            num = num % div[j]
        
            
    return roman
                
                
        
def lengthofLastWord(s):
    
    
    words = s.split(' ')
    n = len(words)
    l= 0
    for i in range(1,n+1):
        if words[n-i] != '':
            
            return len(words[n-i])

    return l

def longestCommonPrefix(strs):
    
    n = len(strs)
    if n == 0 or ("" in strs):
        return ""
    if n ==1:
        return strs[0]
    
    else:
        
        common = ""
        flag = True
        
        tmp = strs[0][0]
        
        mini = min([len(str) for str in strs])
        i=1
        
        while i <= mini:
        
            for str in strs:
                
                if tmp != str[:i]:
                    
                    flag = False
            
            if not flag:
                
                return common   
            
            common = (tmp+'.')[:-1]
            i +=1
            tmp = strs[0][:i]
        
        return common

import re 
   
def isPalindrome(s):
    
    s = re.sub(r"[\W_]",'',s)
    s = s.lower()

    n = len(s)
    if n == 0 or n ==1:
        return True

    flag = True

    i = 0

    while flag and i < n:

        if s[i] != s[n-i-1]:
            flag = False

        i+=1 

    return flag           
            
def isSubsequence(s,t):
    
    n,m = len(s),len(t)
    if n ==0:
        return True
    if n>m:
        return False

    for i in t:
        if i not in s:
            t = t.replace(i,'')
    
    n,m = len(s),len(t)
    

    
    if s in t:
        return True
    elif m>n:
        
        while m>n:
            
            first = [i for i in range(n) if s[i] != t[i]][0]
            
            t = t[:first]+t[first+1:]
            
            if s in t:
                return True
            m -=1

    else:
        return False
    
def twoSum(numbers,target):
    
    n = len(numbers)

    if n ==2:
        return [1,2]
    i =0
    while i < n:

        num = target - numbers[i]

        if num in numbers:
            j = numbers.index(num)
            if i != j:

                return [i+1,j+1]

        i+= 1
                    
def twoSum(numbers, target):
        
    n = len(numbers)

    if n ==2:
        return [1,2]
    i =0
    while i < n:

        num = target - numbers[i]

        if num in numbers:
            j = [k for k in range(n) if numbers[k] == num and k != i ][0]

            if i != j:

                return [i+1,j+1]

        i+= 1

def maxArea(height):
          
    n = len(height)
    l =0
    r = n-1

    if n == 2:

        return min(height[0],height[1])
    else:
        mw = 0

        while l != r:
            
            area = min(height[l],height[r])*(r-l)
            if area> mw:
                mw = area
            if height[l]>= height[r]:
                r-=1
            else:
                l +=1
            
            
        
        return mw

def threeSum(nums):
    
    n = len(nums)

    nums.sort()

    triplets = []

    for i in range(n-2):

        l = i+1
        r = n-1
        
        while l!=r:
            
            if nums[l]+nums[r] > -nums[i]:
                r -=1
                
            if nums[l]+nums[r] < -nums[i]:
                l += 1
            
            if nums[i]+nums[l]+nums[r] == 0:
                triplets.append([nums[i],nums[l],nums[r]])
                l +=1
        
    return triplets
            

def minSubArrayLen(target: int, nums: list[int]) -> int:
    
    l = 0
    total = 0
    res = float("inf")
    
    for r in range(len(nums)):
        
        total += nums[r]
        
        while total >= target:
            res = min(res, r-l+1)
            total-= nums[l]
            l+=1    
    
    return 0 if res == float("inf") else  res

def longestsubstr(s):
    
      
    l = 0
    res = 0

    for r in range(len(s)):

        if s[r] in s[l:r]:
            
            idx = s[l:r].index(s[r])
            l += idx +1
            
        res = max(res,r-l+1)
    
    return res


def findSubstring(s,words):
    
    # Resultant list
    indices = []
    # Base conditions
    if s is None or len(s) == 0 or words is None or len(words) == 0:
        return indices

    # Dictionary to store the count of each word in the words array
    wordCount = dict()
    # Loop to store count of each word in the array
    for i in range(len(words)):
        if words[i] in wordCount:
            wordCount[words[i]] += 1
        else:
            wordCount[words[i]] = 1
    # Length of each word in the words array
    wordLength = len(words[0])
    # Total length of all the words in the array
    wordArrayLength = wordLength * len(words)
    # Loop for each character in the string
    for i in range(0, len(s) - wordArrayLength + 1):
        # Get the current string
        current = s[i:i + wordArrayLength]
        # Map to store the count of each word in the current
        wordMap = dict()
        # Index to loop through the array
        index = 0
        # Index to partition the current string
        j = 0
        # Loop through the words array
        while index < len(words):
            # Partition the string
            part = current[j: j + wordLength]
            # Save this in wordMap
            if part in wordMap:
                wordMap[part] += 1
            else:
                wordMap[part] = 1
            # Update the indices
            j += wordLength
            index += 1
        # Compare the two maps
        if wordMap == wordCount:
            indices.append(i)
    return indices

def minWindow(s,t):
    
        m = len(s); n = len(t)
        l=0;r=1
        if s is None or m == 0 or m < n or t is None or n == 0:
            return ""
        

        freq_map = {}
        for i in t:
            if i in freq_map:
                freq_map[i]+=1
            else:
                freq_map[i] = 1
        result = ['.']*(m+1)
        tot = sum(list(freq_map.values()))
        
        while l <= m-n+1:
            
            char_map = freq_map.copy()
            f =0
            r=l+1
            for i in range(l,m):
                
                if s[i] in char_map and char_map[s[i]] !=0:                    
                    char_map[s[i]] -=1
                    f +=1                    
                    if f == 2:
                        r = i
                
                if sum(list(char_map.values())) == 0:
                    
                    if len(s[l:i+1]) < len(result):
                        result = s[l:i+1]
                    
                    l = r-1   
                    break
            l+=1
                
        
        return "" if len(result) == m+1 else result
                

#s = "ADOBECODEBANC"; t = "ABC"
# s = "abc"; t = "bc"

# print(minWindow(s,t))

import numpy as np

    
b = np.array([3,0,0,0])
A = np.array([[8,4,2],[12,4,1],[512,64,8],[192,16,1]])

print(np.linalg.lstsq(A,b))
    
    
    