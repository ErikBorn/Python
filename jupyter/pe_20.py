# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:16:50 2014

@author: fa11440

Project Euler Problem 20:

n! means n × (n − 1) × ... × 3 × 2 × 1

For example, 10! = 10 × 9 × ... × 3 × 2 × 1 = 3628800,
and the sum of the digits in the number 10! is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27.

Find the sum of the digits in the number 100!
"""

def fact(number = 1):
    product = 1    
    for i in range(1,number+1):
        product = product*i
    return product
    
def sumfinder(value =1):
    S = 0
    for i in range(len(str(value))):
        S = S + int(str(value)[i])
    return S
    
print(sumfinder(fact(100)))