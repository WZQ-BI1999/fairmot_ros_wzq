#!/usr/bin/env python

count = 0
class num(object):
    def __init__(self, i):
        self.n = i
    def add(self, i):
        self.n = self.n + i
    def show(self):
        print(self.n)

def main():
    global count
    global a
    if count==0:
        a = num(3)
    else:
        a.add(1)
        count
    count = count+1
    a.show()

if __name__=='__main__':
    while count<5:
        main()