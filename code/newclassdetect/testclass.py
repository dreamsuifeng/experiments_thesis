# -*- coding:utf-8 -*-

class parent:
    count=0
    def __init__(self,name,desc):
        parent.count+=1
        self.name=name
        self.desc=desc
    
    def get_name(self):
        print('parent name is '+self.name)
        return self.name

    
class child(parent):
    def _init__(self):
        pass
    def get_name(self):
        print('name is '+self.name)


if __name__=='__main__':
    childone=child('chidname1','haha')
    childone.get_name()
    father=parent('fatherone','father')
    father.get_name()
    print(parent.count)
