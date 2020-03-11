# import pkgutil
# search_path = '.' # set to None to see all modules importable from sys.path
# all_modules = [x[1] for x in pkgutil.iter_modules(path=search_path)]
# print(all_modules)

class test:
    def __init__(self):
        self.a = 0

def plusplus(atest):
    atest.a += 1

if __name__ =='__main__':
    atest = test()
    plusplus(atest)
    plusplus(atest)
    print(atest.a)
    print("qwe")