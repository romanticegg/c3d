
def decorators(func):
    def printout(Name):
        print 'Hello' + func(Name)
        # return 'Fuck!'
    return printout


def inner_func(input):
    return input


def main():
    x = decorators(inner_func)
    x('Hello')

if __name__ == '__main__':
    main()