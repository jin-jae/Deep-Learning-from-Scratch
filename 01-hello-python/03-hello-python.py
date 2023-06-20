# algebra
print(1 + 2)
print(1 - 2)
print(4 * 5)
print(7 / 5)
print(3 ** 2)

# type
print(type(10))
print(type(2.718))
print(type("hello"))

# variables
x = 10
print(x)
x = 100
print(x)
y = 3.14
print(x * y) # int * float => float

# list
a = [1, 2, 3, 4, 5]
print(a)
len(a)
a[0]
a[4] = 99
print(a) # [1, 2, 3, 4, 99]

# slicing
a[1:]
a[:3]
a[:-1]
a[:-2]

# Dictionary
me = {"height" : 180}
me["height"] # 180
me["weight"] = 70
me["weight"] # 70

# boolean
isHungry = True
isSleepy = False
not isHungry # False
isHungry and isSleepy # False

# if
if isHungry:
    print("I am hungry.")
else:
    print("I am not hungry.")
    print("I am sleepy.")

# for
for i in range(1, 4):
    print(i)

# function
def hello():
    print("Hello, World!")

hello()

def hello(obj):
    print("Hello " + obj + "!")

hello("cat")

