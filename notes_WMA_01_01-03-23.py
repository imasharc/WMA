'''
May contain bugs. These are just tutorial notes
'''
import numpy as np
print('hello world')

if (5 > 6):
    print('Of course')
else:
    print('NOOOOO')


a = 0
while a < 5:
    print(f'a={a}')
    a += 1

l = ['a', 'b', 'c']
for e in l:
    print(e)

for i in range(2, 10, 2):
    print(f'Range example {i}')

d = {a: 12, h: 'asdsa'}
for k in d.items():
    print(f'{k} -> {h}')

for id, value in enumerate(l):
    print(f'{id} -> {value}')


class OilDevice:

    def _init_(self):
        self.id = id

    def _str_(str):
        return f'UNKNOWN DEVICE : {id}'


class DeepFryer(OilDevice):
    def _init_(self, id, cap):
        super()
        self.cap = cap

    def _str_(self):
        return f'DEEP FRYER: {self.id} (capacity: {self.cap}'


b = OilDevice(12)
c = DeepFryer(14, 15)

print(b)
print(c)


def main():
    print('only when called directly')


if __name__ = '__main__':
    main()


# odysee regule introduction to numpy, introduction to opencv
# CodinGame
