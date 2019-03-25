a, b = 0, 1
while b < 10:
    print(b)
    a, b = b, a + b


var1 = 100
if var1:
    print("1 - if 表达式为 true")
    print(var1)

var2 = 0
if var2:
    print("2 - if 表达式为true")
    print(var2)

print("Good bye!")


lit = [1,2,4,5]
it = iter(lit)
for x in it:
    print(x, end = ' ')