#functions to calculate LCM of task periods.
def GCD(a, b):
    if b > a:
        temp = a
        a = b
        b = temp 
    while(b>0):
        temp = b
        b = a%b
        a = temp
    return a

def LCM(a,b): 
    gcd = GCD(a,b)
    a = a // gcd
    b = b // gcd
    return (gcd * a * b)

def LCMmerge(periods):
    temp = periods[0].p
    for i in range(len(periods)-1):
        temp = LCM(temp,periods[i+1].p)
    return temp