# Card: derivative-rules (the 7 rules, checked numerically) · L1 https://www.youtube.com/watch?v=VMj-3S1tku0&t=488s
# Run from z2h-1-toolkit/: python labs/derivative-rules.py
import math

def f(x):
    return 3*x**2 - 4*x + 5

h = 0.0001
x = 3.0
print(f'f(3) = {f(x)}')
print(f'(f(3+h) - f(3))/h, h={h}: {(f(x + h) - f(x))/h:.6f}')
print(f"f'(3) = 6*3 - 4 = {6*x - 4}")
for hh in [0.1, 0.001, 0.00001, 1e-12]:
    print(f'  h={hh:<7g} slope = {(f(x + hh) - f(x))/hh:.10f}')

print('\nrule            point   exact        numeric (h=1e-6)')
h = 1e-6
rules = [
    ('x^3 -> 3x^2',     lambda x: x**3,          lambda x: 3*x**2,             2.0),
    ('e^x -> e^x',      math.exp,                math.exp,                     1.0),
    ('ln x -> 1/x',     math.log,                lambda x: 1/x,                2.0),
    ('tanh -> 1-tanh^2',math.tanh,               lambda x: 1 - math.tanh(x)**2, 0.8813735870195432),
    ('a*b, d/da -> b',  lambda a: a*(-3.0),      lambda a: -3.0,               2.0),
    ('a/b, d/db -> -a/b^2', lambda b: 2.0/b,     lambda b: -2.0/b**2,          4.0),
    ('a+b, d/da -> 1',  lambda a: a + 10.0,      lambda a: 1.0,                2.0),
]
for name, fn, d, p in rules:
    num = (fn(p + h) - fn(p)) / h
    print(f'{name:19s} {p:<7.4g} {d(p):<12.6f} {num:.6f}')
