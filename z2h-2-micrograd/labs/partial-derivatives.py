# Card: partial-derivatives · Lecture 1 · https://www.youtube.com/watch?v=VMj-3S1tku0&t=852s
# Run from the course folder: python labs/partial-derivatives.py
h = 0.0001

for knob in ['a', 'b', 'c']:
    # inputs
    a = 2.0
    b = -3.0
    c = 10.0
    d1 = a*b + c
    if knob == 'a': a += h
    if knob == 'b': b += h
    if knob == 'c': c += h
    d2 = a*b + c
    print(f'bump {knob}:  d1 {d1}  d2 {d2}  slope {(d2 - d1)/h}')
