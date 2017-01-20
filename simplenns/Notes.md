# Notes

## Session vs InteractiveSession
The difference:
for `InteractiveSession`:

```python
  sess = tf.InteractiveSession()
  print tf.add(2, 2).eval()  # once an interactive session is opened, you can do what ever in between, don't forget to close it
  sess.close()
```

But you can not use the same format for `Session`
```python
  sess = tf.Session()
  print tf.add(2, 2).eval() # no default session is registered
  sess.close()
```

Instead, you may use the most used:
```python
   with tf.Session() as sess:
     print tf.add(2, 2).eval()
```
**either `run()` or `.eval()` cannot be executed outside a session context**

## Main phases of TF code:
1. Graph level : definition of the structure
2. Session level : execution and unravel the results

## Only variables keep their value between multiple evaluations.
Others come as feed-dict or some queue like structure


## (Exponential) Moving Average

```Python
  shadow_variable -= (1 - decay) * (shadow_variable - variable) # simplified version
  shadow_variable = decay * shadow_variable + (1 - decay) * variable # original version
```



## Python shallow copy, deep copy and normal assignment copy

For basic data types
```Python
  x = 3
  y = x # x and y are pointing to the same block in the memory, hex(id(x)) == hex(id(y)), x is y
  x = 4 # now x is pointing to another memory block, making the x is y returning false
```


For compound data such as list, tuple, dict or class:
1. Normal assignment operations will simply point the new variable towards the existing object.
2. A shallow copy constructs a new compound object and then (to the extent possible) inserts references into it to the objects found in the original.
3. A deep copy constructs a new compound object and then, **recursively**, inserts copies into it of the objects found in the original.

More detail take a look on [this](http://www.python-course.eu/deep_copy.php)

Note the operation `list2 = list[:]` is a shallow copy

## build on function:
1. `map()`: Apply function to every item of iterable and return a list of the results
2. `zip()`: a usage that illustrates the function is as follows:

```Python
  x,y = zip(*zip(x,y))
```
