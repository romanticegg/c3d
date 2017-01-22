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

## build in function:
1. `map()`: Apply function to every item of iterable and return a list of the results
2. `zip()`: a usage that illustrates the function is as follows:

```Python
  x,y = zip(*zip(x,y))
```

## Understanding arguments in `tf.train.shuffle_batch`

--------------
`enqueue_many`:

If `enqueue_many` is False, tensors is assumed to represent a single example. An input tensor with shape `[x, y, z]` will be output as a tensor with shape `[batch_size, x, y, z]`.

If `enqueue_many` is True, tensors is assumed to represent a batch of examples, where the first dimension is indexed by example, and all members of tensors should have the same size in the first dimension. If an input tensor has shape `[*, x, y, z]`, the output will have shape `[batch_size, x, y, z]`.

## Never forget to run `tf.train.start_queue_runners(sess=sess)` when using Queue stuff to load data
The potential problem:
```
W tensorflow/core/kernels/queue_base.cc:294] _0_input_producer: Skipping cancelled enqueue attempt with queue not closed
ERROR:tensorflow:Exception in QueueRunner: Attempted to use a closed Session.
ERROR:tensorflow:Exception in QueueRunner: Attempted to use a closed Session.
ERROR:tensorflow:Exception in QueueRunner: Attempted to use a closed Session.
ERROR:tensorflow:Exception in QueueRunner: Attempted to use a closed Session.
ERROR:tensorflow:Exception in QueueRunner: Attempted to use a closed Session.
ERROR:tensorflow:Exception in QueueRunner: Attempted to use a closed Session.
ERROR:tensorflow:Exception in QueueRunner: Attempted to use a closed Session.
```


## The problem of `Your branch is ahead of 'origin/master' by 3 commits`

- In a good workflow your remote copy of master should be the good one while your local copy of master is just a copy of the one in remote. Using this workflow you'll never get this message again.
- If you work in another way and your local changes should be pushed then just git push origin assuming origin is your remote
- If your local changes are bad then just remove them or reset your local master to the state on remote `git reset --hard origin/master`

## get add . commit and push in one line:

function lazypush() {
    git add .
    git commit -a -m "$1"
    git push origin/master
}
