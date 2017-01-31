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

## Understanding the elements of Tensorflow:

### Tensor
>A Tensor object is a symbolic handle to the result of an operation, but does not actually hold the values of the operation's output

Understanding tensor as a handle to the output of the operation, something like function handle, when executing this function in `sess.run`, the value will be returned.

The tensor and operation is closely connected. A Graph in tensorflow is actually a list of operations. Let's say the name of an operation is 'add', the corresponding tensor is to the output of operation, which can be retrieved by running `sess.run('add:0')`

#### Some knowledges of a tensor:
- `rank`: different from the mathematically concept of a tensor. A rank of a tensor is the number of dimensions of a tensor, for example, a tensor of shape `[2,3]` is rank 2, a scalar is rank `0`



### Variable
Do the following:
```Python
tf.reset_default_graph()
value = tf.Variable(tf.ones_initializer(()))
print [n.name for n in tf.get_default_graph().as_graph_def().node]
```
You will see Variable contains a list of operations such as `assign` and `read`.
My understanding of the Variable is that the variable is connected with a memory that stores a state. It's a special group of operation

>Variables are in-memory buffers containing tensors. They must be explicitly initialized and can be saved to disk during and after training. You can later restore saved values to exercise or analyze the model.

#### Variable initialization
When you create a Variable you pass a Tensor as its initial value to the `Variable()` constructor. TensorFlow provides a collection of ops that produce tensors often used for initialization from constants or random values.

Calling `tf.Variable()` adds several **sets of ops** to the graph:

- A variable op that holds the variable value. `Variable`
- An initializer op that sets the variable to its initial value. This is actually a tf.assign op. 'Variable/assign'
- The ops for the initial value, such as the `tf.ones_initializer()` in the example are also added to the graph.

#### Variable placement:
In some code with say the variable is declared on CPU, for example the sample code block excerpted from cifar example:
```Python
def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var
```
Operations that mutate a variable, such as v.assign() and the parameter update operations in a tf.train.Optimizer must run on the same device as the variable. Incompatible device placement directives will be ignored when creating these operations.
It is preferrable to put the variables on cpu if the P2P on GPU is not activated.
CPU is like the core to manage data. GPU is running everything, but the data should be on CPU
>All variables are pinned to the CPU and accessed via tf.get_variable() in order to share them in a multi-GPU version. See how-to on Sharing Variables.


#### Variables collections
- `LocalVariables`
- `GlobalVariables`


####  Saver
`tf.train.Saver` adds two **`ops`** `save` and `restore` to the graph for all or specified list of variables. When the op is running, it writes stuff into the file

#### Session
Session provides a running environment for the graph and variables. When we are talking about the values of variables, we are talking about it in the scope of a session. The values and the containers will be destoried outside of the session, that's the reason why the `saver.restore` and `saver.save` have to be run with a sess argument

#### Different ways of initializing a variable:
See [here](https://www.tensorflow.org/api_docs/python/constant_op/)

## Using `tf.app.flags.FLAGS`

if this is used: `FLAGS = tf.app.flags.FLAGS`, then FLAGS can be used as a global environmental variable safely.

## Scopes in tensorflow:
Check [here](http://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow)
There are 4 different kinds of scopes and can be divided into 2 categories:
- name scope, created using `tf.name_scope` or `tf.op_scope`
- variable scope, created using `tf.variable_scope` or `tf.variable_op_scope`

The explaination:
- Both scopes have the same effect on all operations as well as variables created using tf.Variable, i.e. the scope will be added as a prefix to the operation or variable name. There is no difference in using any of them for creating tf.Variable
- op_scope and variable_op_scope accept as arguments a list of values and validate these values are from the default graph, then create new operations based on scope or defaultscope with scope is None(deprecated)

An example:
```
with tf.name_scope('ns'):
    with tf.variable_scope('vs'):
        v1 = tf.get_variable("v1",[1.0])   #v1.name = 'vs/v1:0'
        v2 = tf.Variable([2.0],name = 'v2')  #v2.name= 'ns/vs/v2:0'
        v3 = v1 + v2       #v3.name = 'ns/vs/add:0'
```
Note the operation v3 does not distinguish name_scope and variable scope.

>**"most of the time if you think you need a scope, use tf.variable_scope()"**  

## Batch Normalization
** A natural way to replace bias term**
### The order of adding batch normalization:

```python
  x= conv()
  x=bn()
  x=activation()
```

An explaination of Batch Normalization can be found @ [here](https://gab41.lab41.org/batch-normalization-what-the-hey-d480039a9e3b#.l1k9l2mmu)

Another [implementation](http://r2rt.com/implementing-batch-normalization-in-tensorflow.html)

## Understanding Xavier Initialization and a lot of other initializations
Check [here](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)


### Why we need to initialize the weights
Check [here](http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network/186351#186351)

Briefly speaking, the initialization of weights from  -1/sqrt(d) to 1/sqrt(d) where d is the number of input is to avoid the range of the output too far away from 0 so that the gradients vanish.

But the initialization is conditioned on the prerequisites that the input is 0 mean and 1 variance.

**That's the reason why BN works so well**

### For training a large network

Refer to ResNet
- Using Batch Normalization
- Update the structure to make better predictions

## Print stuff immediately
Add `sys.stdout.flush()` right after `print`

## A New model

Given an input `d*m*m*c` where d is the number of frames, m is the spatial resolution, c is the number of channels
We learn a weight indicator  `k*k*k*1` to convolve with `d*m*m*c` to have  `d*m*m*1` weights. Based on the rankings of weights(using [nn.top-k](https://github.com/tensorflow/tensorflow/issues/288)), we are going to pick the top-k (if max-pooling, top-k will be half the original elements) using function[tf.gather?](https://github.com/tensorflow/tensorflow/issues/418), and resize them back into `d/2*m/2*m/2*c'` to fit for the pooling size.

**Note** this should be based on fine-tuning, not sure if this is good for training from scratch.
TODO:
Implement this on CIFAR10 first, replace each max-pooling with the proposed method
