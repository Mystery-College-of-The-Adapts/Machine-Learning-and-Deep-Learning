
Topics
----------
1. Overview of TensorFlow
2. Graphs and Sessions



# Graphs and Sessions

## Data Flow Graphs
TensorFlow separates definition of computations from their execution
- Phase 1: Assemble a graph
- Phase 2: use a session to execute operations in the graph
![](assets/Selection_001.png)

## What is a tensor?
An n-dimensional matrix
- o-d tensor: scalar
- 1-d tensor: vector
- 2-d tensor: matrix
- and so on


```python
import tensorflow as tf
```


```python
a = tf.add(3,5)
```

![](assets/Selection_002.png)

why x, y?
TF automatically names the nodes when you don't explicitly name them. 

- x = 3
- y = 5

![](assets/Selection_003.png)
- **Nodes** are operators, variables, and constants
- **Edges** are tensors
- **Tensors** are data.  ---Data Flow -> Tensor FLow (Aha moment)


```python
print a
```

    Tensor("Add:0", shape=(), dtype=int32)


**Note**: (Not 5)

## How to get the value of a?
- Create a **Session**, assign it to variable sess so we can call it later
- Within the Session, evaluate the graph to fetch the value of a


```python
sess = tf.Session()
print sess.run(a)
sess.close()
```

    8


The session will look at the graph, trying to think: hmm, how can I get the value of a, then it computes all the nodes that leads to a

![](assets/Selection_004.png)


```python
b = tf.add(3 ,15)
with tf.Session() as sess:
    print sess.run(b)
```

    18


## More Graphs
![](assets/Selection_005.png)


```python
x = 2
y = 3
op1 = tf.add(x, y)
op2 = tf.mul(x, y)
op3 = tf.pow(op2, op1)
```


```python
with tf.Session() as sess:
    op3 = sess.run(op3)
    print op3
```

    7776


## More (sub) graphs
![](assets/Selection_006.png)
- Because we only want the value of z and z doesn't depend on useless, session won't compute values of useless
-> save computation


```python
x = 5
y = 2
op1 = tf.add(x, y)
op2 = tf.mul(x ,y)
useless = tf.mul(x, op1)
op3 = tf.pow(op2 , op1)
```


```python
with tf.Session() as sess:
    op3 = sess.run(op3)
    print op3
```

    10000000


![](assets/Selection_006.png)
```python 
tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)
```
- Pass all variables whose values you wnat to a list in fetches



```python
x = 2 
y = 3
op1 = tf.add(x, y)
op2 = tf.mul(x, y)
useless = tf.mul(x, op1)
op3 = tf.pow(op2 , op1)

```


```python
with tf.Session() as sess:
    op3, not_useless = sess.run([op3, useless])
    print op3,
    print not_useless
```

    7776 10


## More(sub) graphs
![](assets/Selection_007.png)
Possible to break graphs into several chunks and run them parallelly across multiple CPUs, GPUs, or devices.

```python
# Creates a graph.
with tf.device('gpu:2'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name ='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name ='b')
    c = tf.matmul(a, b)
    
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
print sess.run(c)

```

What if I want to build more than one graph? You can but you don't need more than one graph The session runs the default graph.

But what if I really want to ?

- Multiple graphs require multiple sessions, each will try to use all available resources by default
- Can't pass data between them without passing them through python/numpy, which doesn't work in distributed
- It's better to have disconnected subgraphs within one graph


## tf.Graph()


```python
# create a graph
g = tf.Graph()

# To add operators to a graph, set it as default:
with g.as_default():
    x = tf.add(3, 5)
```


```python
sess = tf.Session(graph=g)
with tf.Session(graph=g) as sess:
    print sess.run(x)
```

    8



```python
g = tf.Graph()
with g.as_default():
    a = 3
    b = 5
    x = tf.add(a, b)

```


```python
# session is run on the graph g
sess = tf.Session(graph=g)

# run session
sess.close()

```


```python
# Do not mix default graph and user created graphs
g = tf.Graph()

# add ops to the default graph
a = tf.constant(3)

# add ops to the user created graph
with g.as_default():
    
    b = tf.constant(5)
```

## Why Graphs

1. Save computation (only run subgraphs that lead to the values you want to fetch)
2. Break Computation into small, differential pieces to facilitates auto-differentiation
3. Facilitate distributed computation, spread the work across multiple CPUs, GPUs, or devices
4. Many Common machine learning models are commonly taught and visualized as directed graphs already.
