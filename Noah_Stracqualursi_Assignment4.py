#!/usr/bin/env python
# coding: utf-8

# # Getting Python
# 
# 
# For this course, we are going to use Jupyter notebook as our environment for developing Python code.
# refer to https://jupyter.readthedocs.io/en/latest/content-quickstart.html on the instructions how to install it, the easiest way is to install from Anaconda (https://www.anaconda.com/download/) website, make sure you install with Python 3.6.
# 
# Also, it is good for the students who are not familiar with python (or they need a quick refreshment) to follow Jim Bagrow tutorial http://bagrow.com/ds1/whirlwindtourpython/00-Title.html. 
# 
# All the assignments to be written in Python 3.6 and can be run using Jupyter on one of the following Internet browsers (Chrome, Safari or Firefox), these are the browsers that officially supported by jupyter.
# 
# <u> Note: for this assignment, submit your local copy of this page, running on IPython. Submit the file to Blackboard under Assignment4 using this file format:</u> <b>Yourfirstname_lastname_Assignment4.ipynb. Marks might be deducted if you do not follow the submission steps</b> 
# 
# #### <b>Deadline</b>: <u>Friday, April-17-2020 11:59 PM.</u>

# # Assignment 4

# In[2]:


##### Always import all needed libraries in the first cell
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Dataset import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

get_ipython().run_line_magic('matplotlib', 'inline')


# # Planar data classification
# 
# In this assignment you will learn how to:
# - Build a Multi layer perceptron neural network.
# - Implement the forward propagation.
# - Implement backpropagation 
# - Testing skills to debug your neural network.
# 
# 
# 
# Hint: Always use vectorized version. It will save your time and the code will run more faster.
# 

# ## Loading dataset
# 
# First,load the dataset on which you will be working on. The following line will load a "flower"dataset with 2-class dataset into variables features and label.

# In[3]:


features, label = load_planar_dataset()
print(features)
print(label)


# Try to visualize the data from the dataset using your favourite visualization library. The data looks like a "flower" with some red and blue points. The goal of this assignment is to build a model that will tell the class of the data(red or blue flower). 

# ### Question 1 --5 points

# In[16]:


# Visualize the data:
#Write you code here
plt.scatter(*features, c=label.ravel(), s=10, cmap=plt.cm.Spectral)


# Before diving deep,Lets try to get a better sense of what our data is. 
# 
# ### Question 2-- 5 points
# 
# How many training examples do you have? In addition, what is the shape of the variables feature and label ? 
# 

# In[6]:



shape_X = np.shape(features)
shape_Y = np.shape(label)
m = np.shape(features[1])

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have %d training samples!' % (m))


# ## Question 3 -- 5 points
# 
# As it is a binary classification, it is good way to give it a try with logictic regression. 
# Use sklearn's built-in Logistic regression (with no polynomials) and report the accuracy score.
# Hint: Accuracy will be around 50%, do cross validation for testing ...

# In[18]:


# Train the logistic regression classifier below:
#Start you code from here
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(features[1], label.ravel())
y_train = y_train.reshape(-1, 1)
X_train = X_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
modelLR = LogisticRegressionCV(cv=5, random_state=0).fit(X_train, y_train)
y_pred = cross_val_predict(modelLR, X_test, y_test, cv=5)
accuracy_score(y_test,y_pred)


# ## Question 4 -- 5 points
# Plot the decision boundary of these model below and also print the accuracy.

# In[19]:


# Write code below for plotting the decision boundary for logistic regression 

def plot_decision_boundary_2(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:,0], X[:,1], c=y.ravel(), s=10)
    # plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    
# Write code below for printng the accuracy
model = LogisticRegressionCV(cv=5)
model.fit(features.T,label.T)
plot_decision_boundary_2(model,features.T,label.T)


# ## Question 5 -- 5 points
# 
# Do you know the reason of poor performance of logistic regression?
# Can you explain it briefly?

# In[20]:


#The reason is becase logistic regression has a hard time separating nonlinearly seperable data. Since this data
#has a flowering pattern, it is not linearly seperable therefore it will have a hard time performing well.


# ## - Neural Network model
# 
# As you can see,logistic regression did not do well. We need to switch from traditional binary classification algorithms to multi layer perceptron (Hopefully it will help). 
# 
# Build a neural network with one hidden layer to see if it can perform better than the logistic regression. 
# 
# For your convenience,a diagram and the mathematical formulas of the network you will build are given below:
# 
# 
# ![Screen%20Shot%202020-03-29%20at%204.39.49%20PM.png](attachment:Screen%20Shot%202020-03-29%20at%204.39.49%20PM.png)
# 
# **Mathematically**:
# 
# 
# $$a^{(1)} =  X \tag{1}$$
# $$z^{(2)} =  W^{1} a^{(1)} \tag{2}$$ 
# $$a^{(2)} = g(z^{(2)}) + a^{(2)}_{0}\tag{3}$$
# $$z^{(3)} = W^{2} a^{(2)}\tag{4}$$
# $${h}_{(w)}(X) = a^{(3)} = g(z^{ (3)})\tag{5}$$
# 
# Given the predictions on all the examples, you can also compute the cost (cross-entropy cost in this case) $J$ as follows: 
# $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{(3) (i)}\right) + (1-y^{(i)})\log\left(1- a^{(3) (i)}\right)  \large  \right) \small \tag{6}$$
# 
# 
# Note that X could contain more than one example, X dimensions is the features numbers x number of samples.
# 
# To build a neural network, you need the following building blocks beforehand:
# 
# 1. Layer defination
# 2. Initialization of parameters
# 3. Forward propagation
# 4. Cost computation
# 5. Backward propagation
# 6. Update of parameters after the run of gradient decent
# 
# This building blocks will be explained as we go along the the assignment.
# We are going to build a three layer MLP network. But you need to define the layer size. For defining the input and output layer size, you need to keep an eye on the size of your training data and number of classes you have respectively.The size of the hidden layer is something you can play with. Choose the size according to your wish.
# Define three variables named input_layer_size( the size of the input layer),hidden_layer_size( the size of the hidden layer),output_layer_size(the size of the output layer).
# 
# For this part of the assignemnt we are going to use ONE hidden layer with four hidden nodes as defined below ...
# 
# 

# In[21]:


def layerSizes(X, Y):
    """
    X -- input dataset of shape 
    Y -- labels of shape
    """
    input_layer_size = X.shape[0]
    hidden_layer_size = 16
    output_layer_size = 1 
    
    """
    Returns:
    input_layer_size -- the size of the input layer
    hidden_layer_size -- the size of the hidden layer
    output_layer_size -- the size of the output layer
    """
    
    return (input_layer_size, hidden_layer_size, output_layer_size)


# In[22]:


# just run this code to test layerSizes function 
input_layer_size, hidden_layer_size, output_layer_size=layerSizes(features,label)
print("The size of the input layer is: = " + str(input_layer_size))
print("The size of the hidden layer is: = " + str(hidden_layer_size))
print("The size of the output layer is: = " + str(output_layer_size))


# 
# ## Question 6 -- 10 points
# You need to initialize neural network with random values. Implement the initialize_parameters function.
# 
# Hint:Initialize the weights matrices with random values .

# In[23]:


def initialize_parameters(input_size, hidden_size, output_size):
    """
    input_size-- size of the input layer
    hidden_size -- size of the hidden layer
    output_size-- size of the output layer
    """
    np.random.seed(2)  # you can pick any seed in this case
    
    Weight1 = np.random.rand(hidden_size, input_size + 1) 
    Weight2 = np.random.rand(output_size, hidden_size + 1) 
    
    parameters = {'Weight1': Weight1, 'Weight2': Weight2}
    
    """
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix 1
                    W2 -- weight matrix 2 
    """
    return parameters


# In[24]:


parameters = initialize_parameters(*layerSizes(features, label))
#Write you code here
print("Weight1 = " + str(parameters["Weight1"].shape))
print("Weight2 = " + str(parameters["Weight2"].shape))


# ## Question 7 -- 15 points
# 
#  In this part forward propagation will be implemented. It will propagate the gradient in the forward pass from one layer to another.
# 
# At first,look at the mathematical representation of the classifier.For this assignment we are asking you to use tanh for the hidden layer and sigmoid for the output layer. (For neural network in general you can use any non-linear activation function, you might try at some point to change those and see if you can achieve better accuracy)
# 
# Then, do the following:
# 
# Retrieve each parameter from the dictionary "parameters" (which is the output of initialize_parameters function) by using parameters dictionary.
# 
# Implement Forward Propagation. Compute $z^{(2)}, a^{(2)}, z^{(3)}$ and $a^{(3)}$ (the vector of all your predictions on all the examples in the training set).
# 

# In[25]:


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def forward_prop(X, parameters):
    """
    X -- input data of size
    parameters -- python dictionary containing your parameters (output of initialization function)
    """
    # Retrieve each parameter from the dictionary "parameters"
    Weight1 = parameters['Weight1'] 
    Weight2 = parameters['Weight2'] 
    
   
    # Implement Forward Propagation to calculate a3 (probabilities)
    one=np.ones(X.shape[1])  
    X=np.vstack ((one, X) ) #Adding ones as bias
    
    z2 = np.dot(Weight1,X)
    
    a2 = np.tanh(z2)
    
    one = np.ones(a2.shape[1])
    a2  = np.vstack ((one, a2) )
    
    z3 = Weight2.dot(a2)
    a3 = sigmoid(z3)
    
    
    #The values will be needed for the backpropagation which are stored in cache. Later, it will be given to back propagation.
    cache = {"z2": z2,
             "a2": a2,
             "z3": z3,
             "a3": a3}
    """
    Returns:
    a3 -- The sigmoid output of the second activation
    cache -- a dictionary containing "z2", "a2", "z3" and "a3"
    """
    return np.array(a3), cache


# In[26]:


#test function for forward propagation

np.random.seed(1) 
#In class you were dealing with only 1 sample or in other words feeding only 1 sample.
#But, you can also feed multiple samples,which we will be doing here. Here you will see, X has multiple samples.

X_assess = np.random.randn(2, 2)

parameters = {'Weight1': np.array([[-0.00416758, -0.00056267,-0.00056127],
        [-0.02136196,  0.01640271,-0.00056123],
        [-0.01793436, -0.00841747,-0.00036123],
        [ 0.00502881, -0.01245288,-0.00026117]                           
        ]),
     'Weight2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208,0.02292223]])}


a3, cache = forward_prop(X_assess, parameters)

# Note: we use the mean here just to make sure that your output matches ours. 
print(np.mean(cache['z2']) ,np.mean(cache['a2']),np.mean(cache['z3']),np.mean(cache['a3']))

#the output of this print should be like below:

#-0.00989624831022699 0.19208511191331104 -0.011139109844842423 0.4972152514624493


# To compute the cost, you need $a^{(3)}$. You have computed  $a^{(3)}$ (in the Python variable "`a3`"), which contains $a^{(3)(i)}$ for every example, you can compute the cost function as follows:
# 
# $$J = - \frac{1}{m} \sum\limits_{i = 1}^{m} \large{(} \small y^{(i)}\log\left(a^{(3) (i)}\right) + (1-y^{(i)})\log\left(1- a^{(3) (i)}\right) \large{)} \small\tag{7}$$
# 
# ## Question 8 -- 15 points 
#  
#  Implement compute_cost() to compute the value of the cost $J$.
# 
# **Instructions**:
# - There are many ways to implement the cross-entropy loss. Feel free to choose your conveneient way.
# 
# 
# 
# 
#  

# In[27]:


def compute_cost(a3, Y):
    """
    Computes the cross-entropy cost given in equation (7) or in other way you may want.
    
    Arguments:
    a3 -- The sigmoid output of the second activation
    Y -- "true" labels vector of shape 
    
    """
    m = Y.shape[1]  # number of example

    # Compute the cross-entropy cost
    logprobs = Y.dot(np.log(a3).T) + (1-Y).dot(np.log(1-a3).T)
    cost = -1/m*np.sum(logprobs) 

    ### Remember that, if you want to use different cross-entropy loss, you need to change logprobs and cost accordingly
    cost = float(np.squeeze(cost))
    
    return cost


# In[28]:


#test function for compute_cost
np.random.seed(1) 
Y_assess = np.random.randn(1, 3)

a3 = (np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]))
print("cost = " + str(compute_cost(a3, Y_assess)))
#the cost will be around 0.69


# ## Question 9 -- 15 points 
# Back propagation is an important part of a network. This will implement ONE back propagation iteration (You can have full batch by feeding all samples in the X array). 
# Now, Implement the backward_propagation function. 
#  Mathematical formula's are given below for your convenience.
# 
# 
# Remember that you need to find the derivative of any weight with respect to the cost function. Then, multiply it with the activation function in order to update using the GD algorithm. In the code we will use d as a shorthand of delta.
# 
#  
#  ![Screen%20Shot%202020-03-29%20at%201.55.29%20PM.png](attachment:Screen%20Shot%202020-03-29%20at%201.55.29%20PM.png)
#  
# 
# - Hint:
#     - To compute d2 you'll need to compute $g^{(2)'}(z^{(2)})$. remember you have two dfferent activation function.
#     - derivative of tanh(x) = 1 - $tanh(x)^2$

# In[29]:


def backward_propagation(parameters, cache, X, Y):
    """
    parameters -- dictionary containing our parameters 
    cache -- a dictionary containing "z2", "a2", "z3" and "a3".
    X -- input data 
    Y -- "true" labels vector 
    """
    m = X.shape[1]
    
    # Copy W1 and W2 from the dictionary "parameters"
    Weight1 = parameters['Weight1']
    Weight2 = parameters['Weight2']
    one=np.ones(X.shape[1])
    X=np.vstack ((one, X) )
    # Copy A1 and A2 from dictionary "cache".
    z2 = cache['z2'] 
    a2 = cache['a2'] 
    a3 = cache['a3'] 
    
    #  calculate d3, dW2. 
    d3 =a3-Y
   
    dW2 = d3.dot(a2.T)
    
    #Below we will calculate delta 2. In the hidden layer,we have bias which we will exclude as it is not going backward.
    #We will be using tanh were as an activation function
    
    d2 = Weight2.T @ d3
    d2 = d2[1:] #We need to exclude the bias as it is not backpropagating.
    
    d2 = d2 * (1 - np.power(np.tanh(z2), 2))
    
    dW1 = d2 @ X.T 
    
    #in the gradient dict we will keep the update for the weight vectors.
    
    gradient = {"dW1": 1/m*dW1,
                "dW2": 1/m*dW2}
    
    return gradient


# In[30]:


#test function for backward propagation
np.random.seed(1) 
X_assess = np.random.randn(2, 1)
Y_assess = np.random.randn(1, 1)

parameters = {'Weight1': np.array([[-0.00416758, -0.00056267,-0.0005612345],
        [-0.02136196,  0.01640271,0.0005612343],
        [-0.01793436, -0.00841747,-0.0005612365],
        [ 0.00502881, -0.01245288,-0.0005211234]]),
     'Weight2': np.array([[-0.01057952, -0.00909008,0.00551432,  0.00551454,  0.02292208]])}

cache = {
    'z2': np.array([[-0.00616586],
         [-0.05229879],
         [-0.02009991],
         [ 0.02153007]]),
    'a2': np.array([[1],
         [-0.05225116],
         [-0.02009721],
         [ 0.02152675],
         [ 0.02152675]]),
    'z3': np.array([[ 0.00092281]]),
    'a3': np.array([[ 0.5002307]])  }

grads = backward_propagation(parameters, cache, X_assess, Y_assess) # call the back propagation here with appropriate parameters

print ("dW1 = "+ str(grads["dW1"]))
print ("dW2 = "+ str(grads["dW2"]))

#The output should be 


#dW1 = [[-0.00934791 -0.01518423  0.00571864]
#[ 0.00565546  0.00918642 -0.00345976]
# [ 0.00566888  0.00920821 -0.00346797]
# [ 0.0235622   0.03827315 -0.01441433]]
#dW2 = [[ 1.02840245 -0.05373522 -0.02066802  0.02213816  0.02213816]]


# ## Question 10 -- 15 points 
# This update rule will help back propagation to pass the update backward. 
# 
# Now,Implement the update rule using your known gradient descent. 
# 
# Hint:
# 
# You have to use (dW1, dW2) in order to update (W1, W2).
# 
# 

# In[31]:


def update_parameters(parameters, grads, learning_rate = 0.5):
    """
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    learning_rate -- learning rate you will choose
    """
    # Copy the following parameter from the dictionary "parameters"
    Weight1 = parameters['Weight1'] 
    Weight2 = parameters['Weight2'] 
     
    # Copy each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    
    dW2 = grads['dW2']
    
    # Update rule for each parameter
    
    Weight1 = Weight1 - learning_rate*dW1
    Weight2 = Weight2 - learning_rate*dW2

    
    parameters = {"Weight1": Weight1,
                  "Weight2": Weight2}
    
    """
    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    return parameters


# In[32]:


#test function update_parameters
np.random.seed(1) 
parameters = {'Weight1': np.array([[-0.00416758, -0.00056267,-0.00056127],
        [-0.02136196,  0.01640271,-0.00056123],
        [-0.01793436, -0.00841747,-0.00036123],
        [ 0.00502881, -0.01245288,-0.00026117],
        ]),
     'Weight2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208,0.02292223]])}

grads = {'dW1': np.array([[ 0.00023322, -0.00205423,-0.00205423],
        [ 0.00082222, -0.00700776,-0.00205423],
        [-0.00031831,  0.0028636,-0.00205423 ],
        [-0.00092857,  0.00809933,-0.00205423]]),
 'dW2': np.array([[ -1.75740039e-05,   3.70231337e-03,  -1.25683095e-03, -2.55715317e-03,-2.55715317e-03]])}
parameters = update_parameters(parameters, grads)

print("Weight1 = " + str(parameters["Weight1"]))

print("Weight2 = " + str(parameters["Weight2"]))


#Output should be 

#Weight1 = [[-0.00428419  0.00046445  0.00046585]
# [-0.02177307  0.01990659  0.00046589]
# [-0.0177752  -0.00984927  0.00066589]
# [ 0.00549309 -0.01650255  0.00076595]]
#Weight2 = [[-0.01057073 -0.01094124  0.00614296  0.02420066  0.02420081]]


# ## Question 11 -- 15 points 
# 
# The building blocks that we have mentioned previously are done now. Now, its time to put all together.
# 
# In this part of the question, those component will be put together to build the model.
# 
# Implement the neural network model in the function named model.
# 
# Hint:Don't forget to use the previous functions in the right order.

# In[33]:


def model(X, Y, n_h, num_epochs=1000, learning_rate=0.1, print_cost=True):
    """
    X -- dataset
    Y -- labels
    n_h -- size of the hidden layer
    learning_rate=the learning rate you are choosing,
    num_epochs -- Number of iterations in gradient descent
    print_cost -- if True, print the cost in every 100 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    n_x = X.shape[0]
    n_y = len(np.unique(Y))
    
    if n_y <= 2:
        n_y = 1
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y) #Write your code here
    Weight1 = parameters['Weight1']
    
    Weight2 = parameters['Weight2']
   
    
    # gradient descent
    

    for i in range(0, num_epochs):
         
        # Call the Forward propagation with X, and parameters.
        
        a3, cache = forward_prop(X, parameters)
        
        # Call the Cost function with a3, Y and parameters.
        
        cost = compute_cost(a3, Y)
        
        # Call Backpropagation with Inputs, parameters, cache, X and Y.
        grads = backward_propagation(parameters, cache, X, Y)
        
        # Update gradient descent parameter with  parameters and grads and learning rate.
        parameters = update_parameters(parameters, grads, learning_rate)
 
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters # refer to the code in the second next cell to run your model code


# ## Question 12 -- 15 points
# 
# Now, you have the model in your hand.It is time to use it and see it in action. Use the model to predict  using the function named predict and also plot the decision boundar
# 
# Hint:Use forward propagation to predict results.
# 

# In[34]:


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    parameters -- python dictionary containing your parameters 
    X -- input data
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    #Write your code here
    a3, cache = forward_prop(X, parameters)
    predictions = (a3 > 0.5)
    
    return predictions


# In[38]:


# Build a model with a n_h-dimensional hidden layer
    
parameters = model(features, label, 4, num_epochs=15000)
#plot the decision boundary
plot_decision_boundary(lambda X: predict(parameters, X.T), features, label.ravel())


# Now, run the model and see how it performs on a planar dataset. Run the following code to test your model with a single hidden layer of $n_h$ hidden units.

# In[37]:


# Print accuracy
predictions = predict(parameters, features)
print ('Accuracy: %d' % float((np.dot(label,predictions.T) + np.dot(1-label,1-predictions.T))/float(label.size)*100) + '%')


# **Expected Output**: 
# 
# <table style="width:15%">
#   <tr>
#     <td>**Accuracy**</td>
#     <td> 88% </td> 
#   </tr>
# </table>

# As you can see,accuracy is really high compared to Logistic Regression. The model has learnt the pattern of the leaves which is awesome. This can give you a sense that neural networks are able to learn even highly non-linear decision boundaries, unlike logistic regression. 
# 

# # Grad Student part (easy to get points)
# 
# 
# ## Question 13 -- 15 points
# 
# Try out several hidden layer(at least 4), to see if it improves the accuracy or not. 
# Run the following code. It may take 1-2 minutes. You will observe different behaviors of the model for various hidden layer sizes.

# In[ ]:

# ## Question 14 -- 5 points
# 
# Do you think that the model is overfitting ? What is the layer size for hidden layer? Why do you think so?

# ## Question 15 -- 5 points
# 
# Play with the learning_rate. What happens for different learning rate?

# # Performance on other datasets
# 
# # Extra credit for all students:
# 
# # Question 16 -- 10 points

# In this part, try to get the accuracy on noisy data above 88%.
# 
# If you want, you can rerun the whole notebook (minus the dataset part) for the following datasets.
# 
# You will get full marks for this only if you can take the accuracy above 88%. It might be time consuming. So, plan accordingly. Feel free to change the size of hidden layer or adding a new hidden layer. If you have any other idea feel free to do(but have to be using Multi layer perceptron)

# In[ ]:



noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}


dataset = "noisy_moons" #use this dataset only


X, Y = datasets[dataset] #Write your code here
X, Y = X.T, Y.reshape(1, Y.shape[0]) #Write your code here

# Visualize the data
plt.scatter(*X, c=Y.ravel())

#Write your code here
from sklearn.preprocessing import StandardScaler

scaled_X = StandardScaler().fit_transform(X.T).T
plt.scatter(*scaled_X, c=Y.ravel())


# In[ ]:


parameters = model(scaled_X, Y, 8)
plot_decision_boundary(lambda X: predict(parameters, X.T), scaled_X, Y.ravel())#Write your code here


# In[ ]:


predictions = predict(parameters, scaled_X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

