The MNIST dataset is a benchmark in the field of statistics, machine learning and computer
vision, consisting of 70,000 grayscale images of handwritten digits, each of size 28x28 pixels,
with each image labeled from 0 to 9. For our exercises, we have 10,000 images’ data as the
training dataset, and 5,000 images’ data as the testing dataset out of the original 70,000
images.

The task is to develop a model to accurately classify the handwritten digits. The
motivation behind this project is to explore and evaluate the accuracy of different statistical
models in addressing the MNIST classification problem. Building on the results of a naive
probabilistic model, which achieved a misclassification rate of about 22%, this project aims
to push the boundaries of accuracy by employing more sophisticated models and techniques.
The specific goals I’ve set for the project are:

## 1. Compare the prediction performance of various models:
(a) Multinomial Logistic Regression (with Lasso, Ridge, and Elastic Net) <br>
(b) Decision Trees and Random Forests<br>
(c) Multi-class Support Vector Machine (SVM)<br>
(d) Multi-class Linear Discriminant Analysis (LDA)<br>
(e) Deep Neural Network (DNN and CNN)

## 2. Describe the working procedures of the algorithms, and rationale for choosing them.

## 3. Optimize and tune each model to achieve the best possible classification accuracy.

## 4. Provide a comprehensive analysis and compare their performances.
### Results: 
<table>
  <tr>
    <th>Method</th>
    <th>Accuracy</th>
    <th>Misclassification Rate</th>
  </tr>
  <tr>
    <td>Multinomial Logistic Regression</td>
    <td>0.899</td>
    <td>11%</td>
  </tr>
  <tr>
    <td>Decision Trees</td>
    <td>0.8047</td>
    <td>20%</td>
  </tr>
    <tr>
    <td>Random Forests</td>
    <td>0.925</td>
    <td>7.5%</td>
  </tr>
    <tr>
    <td>Support Vector Machine</td>
    <td>0.937</td>
    <td>6.3%</td>
  </tr>
    <tr>
    <td>Linear Discriminant Analysis</td>
    <td>0.845</td>
    <td>15.5%</td>
  </tr>
    <tr>
    <td>Deep Neural Network</td>
    <td>0.9287</td>
    <td>7.1%</td>
  </tr>
    <tr>
    <td>Convolutional Neural Network</td>
    <td>0.9784</td>
    <td>2.2%</td>
  </tr>
</table>
