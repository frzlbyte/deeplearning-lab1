<aside>
🧑‍🎓

Jordan Bradley, Ethan Lin, Nicholas Miller

</aside>

## Task I: Neural Network Design

**1. Explanation of Network Design Choices**

Our implementation includes three distinct neural network architectures as required by the assignment. For the fully connected network, we used four layers with decreasing neuron counts (128, 64, 32, 10) to progressively refine features while reducing dimensionality. We employed ReLU activation in the first and third layers to address the vanishing gradient problem, while using tanh activation in the second layer to capture both positive and negative relationships in the data.

For the locally connected network, we implemented a 1D convolutional approach with three locally connected layers that preserve spatial relationships without weight sharing. This architecture achieved 94.57% accuracy, demonstrating the value of incorporating local connectivity patterns for image data.

Our convolutional neural network used both 2D convolutions and max pooling operations. The implementation included batch normalization layers to stabilize learning and accelerate convergence. This architecture achieved the highest accuracy at 95.32%, confirming that weight sharing and hierarchical feature extraction are powerful for handwritten digit recognition.

## Task II: Techniques for Optimization

**1. Parameter Initialization Strategies**

Our analysis of parameter initialization strategies showed dramatic differences in learning dynamics across the three network types. We found that zeros initialization led to extremely slow learning, with accuracy stuck at approximately 17.9% after 3 epochs. This occurs because when all weights are initialized to the same value, neurons in the same layer compute identical outputs, making it impossible for the network to learn diverse features.

He-normal initialization proved highly effective, reaching 92.4% accuracy in just 3 epochs. This initialization method is specifically designed for ReLU activation functions, setting initial values with a variance of 2/n (where n is the number of input connections), which helps maintain healthy activation variances throughout the network.

Ones initialization resulted in unstable learning, achieving only 21.7% accuracy with high loss values. This initialization causes immediate saturation of activations, pushing neurons into regions of their activation functions where gradients are minimal, severely hindering learning.

**2. Learning Rate Analysis**

Learning rate experiments confirmed the critical importance of this hyperparameter. A slow learning rate of 0.0001 showed gradual improvement, reaching 90.8% accuracy after 3 epochs, but with slower convergence. The effective learning rate of 0.001 provided the best balance of speed and stability, achieving 94.9% accuracy with the lowest validation loss.

The fast learning rate of 0.1 demonstrated unstable training behavior, with accuracy decreasing to 45.8% by epoch 3, clearly showing the overshooting of minima in the loss landscape. This large step size caused oscillations that prevented convergence to an optimal solution.

**3. Batch Size Impact on Batch Normalization**

Our experiments on batch size showed significant effects on network performance when using batch normalization. With a small batch size of 8, training required more iterations per epoch but achieved 94.3% accuracy with more stable statistics. The large batch size of 128 failed to generalize well, achieving only 15.4% test accuracy despite good training accuracy.

This result confirms theoretical predictions: batch normalization calculates statistics over mini-batches, and when these batches are too large, they can mask important variations in the data distribution. Smaller batches provide more frequent weight updates and more diverse normalization statistics, leading to better generalization in this case.

**4. Momentum Analysis**

Momentum experiments on our CNN showed that a momentum value of 0.5 achieved 94.9% accuracy with the lowest loss of 0.168, suggesting efficient convergence. A standard momentum value of 0.9 achieved 92.3% accuracy with a slightly higher loss of 0.302, while a high momentum value of 0.99 performed well with 94.7% accuracy and a loss of 0.169.

These results demonstrate that momentum can significantly improve optimization, with higher values maintaining more of the previous gradient information. The appropriate momentum value depends on the specific loss landscape, with our experiments showing that both lower (0.5) and higher (0.99) values can outperform the standard 0.9 setting in certain cases.

## Task III: Techniques for Improving Generalization

**1. Ensemble Methods**

Our ensemble implementation successfully combined four neural networks with different architectures to improve generalization performance. The individual models achieved accuracies of 93.57% (FCN), 94.57% (Local CNN), 95.07% (CNN1), and 95.32% (CNN2) respectively.

Three ensemble strategies were tested:

- Simple averaging (soft voting) improved accuracy to 96.06%, a +0.74% gain over the best individual model
- Weighted averaging further improved performance to 96.11% (+0.79%), demonstrating the benefit of prioritizing more accurate models
- Majority voting (hard voting) achieved 95.76% (+0.44%), slightly lower than the soft voting methods but still better than any individual model

These results confirm that model ensembling effectively reduces overfitting and improves generalization by combining diverse learning perspectives. The weighted ensemble performed best because it adaptively emphasized predictions from more accurate models.

**2. Dropout Regularization**

Our dropout regularization experiments on the fully connected network revealed that for this particular task, dropout was not beneficial. Testing dropout rates of 0.0 (no dropout), 0.2, 0.5, and 0.8, we found that test accuracy consistently decreased as dropout rate increased: 93.92%, 92.92%, 92.48%, and 35.97% respectively.

The visualization of training dynamics with different dropout rates clearly showed that with high dropout (0.8), there was a massive gap between training and validation performance. The model struggled to learn effectively, with training accuracy barely exceeding 30% after 5 epochs.

These results suggest that for this particular dataset and model architecture, the network is not complex enough to benefit significantly from dropout regularization. The model may already be appropriately sized for the complexity of the task, so the additional regularization from dropout only served to impede learning rather than improve generalization.
