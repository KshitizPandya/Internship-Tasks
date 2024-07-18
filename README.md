# Internship-Tasks
 This repo consists of the internship tasks performed in LGM Data Science Internship

# Start of the Internship

Sure! Here's a breakdown of the topics you should cover in the first week to prepare yourself for these projects:

## Week 1

#### Day 1: Introduction to Neural Networks
- **Topics:**
  - Basic concepts of Neural Networks (NNs)
  - Perceptron, activation functions, and layers
  - Forward and backward propagation
  - Loss functions and optimization algorithms
- **Resources:**
  - Online tutorials and videos on Neural Networks
  - Relevant chapters from "Deep Learning" by Ian Goodfellow

#### Day 2: Convolutional Neural Networks (CNNs)
- **Topics:**
  - Understanding CNN architecture
  - Convolutional layers, pooling layers, and fully connected layers
  - Applications of CNNs in image recognition
- **Resources:**
  - Coursera/YouTube videos on CNNs
  - Articles and blogs on CNN fundamentals

#### Day 3: Recurrent Neural Networks (RNNs) and LSTM
- **Topics:**
  - Basics of RNNs and how they handle sequential data
  - Vanishing gradient problem and the need for LSTMs (Long Short-Term Memory)
  - Applications of RNNs and LSTMs in time-series data and NLP
- **Resources:**
  - Online tutorials on RNNs and LSTM
  - Blogs and articles explaining the architecture and use-cases

#### Day 4: Data Preprocessing and Augmentation
- **Topics:**
  - Importance of data preprocessing
  - Techniques for data normalization, standardization, and augmentation
  - ImageDataGenerator in TensorFlow/Keras
- **Resources:**
  - TensorFlow/Keras documentation on data preprocessing
  - YouTube tutorials on data augmentation techniques

#### Day 5: Basics of Transfer Learning and Model Evaluation
- **Topics:**
  - Concept of transfer learning and its advantages
  - Popular pre-trained models (e.g., VGG16, ResNet) and how to use them
  - Model evaluation metrics (accuracy, precision, recall, F1 score)
- **Resources:**
  - Articles and tutorials on transfer learning
  - Online resources explaining model evaluation metrics and their importance

### Week 1 Summary
- **Day 1:** Basic concepts of Neural Networks, forward and backward propagation, loss functions
- **Day 2:** CNN architecture, layers, and applications in image recognition
- **Day 3:** RNNs, LSTMs, and their applications in sequential data and NLP
- **Day 4:** Data preprocessing and augmentation techniques using TensorFlow/Keras
- **Day 5:** Transfer learning, pre-trained models, and model evaluation metrics

By following this study plan, you'll have a solid foundation to start working on the projects you've outlined.

## WEEK - 2
### Weekly Report: Development of Neural Network for Handwriting Recognition

#### Day 1: Research and Planning
**Activities:**
- Conducted extensive research on neural networks, with a focus on Convolutional Neural Networks (CNN) and their applications in image recognition tasks.
- Studied the MNIST Handwritten Digit Classification Challenge to understand its requirements and significance.
- Reviewed relevant literature and tutorials on TensorFlow, Keras, and other essential libraries.
- Planned the project timeline, identifying key milestones and deliverables.

#### Day 2: Dataset Acquisition and Preliminary Exploration
**Activities:**
- Downloaded the MNIST dataset from the TensorFlow Keras dataset module.
- Loaded the dataset into the environment and performed a preliminary exploration to understand its structure and contents.
- Visualized the distribution of digit classes in the dataset using Seaborn's count plot for a better understanding of data balance.
- Displayed sample images from the dataset to ensure the data was loaded correctly and to get a sense of the variation in handwriting styles.

#### Day 3: Data Preprocessing and Model Architecture Design
**Activities:**
- Reshaped and normalized the dataset to make it suitable for model training.
- Verified the shape and structure of the training and testing datasets.
- Designed the neural network architecture using TensorFlow and Keras.
- Included three convolutional layers with ReLU activation and kernel size of (3,3), followed by Batch Normalization and Dropout layers to prevent overfitting.
- Added a MaxPooling layer and a Dense layer with a Softmax activation function to convert predictions into probabilities.
- Summarized the model architecture to ensure it matched the intended design.

#### Day 4: Model Training
**Activities:**
- Compiled the model using the Adam optimizer and sparse categorical cross-entropy loss function.
- Trained the model on the training dataset for 10 epochs, with a validation split of 10%.
- Monitored the training process and recorded the accuracy and loss values for both training and validation sets.
- Adjusted training parameters and hyperparameters as needed to improve model performance.

#### Day 5: Model Evaluation and Visualization
**Activities:**
- Evaluated the model's performance on the test dataset, achieving a high accuracy.
- Saved the trained model for future use.
- Generated plots for training and validation accuracy, as well as loss over the epochs to visualize the model's learning curve.
- Created a confusion matrix to analyze the model's performance across different digit classes.
- Visualized the confusion matrix using a heat map to identify any misclassification patterns.
- Tested the model with individual test images and verified successful predictions.

### Summary
Over the course of five days, a CNN model was developed, trained, and evaluated to recognize handwritten digits from the MNIST dataset. The model achieved a high accuracy, demonstrating its effectiveness. Visualization tools were used extensively to monitor the model's performance and ensure its reliability. This project provided a comprehensive understanding of the workflow involved in developing a neural network for image recognition tasks.

## WEEK - 3
### Weekly Report: Next Word Prediction Using RNN

#### Day 1: Research and Planning
**Activities:**
- Conducted research on Recurrent Neural Networks (RNN) and their applications in natural language processing tasks, specifically next word prediction.
- Reviewed various literature and tutorials on using TensorFlow and Keras for building and training RNN models.
- Studied different text datasets and their suitability for the next word prediction task.
- Planned the project timeline, identifying key milestones and deliverables.

#### Day 2: Dataset Acquisition and Preliminary Exploration
**Activities:**
- Acquired the dataset from the provided Google Drive link, downloading and extracting the text data.
- Loaded the dataset into the environment and performed preliminary exploration to understand its structure and contents.
- Displayed the corpus length to get an overview of the data size.
- Identified and listed unique characters in the dataset, counting the total number of unique characters.

#### Day 3: Data Preprocessing
**Activities:**
- Created dictionaries for character-to-index and index-to-character mappings.
- Chose a sequence length of 40 characters and a step size of 3 for chunking the text.
- Generated training examples by creating overlapping sequences of 40 characters and their corresponding next characters.
- One-hot encoded the sequences and the next characters, preparing the data for model training.

#### Day 4: Model Creation and Training
**Activities:**
- Built the RNN model using TensorFlow and Keras, incorporating an LSTM layer for handling sequential data.
- Added a Dense layer with a softmax activation function to output probabilities for the next character.
- Compiled the model using the RMSprop optimizer and categorical cross-entropy loss function.
- Trained the model on the prepared dataset for 7 epochs, monitoring training and validation accuracy and loss.
- Saved the trained model and training history for future use.

#### Day 5: Model Evaluation and Prediction
**Activities:**
- Loaded the saved model and training history.
- Evaluated the model's performance on the training dataset, recording the loss and accuracy.
- Visualized the training and validation accuracy and loss over epochs using Matplotlib.
- Implemented functions for preparing input text, sampling predictions, and generating text completions.
- Tested the model's next word prediction capability on various sample quotes, observing the generated predictions.

### Summary
Over the course of five days, an RNN model was developed, trained, and evaluated to predict the next word in a sequence using the provided text dataset. The model showed promising accuracy, and various visualization tools were used to monitor its performance. Functions were implemented to generate text completions based on input sequences, demonstrating the model's practical application in next word prediction tasks.

## WEEK - 4
### Weekly Report for ML Facial Recognition and Mood-Based Song Recommendation Project

#### Day 1: Research and Planning
- **Objective**: To understand the scope and requirements of the project.
- **Activities**:
  - Conducted research on facial recognition technologies and mood detection.
  - Studied various machine learning models suitable for image classification tasks.
  - Planned the project timeline and divided tasks for the upcoming days.
- **Outcome**: Clear understanding of project requirements and a detailed plan for execution.

#### Day 2: Dataset Identification and Preparation
- **Objective**: To find a suitable dataset for training the facial recognition model.
- **Activities**:
  - Searched for publicly available facial expression datasets.
  - Found the FER2013 dataset on Kaggle which contains labeled images of various facial expressions.
  - Downloaded and organized the dataset into training and testing directories.
- **Outcome**: FER2013 dataset downloaded and organized, ready for preprocessing.

#### Day 3: Data Visualization and Preprocessing
- **Objective**: To visualize and preprocess the dataset.
- **Activities**:
  - Imported necessary libraries and packages for data visualization and preprocessing.
  - Visualized sample images from each class to understand the dataset distribution.
  - Analyzed image dimensions and computed average dimensions for resizing.
  - Implemented data augmentation techniques to enhance the training dataset.
  - Created data generators for training and testing sets.
- **Outcome**: Preprocessed dataset with data augmentation techniques applied, data generators created.

#### Day 4: Model Development and Training
- **Objective**: To develop and train the convolutional neural network (CNN) model.
- **Activities**:
  - Defined the CNN architecture using TensorFlow and Keras.
  - Compiled the model with appropriate loss function, optimizer, and metrics.
  - Trained the model using the training data generator and validated it with the test data generator.
  - Monitored the training process and adjusted hyperparameters as needed.
- **Outcome**: CNN model trained with satisfactory training and validation accuracy.

#### Day 5: Model Evaluation and Song Recommendation System
- **Objective**: To evaluate the trained model and implement the song recommendation system.
- **Activities**:
  - Evaluated the model's performance on the test set and analyzed accuracy and loss.
  - Plotted training and validation accuracy and loss curves.
  - Created a confusion matrix to visualize the model's performance across different classes.
  - Tested the model with new images and predicted moods.
  - Implemented a song recommendation system that suggests songs based on the detected mood.
- **Outcome**: Model evaluated with performance metrics, confusion matrix plotted, and song recommendation system successfully implemented.

#### Conclusion
The project successfully achieved its objectives within the planned timeline. The facial recognition model was trained and evaluated, achieving an accuracy of approximately 52.5%. The song recommendation system was integrated based on the detected mood, demonstrating the practical application of the model. Further improvements can be made by fine-tuning the model and expanding the song database for better recommendations.
