 # Project: Session-based Recommender Systems using Graph Neural Networks

This repository contains the development of a project carried out as part of the **Artificial Intelligence with Deep Learning** postgraduate program.

## Advisor

- Oscar Pina Benages

## Authors

- Joan González-Conde Cantero
- Judit Llorens Diaz
- Miguel Palos Pou
- Xavier Rodriguez Moreno

## Project Description

This project aims to explore and apply deep learning architectures for Session-Based Recommender Systems. In this repository, we provide the code to:

- Implement architectures for deep learning in session-based RecSys
- Process the needed data.
- Train, evaluate, visualize and compare the diffrenet architectures and configurations.

The implementation of these architectures uses pytorch (specifically, PyGraph), and visualizes results through TensorBoard.

[Project repository](https://github.com/joangcc-pers/aidl-upc-w20245-rec-sys)

# 1. Motivation

Businesses need tools that help predict customer needs in order to speed up conversion, reduce friction, and increase customer value through proper suggestions (Salesforce, 2021).

In today’s competitive e-commerce landscape, businesses strive to predict customer needs to accelerate conversions, reduce friction, and enhance customer value through relevant product recommendations. Traditional recommendation systems, such as collaborative filtering and matrix factorization, rely heavily on historical user data, including past purchases and interactions. While effective for registered and returning customers, these approaches struggle with two growing challenges
- Data Privacy and Regulations: Stricter [data privacy laws](https://gdpr.eu/what-is-gdpr/) limit access to user information, making it difficult to rely on long-term customer profiles for personalized recommendations.
- Sparse or Non-Existent Data for New Users: Non-registered and first-time visitors lack historical data, reducing the effectiveness of conventional recommendation methods.
This calls for the need of strong algorithms that leverage implicit feedback such as navigation history, so that both non-registered and new users can benefit from a faster navigation and purchase, and receive offers of interesting products that would fulfill their needs.

To address these challenges, we focus on session-based recommendation systems, which infer user preferences from short-term interactions rather than relying on past behavior. The usage of implicit feedback is key for current recommendation systems (see [the analysis of Esmeli et al., 2023](https://link.springer.com/article/10.1007/s42979-023-01752-x)) By analyzing user navigation within a single session, such as product views and clicks, we can dynamically predict and suggest relevant items without requiring long-term user data. This approach is particularly advantageous as it enables fast, personalized recommendations while adhering to evolving data privacy standards.

Inspired by the work of Wu et al. (2023) in Session-Based Recommendations for E-Commerce with Graph-Based Data Modeling, we adopt a graph-based framework to model session interactions. This method allows us to:

- Capture implicit relationships between products through user browsing behavior.
- Predict the next likely action or click to optimize the customer journey.
- Enhance user experience by providing relevant recommendations even for first-time visitors.
- Increase business revenue by reducing decision-making time and improving product discovery.

For our team, this project is more than an academic exercise; it is a practical response to the challenges of modern e-commerce personalization. As businesses move toward more privacy-conscious solutions, session-based recommendations offer a viable, future-proof strategy for engaging customers in real-time. Additionally, for many of us, this project represents an opportunity to deepen our expertise in recommendation systems and take meaningful steps in our professional development.

By leveraging advanced session-based models, we aim to contribute to the evolution of personalized e-commerce experiences, ensuring that both businesses and customers benefit from seamless, privacy-compliant recommendations.

# 2. Problem to solve

Apply Graph Neural Networks to predict next visited item in an e-commerce, based on session-info (navigation trhough items) and item info (brand and product category). The architecture has the objective to detect as many true positives as possible (we want to propose products users would click) in top 20 predicted products. 

# 3. Dataset

Dataset used was ["eCommerce behavior data from multi category store"](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).
It contains data from multi-category store from October 2019 to April 2020. In this project we used October and November 2019, given constraints on time and compute resources.

This data was collected by the Open CPD Project.
This file contains behavior data for a one month (November 2019) from a large multi-category online store, collected by [Open CDP (Customer Data Platform) project](https://rees46.com/en/open-cdp).

Each row in the file represents an event. An event is defined as an interaction related to products and users. Here are the columns, quoting the description of the Kaggle Dataset:
- event_time:	    Time when event happened at (in UTC).
- event_type:	    Only one kind of event: view, add_to_cart, remove_from_cart, purchase.
- product_id:	    ID of a product
- category_id:	    Product's category ID
- category_code:	Product's category taxonomy.Might be missing. Contains the info of the place of the product taxonomy that this product falls in (kitchen.appliances.microwave).
- brand:	        Downcased string of brand name. Can be missed.
- price:	        Float price of a product. Present.
- user_id:	        Permanent user ID.
- user_session:	Temporary user's session ID. Same for each user's session. It changes every time user come back to online store from a long pause.


# 4. Architectures

In our project, we learned about multiple approaches to deal with the modelling problem: how to predict the next item that a customer will click on. The cleark (or the AI in our case) will have to learn how the products they are selling relate to each other, what the user has attended during their shopping, and what items may have more importance to predict the next-viewed item based on what have browsed.
In more technical terms, our architecture must be able to know the relationship between items, how to process the sequence of interactions while maintaining a relevant context and what to remember or discard (i.e. GRU Layer), and how to weight each interaction (i.e., attention mechanism) in the final prediction (i.e., where to put the spotlight on when predicting the next-viewed item).
We have learned that we need to attend to each of these items, but we evolved oru architectures, getting them more and more sophisticated: embeddings provide the footprint of each product category and brand, GRU manages the temporal context deciding which of all the pieces of information have to be remember of past interactions while keeping the context relevant for the sequence, and the attention mechanism decides which of these interactions processed by GRU are important for the task at hand.

We want to underscore that, while developing the architectures, we made important additions to the reference paper (Wu et al., 2019), by incorporating into our architectures the product information (product category and brand) and more sophisticated attentional mechanisms. Here we give a brief definition of both GRU and the attention mechanisms, and later on how we implemented the product information.

## 4.1 Key considerations
### 4.1.1 GRU: modelling the temporal context of the session

In a session-based recommender, managing the context and the sequential information properly is key for making good predictions and capturing the nature of the customer journey. GRU stands for Gated Recurrent Unit. It is a variant from Recurrent Neural Networks (RNN), adding gates, which allows it to assess the pieces of information that GRU has to remember and forget about the past interactions (in our case, views). The information flows through the sequence of steps, updating the elements of each node (i.e., of each product), based on the messages that come from the previous element sof the custoemr journey sequence. As a result, the network ends up having **temporal memory** of the past, building the context properly.

### 4.1.2 Attention: implicit or explicit

In a customer journey, when the client is shopping, the past itneractions of the item may have different weights as to what they view next. Modelling which steps of the interaction should be attended to is a key factor in the prediction of the next viewed item. However, such mechanisms have increased computational costs and added complexity.

One could opt for implicit attention. That is, the network gives uniform weights to each of the product nodes. This is less computationally demanding, whilst hindering the ability to prioritize more relevant products. The alternative is to use explicit attention: add mechanisms to set weights for each interaction differently. In the next description of our architectures, you will see the different alternatives we used: AttentionalAggregation and self-attention based on session products.

### 4.1.3 Modelling product information: embedding vs one-hot enconding followed by a fully-connected layer

Product category and brand info are a crucial part for users to navigate and choose products of categories they are attracted to and brands the know of. There are two main ways to manage categorical information such as this: embeddings or one-hot encodings with a fully connected layer.
In early iterations, the team developed the code for one-hot encoding. However, the usage of one-hot encodings with afully conencted layer was causibgn many RAM problems, as there are multiple categories within each layer of the product taxonomy, as well as brands. Therefore, this implmentation was discarded, and continued with embeddings.


## 4.2 Architectures tested

### 4.2.1 Gated Graph Neural Network with node embeddings and Sequential Message Propagation with GRU (saved as "graph_with_embeddings")
<br>
</br>
<figure>
  <img width="813" alt="image" src="https://github.com/user-attachments/assets/7de6883f-d0e0-4d58-9eae-9f8ce427d884" />
  <figcaption><em>Figure 1. GGNN with node embeddings </em></figcaption>
</figure>
<br>
</br>

This architecture is based on a Gated Graph Neural Network (GGNN), that relies on Graph Neural Networks (GNNs) combined with GRU cells in order to work with sequential information in-session. The main characteristics of this implementation are:

- Node embeddings: the class NodeEmbedding is capable of mapping products and its featrures (category, subcategory, element, brand) to dense representations of that info. Price of the product is added as a separate tensor.
- GNN layer (GRUGraphLayer): Propagation of messages in the session graph are leveraged to update the node embedding sin each session. The GRUCell updates iteratively the states of each of the nodes, and thus the network is capable of keeping information sequentially.
- Global pooling (global_mean_pool): Node embeddings are group at a session level to consolidate the information.
- Fully connected layer: it is used to map the final representation of the session into a score for each of the products.

#### 4.2.1 Key highlights of this architecture
GRUGraphLayer inherits from Message Passing and it works through these steps:

1. Linear transformation of node features
```
node_embeddings = self.message_linear(x)
```
Transforms node embeddings to hidden dimension.

2. Message propagation
```
messages = self.propagate(edge_index, x=node_embeddings)
```
Its objective is to call the message ```message(x_j)```, with x_j representing the features of the neighbour nodes.

3. Updating states with GRU

```
node_embeddings = self.gru(messages, node_embeddings)
```
Network learns past interactions sequentially through GRU, as it updated the states of the nodes.

##### Implicit attention

This architecture implements implicit attention mechanisms. It leverages mean aggregation in propagate. The mean of the embeddings of the neighbors is calculated, setting the same weight to all of the neighbors without differentiating its importance.

In addition, the GRUCell also adds implicit attention. Based on the previous state of the node, it adjusts dinamically the importance of all the messages passed. As a result, it decides:
- Which are the parts of the message of the neighbors that will be used.
- How much it has to remember of the previous state of the node.
- It filters the information that's less relevant or less useful, causing that the more relevant and recent messages passed have more weight.

However, this architecture has a key weakness. It gives equal importance to all the viewed products. Some actions may be more telling than others, but this is something that an equally distributed importance cannot tackle. Therefore, we updated this architecture by adding a self-attention mechanism that would bring closer our model to how the shopping behavior occurs.

### 4.2.2 GGNN with Implicit Self-Attention using Sigmoid (saved as "graph-with-embeddings-and-attention")
<br>
</br>
<figure>
  <img width="1604" alt="image" src="https://github.com/user-attachments/assets/78e350e9-70bc-4a29-915f-a6a5e8b33b91" />
  <figcaption><em>Figure 2. GGNN with Implicit Self-Attention using Sigmoid</em></figcaption>
</figure>
<br>
</br>

In this architecture, we keep using GNN mean aggregation in the GRUPGraphLayer and the GRUCell adjusting dinamically the importance of the messages. However, we introduce a self-attention mechanism on the session, based on the second before last (penultimate) item. That is, we put under the spotlight (give more importance to) the last product of the session. Here are the details. 

#### Explicit attention
We adapted the architecture so it would be closer to the real world case: give more importance to the last item. However, we cannot use directly the last item, as we would have, then, no item to predict. Therefore, we reproduced Wu et al approach, and used the penultimate visited item. This means that, in this code and iscussion of architecture and results, whenever we talk about using the last visited product, we will **always refer to the penultimate item**.

The architecture for such improvement consisted of these key aspects:
- Transformed embedding calculations

We applied linear layers to node embeddings (```ìtem_embeddings_lt```) and the last visited product (```last_visited_product_embeddings_lt```) separately. The objective of this separation is to capture more sophisticated relationships than mean aggregation of neighbors.

- Interaction between products and last visited item

We sum the embeddings of the last visited product with the embeddings of the rest of the products of the session. After doing so, we use a linear layer (```attention_score_fc```) with a sigmoid function in order to calculate the attention weights of each session. By doing so, the model learns how important is each product in terms of the last visited product.

- Weight normalization through scatter_soft_max
Scatter_soft_max is used to normalize the attention weight in each session, so that attention distribution in each individual graph is coherent with the rest of the sessions of the batch.

-  Session embeddings compute

The embeddings of the products are then weighted by the attention weights ```(attention_weights * item_embeddings_lt) ```. Afterwards, these session-weighted embeddings are summed ```(scatter_sum)```, building the final representation of the session.

#### Key highlight

The introduction of self-attention let us model the fact that certain nodes or products within a session may have more importance in the final representation of the session, instead of giving them equal importance. This is an improvement, as it allows to capture more complex relationships between products.

The key limitation of this architecture though is that it prioritizes the penultimate item, but it might be that there are other browsed items or events of the session that should have more weight in the session representation. That's what we did in our third and last iteration: add Attentional Aggregation mechanism.

### 4.2.3 GGNN with Explicit Self-Attention using Attentional Aggregation (saved as "graph-with-embeddings-and-attentional-aggregation")
<br>
</br>
<figure>
  <img width="1416" alt="image" src="https://github.com/user-attachments/assets/5673f5f3-33e2-4ab1-8327-7d9dda9abf3e" />
  <figcaption><em>Figure 3. Graph with embeddings and attentional aggregation</em></figcaption>
</figure>
<br>
</br>
In this architecture, we introduce Attentional Aggregation, a mechanism that refines the way information is aggregated across nodes in the session graph. While the previous self-attention model applied attention weights based on the interaction between each product and the last visited product, this architecture further improves the aggregation process by explicitly modeling the importance of each interaction during message passing.

#### Key Differences between Attentional Aggregation and Self-Attention
The main difference between this approach and the previous explicit self-attention mechanism lies in how attention is applied:

##### Self-Attention (Previous Architecture):

The attention mechanism focused on the interaction between each product and the last visited product.
It computed attention scores for all nodes in the session relative to the last visited item, using a sigmoid activation and normalization.
The weighted sum of product embeddings created a session representation.
##### Attentional Aggregation (This Architecture):

Instead of applying attention only at the session level, this approach incorporates attention directly into the message-passing phase of the GNN.
Each node in the session graph dynamically attends to its neighbors using an attention mechanism that weighs incoming messages differently.
This allows the model to capture node-level contextual importance, ensuring that more relevant interactions contribute more effectively to the session representation.
How does Attentional Aggregation work? Here are the key elements.
- Neighborhood Interaction: During message propagation, instead of simply taking the mean of neighbor embeddings (implicit attention in the first architecture), we compute attention scores dynamically based on pairwise interactions between nodes.

- Learned Attention Weights: The importance of each neighbor is determined via a trainable attention function. This function scores interactions between nodes, allowing the model to prioritize the most relevant information during aggregation.

- Weighted Message Passing: Instead of treating all neighbors equally, the aggregation function applies attention weights to prioritize relevant past interactions.

- Session Representation Construction: After message passing, the updated node embeddings are aggregated into a session representation. The model effectively learns how much each past interaction should contribute to predicting the next interaction.

#### Key Highlights
- More flexible attention: Unlike the previous self-attention, which only adjusted the final session representation, Attentional Aggregation adapts the entire message-passing process to highlight important interactions dynamically.

- Stronger representation learning: By allowing attention-based aggregation at the node level, the model captures more fine-grained relationships between products in a session.
- Better adaptability to different session types: Some sessions may have interactions that are more sequential, while others may rely more on global patterns. Attentional Aggregation allows the model to adjust dynamically, depending on the session structure.

In more plain terms, the way in which GRU and Attentional Aggregation help each other is the following:

1. GRU processes the sequence of interactions, creating contextualized representations of each product according to the other in whihc they were seen. It maintains a memory of the past interacitons and updates the importance of the information throughour time.
2. Attentional aggregation then takes these node (product) representations generated (and updated sequentially) by the GNN with GRU cells and applies an attetion mechanism during the message aggregation process between the session graph nodes. THis means that, when deciding how to combine the representations of the neighbors in a node to update their own state, Attentional Aggregation gives more weight to the interactions tha tconsiders mor eimportant, based on learned pairwise interactions.

This approach enhances the effectiveness of session-based recommendations by ensuring that the most relevant past interactions are emphasized at every stage of the model, leading to a more contextually aware and accurate prediction process.
In sum, GRU is the mechanism reposnible to stablish the temporal context and produces informative representations of interactions between browsed items. Attentional Aggregation the come sinto play and refines the wat in which those itneractions contribute to general session representation. It focuses on the relative importance of each and eveyr one of the itneracions when doing the message propagation. As a result, the models would be then be capable of capturing more complex relationships between session products, and adapt more to different type of sessions, such as more sequential (paint supplies: first paint, then the brush) or natura or more global patterns (booking a trip: jumping around, as in preparing a holiday). 

Now that the network emphasizes the past interactions that are more relevant, then we are capable of having a more precise prediciton of next-viewed item and aware of the context of the session.

# 5. Preprocessing and training

## 5.1 Preprocessing

The preprocessing script performs the following main operations:
1. Preprocesses CSV files to generate the train / validation and test sets.
2. Precomputes graphs and stores the in an LMDB database.
3. Creates the train / validation / test splits.

### 5.1.1 Parameters

Below are the parameters expected for data preprocessing. This parametres are defined in `experiments/config.yaml` (when running `run_experiment.py`) or in `experiments/config-hyp.yaml` (when running `run-optim.py`).
- `start_month`: Start month for data processing (format: "YYYY-MM")
- `end_month`: End month for data processing (format: "YYYY-MM")
- `test_sessions_first_n`: Optional limit on number of sessions (for testing)
- `limit_to_view_event`: Whether to only include view events
- `drop_listwise_nulls`: Whether to drop rows with null values
- `min_products_per_session`: Minimum number of products per session
- `normalization_method`: Normalisation method for the price. Allows `min_max` or `zscore`. 
- `train_split`: Proportion of data for training
- `val_split`: Proportion of data for validation
- `test_split`: Proportion of data for testing
- `split_method`: "random" or "temporal"

### 5.1.2 Features
#### Data preprocessing
The data preprocessing script performs the following operations:
- Load the dataset original CSV files needed to get the data between `start_month` and `end_month`.
- Limit data to view events, if defined in `limit_to_view_event`
- Filters out product with null category_code, brand or price if defined in `drop_listwise_nulls`
- Filters out sessions with less than `min_products_per_session` unique products
- Normalises the price value using min_max or zscore 
- Defines the train, val, and test splits with the proportions defined in `train_split`, `val_split` and `test_split`, using the split method defined in `split_method`.
- Normalisea the product price values price using the method defined in `normalisation_method`.
- Handled the `category_code` into the category, sub_category and element hierarchy, managing unknown values.
- Remaps the product id to have values between 1 and the count of different product ids.
- Label-encodes categories, sub_categories, element and brand using `sklearn.processing.LabelEncoder`.
- Sorts the data by user session and event time. 
- Generates a json file with the count of different products, categories, elements and brands. This information is used to initialise the embeddings when training.

The output of the preprocessing operation is:
- `num_values_for_node_embedding.json` with the count of products, categories, elements and brands.
- `train_dataset.pth`, `test_dataset.pth` and `val_dataset.pth` dataset files.
- `graphdb` folder, with the lmdb storing all the precomputed session graphs.


#### LMDB Storage
Uses Lightning Memory-Mapped Database ([LMDB](http://www.lmdb.tech/doc/index.html)) for efficient storage and retrieval of preprocessed graphs. LMDB provides:
  - Fast access through memory 
  - Ability to handle large datasets
  - Storage of graph structures

The initial implementations did not use LMDB, and computed the session graph within the `__getitem__` method of the dataset. However, this created an important bottleneck, leading to very high training times (112 hours per epoch, when limiting the training dataset to 500k unique sessions).

To overcome this bottleneck, we decided precompute session graphs and store them: initially as `pth` files, and inside an lmdb database in a second iteration. We decided to keep working with the lmdb approach, as the file-based approach created ~5M files, which led to some performance issues. 

As a reference, using a training dataset limited to the first 500.000 unique sessions, the training time was reduced significantly, reducing per-epoch training times from 112 hours per epoch to 40 minutes per epoch.

#### Dataset Splitting
Supports two splitting methods:
- Random splitting: Randomly assigns sessions to train/val/test sets
- Temporal splitting: Splits sessions based on temporal order

## 5.2 Training
### 5.2.1 SR-GNN model

Located in `scripts/train_scripts/train_sr_gnn.py`

Trains the base SR-GNN model which uses graph neural networks for session-based recommendations.

**Key Features:**
- Basic graph neural network architecture
- Node embeddings for items and their attributes (categories, sub-categories, elements, brands)
- Support for learning rate scheduling
- Checkpoint saving and resuming
- TensorBoard integration for monitoring metrics

### 5.2.2 SR-GNN with Attention
Located in `scripts/train_scripts/train_sr_gnn_attn.py`

Trains the SR-GNN model with attention mechanism, enhancing the model's ability to focus on relevant parts of the session graph.

**Key Features:**
- Attention mechanism on top of base SR-GNN
- Same support for embeddings as base model
- Enhanced feature learning through attention
- TensorBoard monitoring and checkpoint management

### 5.2.3 SR-GNN with attentional aggregation 
Located in `scripts/train_scripts/train_sr_gnn_attn_agg.py`

Trains the SR-GNN model with attentional aggregation, providing more sophisticated ways to combine node features.

**Key Features:**
- Attentional aggregation mechanism
- Advanced feature combination strategies


## 5.3 Common Parameters

All training functions accept these common parameters:

- `model_params`: Dictionary containing model hyperparameters
  - `hidden_dim`: Dimension of hidden layers
  - `num_iterations`: Number of GNN message passing iterations
  - `embedding_dim`: Dimension of embeddings
  - `dropout_rate`: Dropout rate for regularization
  - `lr`: Learning rate
  - `weight_decay`: L2 regularization parameter
  - `batch_size`: Training batch size
  - `epochs`: Number of training epochs
  - `optimizer`: Optimizer type (currently supports "Adam")
  - `use_scheduler`: Boolean for learning rate scheduling

- `train_dataset`: PyTorch dataset for training
- `eval_dataset`: PyTorch dataset for validation
- `output_folder_artifacts`: Path to save model artifacts
- `top_k`: List of K values for evaluation metrics
- `experiment_hyp_combinat_name`: Optional name for experiment tracking
- `resume`: Flag for resuming training from checkpoint

## 5.4 The bottleneck

As explained in the preprocessing section, we experienced a bottleneck in the `__getitem__` operation. We overcome this by preprocessing the session graphs, first as files and after a second iteration, in an lmdb database, which performed better that files, especially when dealing with millions of unique sessions. 

Even after precomputing the graphs, we kept experiencing a bottleneck. This bottleneck was evident when we tried to train our model using a GPU. We saw how the GPU usage was really low, and the CPU usage close to 100%. We did some experiments, and diagnosed that the issue is still a bottleneck in __getitem__, mainly because we are loading lots of very small files at each training batch. 

![gpu_usage](https://github.com/user-attachments/assets/448df9fb-1a77-45ad-949d-93a74fe79c4a)

Because of this bottleneck, we experience lower training times with our local computers than with the cloud virtual machine using GPU. In order to be able to meet the deadlines and get results, we followded a fail-fast approach: we started with the aforementioned 500K unique visits, and then trained the best three candidates with the two months dataset.Also, we used our own computers to run the models on. This way, we were able to deliver key insights, iterate fast, and produce a first view on whether and by how much each architecture outperforms the others.

# 6. Hyperparameter tuning
## 6.1 Process

`run_optim.py` allows to perform grid search and random search hyperparameter exploration. 

We ran hyperparameter tuning for the following models:
* graph_with_embeddings
* graph_with_embeddings_and_attention
* graph_with_embeddings_and_attentional_aggregation

The code can easily be adapted to allow defining the search space as a configuration, but right now, the hyperparameter search values are defined in the code as: 

```python
weight_decay_values = [1e-4, 1e-5, 1e-6]
dropout_rate_values = [0.0, 0.2, 0.5]
learning_rate_values = [1e-3, 1e-4, 1e-5]
```

As a result, our hyperparameter tuning explored:
- 3 models (graph_with_embeddings, graph_with_embeddings_and_attention, graph_with_embeddings_and_attentional_aggregation)
- 27 different hyperparameter configurations per model:
  - 3 different weight_decay values
  - 3 different dropout_rate values
  - 3 different learning_rate values

In this case, each model was trained for 5 epochs, with the dataset limited to the first 500.000 sessions in order to limit the training time (the whole dataset for Oct 2019 and Nov 2019, after preprocessing contains ~5 million sessions). After that, we assessed the best configuration for each model against the "test" dataset.

An early conclusion we got from this grid search is that the loss can keep going down, and the metrics going up if we train during more epochs. Also, there were very promising combinations of architecture + hyperparameters that were outperforming the baseline.

Because of that, we selected the 3 best configurations of the best model and trained them for the whole dataset, during 30 epochs, with a `ReduceLROnPlateau` scheduler. The results of this extended training are the ones taken into account for the models benchmarking summary.

## 6.2 Hyperparameter tuning results: Graph with embeddings

This section shows the best and worse configurations for each model hyperparameter grid search.

The metrics displayed in the tables are always the metrics against the validation dataset.

This grid search execution is limited to 5 epochs. Having 3 models to assess, with 27 configurations each, we needed to limit the dataset length to 500.000 sessions and the train epochs to 5. 

In order to do a more accurate hyperparameter training, weare aware we would need to execute the grid search with more epochs, especially on the configurations with lower learning rate.

After the grid search, we selected the three best configurations of the overall best model, and execute them for longer, with a scheduler to adapt the learning rate. 

See below the plots for MRR20 and R20 values for the best and worst configurations for this model. We plot the best and the worst so that the comparison is easier than plotting the 27 tested combinations.

<br>
</br>
<figure>
  <img width="819" alt="image" src="https://github.com/user-attachments/assets/800bfa89-2bde-499d-af12-19219d226cf2" />
  <figcaption><em><br></br>Figure 4. Graph with embeddings: MRR@20 and R@20</em></figcaption>
</figure>
<br>
</br>

| Result | Weight Decay | Dropout Rate | Learning Rate  | R@20 | MRR@20  |
|-------|-------------|--------------|---------------|-------|-----------|
| Best setup  | 1e-06 | 0.2 | 0.001 | 0.5052400231361389 | 0.19707946479320526 |
| Worst setup | 0.0001 | 0.5 | 1e-05 | 0.22910000383853912 | 0.06784408539533615 |

All the details of the grid search result are available in the [detailed report](experiments/experiments_results/experiments_report.md#graph-with-embeddings).

## Hyperparameter tuning results: Graph with embeddings and attention

See below the plots for MRR20 and R20 values for the best and worst configurations for this model. We plot the best and the worst so that the comparison is easier than plotting the 27 tested combinations.

<br>
</br>
<figure>
  <img width="819" alt="image" src="https://github.com/user-attachments/assets/521370cf-04cf-4962-ad90-dc87a9f4f1a3" />
  <figcaption><em><br></br>Figure 5. Graph with embeddings and attention: MRR@20 and R@20</em></figcaption>
</figure>
<br>
</br>

| Result | Weight Decay | Dropout Rate | Learning Rate  | R@20 | MRR@20  |
|-------|-------------|--------------|---------------|-------|-----------|
| Best setup - R@20 | 1e-05 | 0.2 | 0.001 | 0.5530200004577637 | 0.2538684904575348 |
| Best setup - MRR@20 |  1e-05 | 0.0 | 0.001 | 0.5494800209999084 | 0.2606692910194397  |
| Worst setup | 0.0001 | 0.5 | 1e-05  | 0.24661999940872192 | 0.07098620384931564 |

All the details of the grid search result are available in the [detailed report](experiments/experiments_results/experiments_report.md#graph-with-embeddings-and-attention).

## Hyperparameter tuning results: Graph with embeddings and attentional aggregation

See below the plots for MRR20 and R20 values for the best and worst configurations for this model. We plot the best and the worst so that the comparison is easier than plotting the 27 tested combinations.

<br>
</br>
<figure>
  <img width="819" alt="image" src="https://github.com/user-attachments/assets/15758e97-7396-4426-b3d7-2c5068de5eae" />
  <figcaption><em><br></br>Figure 6. Graph with embeddings and attentional aggregation: MRR@20 and R@20</em></figcaption>
</figure>
<br>
</br>

| Result | Weight Decay | Dropout Rate | Learning Rate  | R@20 | MRR@20  |
|-------|-------------|--------------|---------------|-------|-----------|
| Best setup - R@20 | 1e-05 | 0.5 | 0.001 | 0.5768399834632874 | 0.2954697012901306 |
| Best setup - MRR@20 | 1e-06 | 0.5 | 0.001 | 0.5744199752807617 | 0.29744043946266174 |
| Worst setup | 0.0001 | 0.5 | 1e-05  | 0.31856000423431396 | 0.12075486779212952 |

All the details of the grid search result are available in the [detailed report](experiments/experiments_results/experiments_report.md#graph-with-embeddings-and-attentional-aggregation).

## Best model selection and long execution with schedulers

After the grid search, we identified our model "graph with embeddings and attentional aggregation" as the most promising model. 

We decided to pick the best 3 configurations and execute them up to 30 epochs, and against assess them against the test partition of our data. In this case, we started the execution with a learning rate of 0.001 and adapted it with a scheduler.

**Note**. The initial pick of the models was made using metrics on train partition by mistake. We report some of those odels nonetheless as they are amongst the top 10 performers in the validaiton partition.

Scheduler definition:
```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
```
### Long execution 1
Execution parametres:
```python
weight_decay=1e-06
dropout_rate=0
epochs=30
```
MRR@20 evolution

<img width="882" alt="wd_1e-06__dr_0_mrr" src="https://github.com/user-attachments/assets/92b08ec1-0a52-48da-80d8-d2125bf42958" />

R@20 evolution

<img width="882" alt="wd_1e-06__dr_0_r" src="https://github.com/user-attachments/assets/c5ed6c70-e554-44d1-be5c-45553a289946" />

Training loss vs Validation loss evolution

<img width="675" alt="wd_1e-06__dr_0_losses" src="https://github.com/user-attachments/assets/ec90e41f-5c7c-416c-9db1-6fd667bf0936" />

In this case, we can see how the model seems to overfit from epoch 10. The MRR@20 and the R@20 keep growing epoch after epoch. 

When running this trained model against the test set, we are getting the following metrics:.
- **R@20**: 0.6048
- **MRR@20**: 0.3544

### Long execution 2
Execution parametres:
```python
weight_decay=1e-05
dropout_rate=0
epochs=30
```
MRR@20 evolution

<img width="633" alt="le2_mrr" src="https://github.com/user-attachments/assets/035b1f47-4aaa-4f91-91ad-64d86a03c728" />


R@20 evolution

<img width="635" alt="le2_r" src="https://github.com/user-attachments/assets/592bdef7-ce9a-44d6-909b-8b2e41acc6da" />

Training loss vs Validation loss evolution

<img width="682" alt="le2_losses" src="https://github.com/user-attachments/assets/bb923126-7c86-4e77-ac0f-e72909f3fef1" />

In this case, we can see how a higher weight decay effectively helps preventing the model from overfitting. 

When running this trained model against the test set, we are getting the following metrics:.
- **R@20**: TODO
- **MRR@20**: TODO

# 7. Models benchmarking

The table below summarises the results benchmarking different model configurations using the same dataset (Oct 2019 and Nov 2019).


| Model | R@20 | MRR@20 |
|-------|------|--------|
| GRU4Rec (baseline) | 0.5293 | 0.2008 |
| SR_GNN (own implementation) | 0.5703 | 0.3200 |
| Graph with Embeddings and Attentional Aggregation: wd=1e-06, dropout_rate=0 | 0.6048 | 0.3544 |

<!-- TODO: add new long-runs as they finish-->

We created our own implementation of SR_GNN because the SR_GNN does not report results on the dataset we are using for the baseline and the assessment of our model.
We can see how our model "Graph with Embeddings and Attentional Aggregation (wd=1e-06, dropout_rate=0)" represents:
- A 6.04% of improvement in the Recall@20 over SR-GNN and a 14.26% over the Baseline.
- A 10.75% of improvement in the MRR@20 over SR-GNN and a 76.49% over the Baseline.

# 8. Repository structure and MLOPS features

## Repository structure

The repository is organized as follows:
- **data**: Contains the data used in the project.
    - **raw**: Stores the original, unprocessed dataset downloaded from Kaggle.
    - **processed**: Stores the data after preprocessing and feature engineering. This data is ready to be used for training and evaluation.
- **experiments**: Contains configuration files and results for different experiments. 
    - **experiment_folder**: Each experiment has its own subfolder, where the following artifacts will be stored:
        - **graphsdb**: lmdb files storing the preprocessing outcome for each experiment.
        - **train_dataset.pth**: Train dataset generated during the data preprocessing.
        - **val_dataset.pth**: Validation dataset generated during the data preprocessing.
        - **test_dataset.pth**: Test dataset generated during the data preprocessing.
        - **logs**: Tensorboard files for the training of each experiment.
        - **trained_model_000x.pth**: Model checkpoints after each training epoch.
    - **config.yaml**: Contains the configuration for preprocessing and executing each experiment using `run_experiment.py`.
    - **config_hyp.yaml**: Containst the configuration for preprocessing and executing hyperparameter tuning using `run_optim.py`
- **models**: Contains the different deep leanning models implementations.
- **notebooks**: Folder to store Jupyter notebooks that can be useful to explore different parts of this project. 
- **scripts**: The scripts folder contains all the logic needed to implement the whole lifecycle of this machine learning project. Contains `preprocessing_scripts`, `train_scripts`, `evaluate_scripts` and `test_scripts`.
- **utils**: Helper classes and methods used in by other classes in the repository.
**run_experiment.py**: Entry point for running experiments defined in `experiments/config.yaml` file.
**run_optim.py**: Entry point for running grid search defined in `experiments/config-hyp.yaml` file.


## Contributing to the repository

- Create a new branch from `develop` to work on your contribution to this repository.
- Create a merge request and assign it to at least one of the authors.
- Merge requests will be reviewed, approved and merged to `develop`and `main` by the authors.
- Do **NOT** merge to `main` or `develop` directly.

Not sure where to start? Take a look at the [open issues](https://github.com/joangcc-pers/aidl-upc-w20245-rec-sys/issues) in our repository.

## Project setup
### Install project dependencies

To install the requirements, run:
```bash
pip install -r requirements.txt
```

### Download the dataset

Download the dataset files from [kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) and place the CSV uncompressed files inside the `data/raw/` directory. Create the `raw` directory if needed. Make sure to keep the original CSV file names.

## Experiments

### Definition

Define experiments in `experiments/config.yaml`. The recommended approach is to create a new experiment using as a reference one of the already defined experiments. For instance, a valid experiment definition for the sr_gnn with attentional aggregation model is: 

```yaml
experiment_name:
    model_name: "graph_with_embeddings_and_attentional_aggregation"
    model_params:
      lr: 0.001                         # learning rate
      epochs: 30                        # number of epochs
      optimizer: "Adam" 
      num_iterations: 1                 # number of iterations within the GGNN
      embedding_dim: 64                 # embedding dimension for product date embeddings.
      hidden_dim: 100                   # hidden dimension value for the GGNN
      batch_size: 50
      shuffle: False
      weight_decay: 0.0001
      dropout_rate: 0.3
      use_scheduler: True               # If False or not defined, no scheduler is used. If True, using reduceLRonPlateau
    data_params:
      input_folder_path: "data/raw/"
      output_folder_artifacts: "experiments/experiment_name/" 
    preprocessing:
      start_month: '2019-10'            # Start month of data to consider for the dataset generation
      end_month: '2019-10'              # End month of data to consider for the dataset generation
      test_sessions_first_n: 10000      # Limit number of sessions. Remove to use the whole dataset.
      limit_to_view_event: True         # Limiting to "view" events or not.
      drop_listwise_nulls: True
      min_products_per_session: 3       # Smaller sessions will be discarded.
      normalization_method: 'zscore'
      train_split: 0.8
      val_split: 0.1
      test_split: 0.1
      split_method: 'temporal'          # Allowed values are 'temporal' and 'random'
    evaluation:
      top_k: [1, 5, 10, 20]             # top K values to take into account for metrics calculation
    test_params:                        
      best_checkpoint_path: "model_checkpoint.pth" # Path to the best evaluated model checkpoint
```

### Execution

Command to preprocess the data and run training.
```bash
python run_experiment.py --config experiments/config.yaml --experiment your_experiment_name --task preprocess train
```

Command to run the testing on the `best_checkpoint_path`:
```bash
python run_experiment.py --config experiments/config.yaml --experiment your_experiment_name --task test
```

## Grid Search

### Definition
Define the grid search scenarios in `experiments/config-hyp.yaml`.  The definition is quite similar to the experiments definintion. 

```yaml
model_graph_with_embeddings_and_attentional_aggregation:
    model_name: "graph_with_embeddings_and_attentional_aggregation"
    model_params:
      epochs: 5
      optimizer: "Adam"
      num_iterations: 1
      embedding_dim: 64
      hidden_dim: 100
      batch_size: 50
      shuffle: False
      use_scheduler: False
    data_params:
      input_folder_path: "data/raw/"
      output_folder_artifacts: "experiments/graph_with_embeddings_and_attentional_aggregation/"
    preprocessing:
      start_month: '2019-10'
      end_month: '2019-10'
      test_sessions_first_n: 500000
      limit_to_view_event: True
      drop_listwise_nulls: True
      min_products_per_session: 3
      normalization_method: 'zscore'
      train_split: 0.8
      val_split: 0.1
      test_split: 0.1
      split_method: 'temporal'
    evaluation:
      top_k: [1, 5, 10, 20]
```

### Execution
```bash
python run_optim.py --model your_model_name --task preprocess train --force_rerun_train yes/no --resume yes/no 
```

- **--force_rerun_train**
  - "no": Will check the local directory for existing runs. If a complete training has been done before (i.e., the pertinent .pth file exists), it will not rerun. New or partially calculated scenarios will be executed.
  - "all": Will execute all scenarios, regardless of whether they have been calculated before.
  - "rerun_list": Will only overwrite those scenarios in the list.
    
- **--resume** : Resume training. If "yes":
  - Will skip fully executed scenarios
  - Will resume started scenarios from the last checkpoint

# 9. Future work

We identified 3 areas of future work for this project:
- Inference service
- Hybrid models
- Model interpretability

## Inference Service

When we envision this model in production-ready setup, we think about a service that gets the different events that happen in a website, and is able to propose recommendations. 

To some extent, we think of a service collecting click events that contain the `session_id`, and the visited `product_id`. With this information, the inference service will:
- Enrich the data with category, sub_category, element, brand and price of the product (in case this information is not provided in the event).
- Generate the session graph representation.
- Forward the graph representation through the trained model to get the list of recommendations for that session.
- Expose the results so they can be used in the ecommerce site.

This will require creating another flow, that we can call inference, that will require:
- Access to data to generate the graph: product database, label encoders, min-max price values.
- The model and the parametres used to train the model.

## Hybrid models

This proposal has focused on scenarios where the information of the user is non-existant. However, in a real setting, the company can have access to previous purchass of the same user, as well as be able to asses the similarity of users through collaborative filtering and clustering techniques. Therefore, an interesting additiong would be to expand the set of algorithms we would use, to incorporate pre-computed recommended products using collaborative filtering techniques whenever possible, or most popular products, so that the user sees those whenever they enter the e-commerce page, and then update those scores dynamically as we have more information of the session with the architecture we have developed here.

Such combination of models is well-known in the industry (Burke, 2002; Chiang, 2021 for a digest). Amongst the approaches described in the paper, one approach we could explore would be the weighted approach. We could have pre-computed scores using a collaborative filtering approach and most popular items, and then weight those scores on-line with the session-based recommender system. 

## Model interpretability

Since we are using attention mechanisms, we should be able to further understand how the model works by looking at the attention weights. 

In the case of our best-performing model (graph with embeddings and attentional aggregation), that would likely mean forward data to the model and extract the attention weights of the `gate_nn` network from the attentional aggregation layer in our model (see layer definition below).

```python
self.attentionalAggregation = torch_geometric.nn.AttentionalAggregation(
  gate_nn = nn.Sequential(
    nn.Dropout(dropout_rate, inplace=True),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1)
  )
)
```


# 10. References
- Burke, R. Hybrid Recommender Systems: Survey and Experiments. User Model User-Adap Inter 2002, 12, 331–370. doi: 10.1023/A:1021240730564
- Chiang, J. 7 types of Hybrid Recommendation Systems. Medium. Retreived from https://medium.com/analytics-vidhya/7-types-of-hybrid-recommendation-system-3e4f78266ad8
- Delianidi, M.; Diamantaras, K.; Tektonidis, D.; Salampasis, M. Session-Based Recommendations for e-Commerce with Graph-Based Data Modeling. Appl. Sci. 2023, 13, 394. https://doi.org/10.3390/app13010394
- Esmeli, R.; Bader-el-Den, M.; Abdullahi, H.; Henderson, D. Implicit Feedback Awareness for Session Based Recommendation in E-Commerce. Sn. Comp. Sci. 2023, 4, 320. https://doi.org/10.1007/s42979-023-01752-x
- Salesforce (2021). Personalization in Shopping. https://www.salesforce.com/content/dam/web/en_us/www/documents/commerce-cloud/Personalization_in_Shopping.pdf
- Wu, S.; Tang, Y.; Zhu, Y.; Wang, L.; Xie, X.; Tan, T. Session-based recommendation with graph neural networks. AAAI Conf. Artif.
Intell. 2019, 33, 346–353.

# License



# Contact

For any inquiries, you can contact us at:

- Joan González-Conde Cantero: [jgonzalezconde90@gmail.com](mailto:jgonzalezconde90@gmail.com)

- Judit Llorens Diaz: [juditllorens1998@gmail.com](mailto:juditllorens1998@gmail.com)

- Miguel Palos Pou: [miguelpalospou@gmail.com](mailto:miguelpalospou@gmail.com)

- Xavier Rodriguez Moreno: [xeeevi@gmail.com](mailto:xeeevi@gmail.com)

