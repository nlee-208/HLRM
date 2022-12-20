# Hierarchical Latent Relation Modeling for Collaborative Metric Learning
Code Implementation of V-A. Tran, et al. *Hierarchical Latent Relation Modeling for Collaborative Metric Learning*. In: Proceedings of the 15th ACM Conference on Recommender Systems (RecSys 2021), September 2021.

[Paper Link](https://arxiv.org/pdf/2108.04655.pdf)
<br>

[Paper Summary(Korean)](https://nlee208.notion.site/Hierarchical-Latent-Relation-Modeling-for-CML-22551b6d72ac43ba8ccfc5e1b7bb838c)
<br>

## Brief Summary
### Collaborative Metric Learning

- Previous recommendation approaches focused heavily on Collaborative Filtering(CF) methods
    - Matrix Factorization (MF)
        - Factorization of the user-item matrix into dense, lower dimensional latent vectors
        - Prediction based on User-Item Similarity (usually dot-product)
- Metric Learning introduced as an alternative to previous approaches.
    - Interested in learning the metrics among data points
- Limitations
  - Over-simplification of the user-item relations
  - Each user&item represented with a single mapped vector
  - Does not incorporate any item-item relations
<br>

## Hierarchical Latent Relation Modeling for CML
- CML DL Based approach to learn hierarchical relations of user-item over item-item relations based on the following assumption...
  > ***"there exists a hierarchical structure in different relation types, and that user-item relations are built on top of item-item relations"***

### Architecture
---
![image](https://user-images.githubusercontent.com/61938580/208623457-f2138bd5-bf25-40f8-a08d-5c4d2f5cd015.png)
![image](https://user-images.githubusercontent.com/61938580/208623622-596d9228-9378-4101-91f8-0451d0d3454c.png)

- The author proposes an enhanced consideration over the user-item relations with the User Attention & Item Attention Module
- For detailed descriptions of the model implementation & model experiments, refer to the [Paper Summary(Korean)](https://www.notion.so/Hierarchical-Latent-Relation-Modeling-for-CML-22551b6d72ac43ba8ccfc5e1b7bb838c)

<br>

## Configurations
### Train Phase



### Eval Phase

