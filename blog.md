## Introduction

Over the past decade, there has been an industry wide sure to begin integrating machine learning into their future strategies. Whether to create a new service or enhance existing services to customers, an internal tool to improve their infrastructure, or anything in between, there is no denying the value of machine learning. It has an endless variety of applications and is not limited to any specific industry. Adapting AI/ML has become a key to innovation, and it’s important to understand what goes into making this innovation possible by examining the model lifecycle. 

The data science model life cycle demonstrates the end to end process for creating machine learning models and deploying them in production. This requires many hands on the back end and isn’t a straight forward path. In this blog, we’ll walk through the ML model life cycle with an example use case created by a few data scientists at Red Hat.

## Overview

Machine learning is a subset of the larger field called artificial intelligence and employs the use of computer algorithms to learn from historical data in order to make future predictions. In a typical workflow, a data scientist will train a machine learning model and then deploy the model for inferencing. During training, a computer algorithm is applied to a training data set that produces a machine learning model. During inferencing, a model returns predicted results when given live input data. 

A Data Science project contains data as its main element. Without any data, we are not able to do any analysis or predict any outcome as we are looking at something unknown. Hence, before starting any data science project that we have been given from either our clients or stakeholder we must first understand the underlying business problem statement. Once we understand the business problem, we can dive into the different phases of development and implementation. 

A data science life cycle is an iterative set of data science steps you take to deliver a project and conduct analysis to achieve a business outcome. Because every data science project and team are different, every specific data science life cycle is different. However, most data science projects tend to flow through the same general life cycle of data science steps. Some data science life cycles narrowly focus on just the data, modeling, and assessment steps. Others are more comprehensive and start with business understanding and end with deployment of intelligent applications into production. And the one we’ll walk through is even more extensive to include operations. It also emphasizes agility as a key characteristic across the life cycle.

This life cycle has five steps:

1. **Problem Definition** - Just like any good business or IT-focused life cycle, a good data science life cycle starts with “why”. In this phase, we state clearly the problem to be solved and why. 
2. **Data Engineering** - The next step after problem understanding is to collect the right set of data. In this phase, we gather relevant data needed to solve the problem, store the data and perform exploratory analysis to prepare the data for ML model training.
3. **Model Development** - In this phase, we build the ML algorithm to solve the problem. We train the model on the data set and evaluate the model’s performance on the unseen data points (test set).
4. **Model Deployment** - Once the stakeholders are pleased with the model’s results, the next step is to deploy the model within an intelligent application. The purpose of deploying your model is so that you can make the predictions from a trained ML model available to others, whether that be users, management, or other systems.
5. **Model Monitoring and Management** - Once the model is deployed we need to continuously monitor the model to ensure that its performance is optimized and the business objectives are not compromised.
