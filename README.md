# Stock-Market-Price-Prediction-App

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [System Overview](#system-overview)
  - [Terms used for Stock-Market-Price-Prediction-App](#terms-used-for-Stock-Market-Price-Prediction-App)
- [System Architecture](#system-architecture)
- [Application Overview](#application-overview)
- [Application Architecture](#application-architecture)
  - [Flask App](#flask-app)
  - [RESTful API](#restful-api)
  - [CRUD Operations](#crud-operations)
  - [External APIs Used](#external-apis-used)
- [Cloud Infrastructure](#cloud-infrastructure)
  - [Google Cloud](#google-cloud)
  - [Kubernetes](#kubernetes)
  - [Docker](#docker)

## About

This application leverages machine learning algorithms to predict stock market prices, providing users with data-driven insights into potential market trends. By analysing historical price data and market indicators, the app aims to assist traders and investors in making informed decisions based on predicted price movements. We will be utilising **Google's Finance API** in order to fetch real-time and historical stock data - giving us confidence that we have up-to-date and accurate information to do our analysis/predictions.

## System Architecture

![Screenshot 2024-11-19 at 11 18 35](https://github.com/user-attachments/assets/f8960847-e587-4efc-91b0-4c27fac68059)


## Application Architecture

## Flask App
Flask is the efficient way to be used as Python web framework that is innovated to produce a web development in an easier and more smoother way. It is very popular method that most of the developers use to simplify their flexibility in order to come up with an easier way to produce a website developed in Python. This is progressed through the advantages that Flask has when it collaborates with Python, such as it contributes to minimal setup, it only require a single Python file to start with, and then the development could be progressed after this step. It is also a reliable initial point to be used for beginners as well as this who are looking to develop small to medium level of applications without considering a step learning curve. 

One of the biggest strength that flask has is the adaptability, unlike other frameworks, as it does not enforce a specific project structure or force developers to adhere to strict regulations. This provides a better way to cause freedom to organise the appication in whatever way meets the developer's requirments. As well as this, Flask also helps in the integration with Jinja, that is considered to be a powerful templating engine, which allows dynamic generation of HTML content. In comparison with static HTML which only display a fixed content. This leads to the prevention of building moderns, and data-driven applications where content is constantly changing based on the backend requirements. 

Through the consideration to the above, developers can easily insert Python as syntax into HTML files, such as variables, loops, and conditionals, which would make it simple to generate dynamic web pages based on the data, or the change of any external factors. This is a strong key that makes Flask very rapid for web application development. 

## RESTFUL-API
A REST API (Representational State Transfer Application Programming Interface), which is an architectural style that is used to create a networked applications for the purpose of collaboration. This is progressed by allowing different software components to communicate with each other over the web standard protocol HTTP (HyperText Transfer Protocol) methods such as GET, POST, PUT, DELETE, etc. 

In 2000, the computer scientist Roy Fielding has completed his doctoral dissertation titled "Architectural Styles and the Design of Network-based Software Architectures", which after this completion the term "REST" began. In this dissertation the article outlined some guidance which are used to increase the level of concentration on the design of web services. The first point is "statelessness" which means that in order for a server to respond to the client's request, the request must contain all the needed information the server needs to process the request, and the server does not store any information of previous requests asked by clients. The second point is the "Client-Server Architecture", this contains the segregation between the UI (User Interface) and the server, each of the ends has a specific task, as the UI manages the user experience and the server manages the data and the business logic behind the UI (Backend). 

Subsequently, the "uniform Interface" which is the collaboration between the UI with the API without needing to understand the underlying infrastructure. This is progressed by the utilisation of uniform set of regulations to help in the resources accessibility. The next term is the "cacheability", at each request being progressed at each time the process to deal with this request takes long to output the needed result, therefore at this process the server maintains the request by caching it and if this request been requested again by the user then it will be faster to procced than the first attempt. The last point is the "Layered System", which is the segregation of the layers considering each layer is responsible for a certain task, without the need for the client to know if there is interaction with other layers such as the database layer.


## CRUD Operations 

CRUD stands for Create, Read, Update and Delete, which are the four basic operations that are used to help in the interaction with the resources in the system. These operations are fundamental for managing data, and they used in the REST APIs for web services. These operations are considered to be HTTP methods by the correspondence to these operations. The first HTTP method is "GET", which is used to retrieve data, this is progressed by fetching the representation of the source from a specific URL in particularly (URI). However, this won't cause any changes to the server, it will only read the data. For example assuming that we need to retrieve the user's profile, then the following command would be GET/users/userID, which will return the user ID "userID". The next method is the "POST" HTTP method, which is used to innovate a new data or resources on the server. This is progressed by automatically assigning a unique URI to the most recent innovate resource and retrieves it. This method could be used in order to innovate new records in the database, or creating a new user profile. For example, POST/USERS which the return would be the URI of users/userID. The third method is the "PUT" method, which is used to update an existing resource to create a new resource at a specific URI (assuming that the server allows it). For example, PUT/users/userID which will modify the user with the "userID" with the new data being requested in the request. Lastly, is the "DELETE" HTTP method, which is used to remove an existing resource from the server at a specific URI. For example, DELETE/users/userID, this command would remove "userID" from the database. 

## Simplified terms: 
GET is for reading (retrieving) data,
POST is for creating new data,
PUT is for updating existing data,
DELETE is for removing data


## Method Responses
Based on the above HTTP methods we receive different types of responses to check whether the resposes went successsfully or not. These responses are considered to be HTTP status codes which conveys whether the operation of the request is preformed or not, and based on these responses different actions are taken.

## 200 OK:
This command determines that the request went successfuly.
## 201 Created:
The resource was successfully created. 
## 204 No Content:
The request was successful, but there is no content to return.
##  400 Bad Request: 
The request was malformed or missing required data.
## 404 Not Found: 
The resource was not found.
## 405 Method Not Allowed:
The HTTP method used is not supported for the requested resource.
## 500 Internal Server Error:
A general server error occurred (something went wrong on the server side).
## 401 Unauthorized: 
The request requires authentication.
## 403 Forbidden: 
The server understands the request but refuses to authorize it.




## External API User

An API (Application Programming Interface) is a set of rules and tools that allow different software applications to communicate with each other. It acts like a contract between the provider (who offers the data or service) and the consumer (who uses the data or service). The API defines what information the consumer needs to send (the request) and what kind of response the provider will give back.


## Cloud Architecture

![Screenshot 2024-11-28 at 22 37 24](https://github.com/user-attachments/assets/6cd9e58a-c665-4186-99d9-99de415926df)

## Cloud Infrastructure

Cloud computing infrastructure consists of the essential hardware and software components that enable cloud services. It provides on-demand access to computing resources such as servers, storage, networking, and virtualization. Cloud Infrastructure is attractive because acquiring computing resources to run applications or store data the traditional way requires time and capital. Its virtualization technology allows multiple virtual machines to run on one physical server. User interfaces such as dashboards and APIs enable users to manage cloud resources, while management software automates and monitors the infrastructure. Together, these components deliver scalable, flexible, and on-demand access to computing resources over the internet.

**Why Cloud Computing Infrastructure?**
Cloud infrastructure delivers the same functionalities as physical infrastructure, with added benefits like reduced ownership costs, increased flexibility, and better scalability.
It is available for private, public, and hybrid cloud environments, and businesses can also rent infrastructure components through Infrastructure as a Service (IaaS) from a cloud provider. Cloud providers offer backup and disaster recovery features. Storing data in the cloud rather than locally can help prevent data loss in the event of an emergency, such as hardware malfunction, malicious threats, or even simple user error. The distribution of resources provides several benefits, including redundancy in case of failure and reduced latency by locating resources closer to clients.  


## Google Cloud Firestore

Firestore, also known as Google Cloud Firestore, is a cloud-based NoSQL database designed for developers to build scalable, high-performance applications. It organizes data into collections and documents, using a JSON-like structure that allows for nested data and subcollections.

One of Firestore's standout features is real-time synchronization, where changes made to the database are instantly reflected across all connected devices, making it ideal for collaborative and interactive apps such as chat systems or live dashboards. Additionally, it supports offline data access, enabling users to read and write data locally, with changes automatically synced to the cloud once the device reconnects.

Firestore is highly scalable, capable of handling small projects to enterprise-level applications. It provides powerful querying capabilities, automatic indexing, and built-in security through Firebase Authentication and customizable rules. Firestore integrates seamlessly with Firebase and Google Cloud services, making it suitable for web, mobile, and IoT applications requiring speed, reliability, and flexibility.

## Firebase

Firebase is a comprehensive platform by Google designed to simplify the development and management of web, mobile, and serverless applications. It provides tools for real-time data synchronization, authentication, cloud storage, hosting, and serverless backend solutions with Cloud Functions. Developers can use its NoSQL databases, Realtime Database, and Cloud Firestore to store and sync data efficiently.

Firebase also offers analytics, crash reporting, performance monitoring, and Firebase Cloud Messaging for targeted push notifications. With integrated machine learning tools, it supports ready-to-use and custom models for tasks like image recognition or text translation. Its scalability, cross-platform compatibility, and serverless architecture make it a popular choice for building dynamic and engaging applications quickly.



## Docker

Docker is an open platform that simplifies the development, delivery, and execution of applications by providing a consistent environment for developers. It streamlines the process of shipping, testing, and deploying code by isolating applications from the underlying infrastructure. This isolation not only accelerates software delivery but also reduces the time it takes to move from development to production deployment. Docker achieves this by packaging applications and their dependencies into containers, which run in isolated environments. These containers can operate independently, allowing multiple containers to run concurrently on the same host without conflicts.

Docker empowers developers to create consistent environments across all stages of application development, from building and testing to deployment. It is especially beneficial for running microservices, automating CI/CD (Continuous Integration/Continuous Deployment) pipelines, and managing infrastructure as code, which are essential practices for modern development workflows. Docker’s containerization helps in consolidating multiple applications on a single host, optimizing resource utilization and enabling cross-platform development, as containers can run on different systems without modification.

In cloud environments, Docker is ideal for quickly prototyping and scaling applications. Its lightweight nature and portability allow for rapid deployment and scaling across various cloud platforms. Docker also supports automation, making it easier to manage complex workflows, and accelerates development cycles by allowing teams to work in parallel. Its scalability and flexibility make it an essential tool for modern software development, particularly in environments where speed, consistency, and efficiency are crucial.

## Kubernetes

Kubernetes is an open-source, extensible platform that manages containerized workloads and services. It simplifies the deployment, scaling, and operation of applications using declarative configuration and automation. With a rapidly expanding ecosystem, Kubernetes offers numerous tools, services, and robust support for modern cloud-native environments.

Kubernetes coordinates containerized applications across clusters of machines, managing their lifecycle from deployment to scaling and monitoring. It ensures scalability, high availability, and predictable behavior, making it an ideal choice for both simple and complex applications. Kubernetes abstracts the underlying infrastructure, allowing users to run scalable workloads efficiently and securely.

Though Kubernetes' architecture may seem complex at first, its flexibility and powerful features provide unmatched control for managing distributed applications. It is widely recognized for its self-healing capabilities, rolling updates, and automated load balancing, all of which contribute to high reliability and performance. To deploy an application on Google Kubernetes Engine (GKE), you must first create a Kubernetes cluster in the Google Cloud Platform (GCP) console.
