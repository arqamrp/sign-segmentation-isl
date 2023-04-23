# Sign Segmentation for ISL
ISL is the variety of sign language used most commonly in India. It is an independent language with its own grammar, syntax and vocabulary. It is not a mapping of a spoken language like Hindi. There are at least 6 million users of ISL, and it is one of the most used sign languages in the world. Despite this, there are very few resources for automated processing of ISL.

In this project, we investigate different methods to train models for segmenting signs in Indian Sign Language.

## Data
For the training set, we used YouTube videos from Deaf Enabled Foundation, which releases daily videos on “Word of the Day” , which also include continuously signed sentences describing the meaning of the chosen word. We scraped 722 videos in total.The links to our dataset are given in ```/train_data/readme.MD```.   
Deaf Enabled Foundation YouTube Channel: https://www.youtube.com/@deafenabledfoundation1117. 


For the test set, we used the ISL CSLTR Dataset, which contains multiple videos of 100 sentences in ISL. We annotated two videos for each sentence manually using VIA Annotation Tool. The annotated dataset is available in ```/test_data/VIA_annotations.csv```.    
ISL CSLTR Dataset: https://data.mendeley.com/datasets/kcmpdxky7p/1
VIA Annotation Tool: https://www.robots.ox.ac.uk/~vgg/software/via/

## Model
 We used the TCN + i3D model, and trained it on ISL data using Pseudolabelling and Changepoints. The details of the models are given in the following research papers:
 - [Katrin Renz](https://www.katrinrenz.de), [Nicolaj C. Stache](https://www.hs-heilbronn.de/nicolaj.stache), [Samuel Albanie](https://www.robots.ox.ac.uk/~albanie/) and [Gül Varol](https://www.robots.ox.ac.uk/~gul),
*Sign language segmentation with temporal convolutional networks*, ICASSP 2021.  [[arXiv]](https://arxiv.org/abs/2011.12986)

- [Katrin Renz](https://www.katrinrenz.de), [Nicolaj C. Stache](https://www.hs-heilbronn.de/nicolaj.stache), [Neil Fox](https://www.ucl.ac.uk/dcal/people/research-staff/neil-fox), [Gül Varol](https://www.robots.ox.ac.uk/~gul) and [Samuel Albanie](https://www.robots.ox.ac.uk/~albanie/),
*Sign Segmentation with Changepoint-Modulated Pseudo-Labelling*, CVPRW 2021. [[arXiv]](https://arxiv.org/abs/2104.13817)

The models are stored in ```/models/mstcn_ISL_CMPL.zip``` and the Jupyter Notebooks for our code are stored in ```/notebooks``
