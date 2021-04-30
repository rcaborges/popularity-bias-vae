# On Mitigating Popularity Bias in Recommendations via Variational Autoencoders

Rodrigo Borges and Kostas Stefanidis

SAC'21, March 22–26, 2021, Virtual Event, Republic of Korea



Variational autoencoders for collaborative filtering is a framework for making recommendations. 

>  "Variational autoencoders for collaborative filtering" by Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara, in The Web Conference (aka WWW) 2018.

The code presented here was adapted from [this one](https://github.com/cydonia999/variational-autoencoders-for-collaborative-filtering-pytorch).


> @inproceedings{10.1145/3412841.3442123,
> author = {Borges, Rodrigo and Stefanidis, Kostas},
> title = {On Mitigating Popularity Bias in Recommendations via Variational Autoencoders},
> year = {2021},
> isbn = {9781450381048},
> publisher = {Association for Computing Machinery},
> address = {New York, NY, USA},
> url = {https://doi.org/10.1145/3412841.3442123},
> doi = {10.1145/3412841.3442123},
> abstract = {Recommender systems are usually susceptible to Popularity Bias, in the sense that their training procedures have major influence of popular items. This promotes a long term unfairness: models end up adjusted to common options, associated to few popular items, and with restricted knowledge about unpopular ones. We propose a method that penalizes scores given to items according to historical popularity for mitigating the bias and promoting diversity in the results. A parameter is available for controlling the weight of the penalty, to be decided according to the necessity in the application. Our method is based on Variational Autoencoders, considered today as the state-of-the-art method for the task of Collaborative Filtering, and the price for removing the bias is paid by having its accuracy and ranking metrics reduced. We managed to reduce the bias by 8% while reducing accuracy in 3% in a movie consumption dataset.},
> booktitle = {Proceedings of the 36th Annual ACM Symposium on Applied Computing},
> pages = {1383–1389},
> numpages = {7},
> keywords = {recommendations, popularity bias, variational autoencoders, bias},
> location = {Virtual Event, Republic of Korea},
> series = {SAC '21}
> }
