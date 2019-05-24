# FakeNewsPhilippines
A quick machine learning web - application that can discriminate between credible and not credible news. The models were based on the paper *"Computing the Linguistic-Based Cues of Fake News in the Philippines Towards its Detection"* and had been trained on the **Philippine Fake News Corpus**. This project is just a proof-of-concept to show that the difference in word, sentence, punctuation, and part-of-speech usage as well as readability scores can be effective in discrimininating the veracity of a news article towards detection of fake news.

This repository contains the trained models but does not include how those were trained. Also, the models underwent a feature selection algorithm, specifically recursive feature elimination, to select only the best performing feature set.

This project is currently live at: https://fakenewsphilippines.herokuapp.com/

The user can just provide that URL of the news to be classified or key in the news headlines and/or news content manually. The models in this project can classify using both the headlines and news content or either of the two alone.

![Image](https://github.com/aaroncarlfernandez/FakeNewsPhilippines/blob/master/images/landing-page.png)
![Image](https://github.com/aaroncarlfernandez/FakeNewsPhilippines/blob/master/images/results_sample.png)


