import weka.core.converters as converters 
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random

def naive_bayes(data):
    classifier = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
    evaluation = Evaluation(data)
    evaluation.crossvalidate_model(classifier, data, 10, Random(42))
    print(evaluation.summary())
    print("pctCorrect: " + str(evaluation.percent_correct))
    print("incorrect: " + str(evaluation.incorrect))
