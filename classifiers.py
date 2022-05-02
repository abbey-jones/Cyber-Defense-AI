import weka.core.converters as converters 
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random

def build_classifier(data):
    classifier_name = "weka.classifiers.bayes.NaiveBayes"
    print(f"Training {classifier_name} classifier")
    classifier = Classifier(classname=classifier_name)
    classifier.build_classifier(data)
    # print("evaluating classifier")
    # evaluation = Evaluation(data)
    # evaluation.crossvalidate_model(classifier, data, 10, Random(42))
    # print(evaluation.summary())
    # print("pctCorrect: " + str(evaluation.percent_correct))
    # print("incorrect: " + str(evaluation.incorrect))

    return classifier
