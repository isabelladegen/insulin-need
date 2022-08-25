import pandas as pd
import graphviz
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text


class DecisionTreeRuleExtraction:
    def __init__(self, x_train: pd.DataFrame, y):
        """Collection of convenience function for sklearn DecisionTreeClassifier

                Parameters
                ----------
                x_train : pd.DataFrame
                    timeseries to cluster as np.array of shape=(n_ts, n_features), where n_ts is number of days as rows
                    and n_features is the features to consider as columns e.g hours of day, year, month, weekday

               y: []
                    class prediction for each of the n_ts
        """
        self.__x_train = x_train
        self.model = DecisionTreeClassifier(criterion='gini',
                                            splitter='best',
                                            max_depth=None,
                                            min_samples_split=2,
                                            min_samples_leaf=1,
                                            min_weight_fraction_leaf=0.0,
                                            max_features=None,
                                            random_state=66,
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            class_weight="balanced",
                                            # The “balanced” mode uses the values of y to automatically adjust weights
                                            # inversely proportional to class frequencies in the input data as n_samples
                                            # / (n_classes * np.bincount(y))
                                            ccp_alpha=0.0)
        self.model.fit(x_train, y)

    def tree_rules(self):
        """Returns rules as tree decisions

        """
        rules = export_text(self.model, feature_names=self.__get_list_of_features())
        print(rules)
        return rules

    def plot_tree(self):
        """Plots tree

        """
        dot_data = tree.export_graphviz(self.model,
                                        feature_names=self.__get_list_of_features(),
                                        class_names=[str(x) for x in self.model.classes_],
                                        filled=True,
                                        rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        return graph

    def plot_feature_importance(self, sns=None):
        """Plots feature importance

        """
        plt.rcParams.update({'figure.facecolor': 'white'})
        plt.figure(figsize=(10, 5), dpi=150)
        feature_imp = self.model.feature_importances_
        features = self.__get_list_of_features()
        pd.DataFrame({'Feature': features, 'Feature Importance': feature_imp}).plot.bar(x='Feature',
                                                                                        y='Feature Importance')

    def __get_list_of_features(self):
        return list(self.__x_train.columns)
