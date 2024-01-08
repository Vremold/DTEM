import os
import sys
import json
import joblib
import pickle
import numpy as np

from sklearn.decomposition import PCA

REPO_STATISTIC_FILE = "../GHCrawler/cleaned/repo_statistics.txt"
REPO_LANGUAGE_FILE = "../GHCrawler/cleaned/repo_languages.txt"

class PCAModel():

    def __init__(self, n_components, model_file) -> None:
        if os.path.exists(model_file):
            self.pca = joblib.load(model_file)
        else:
            self.pca = PCA(n_components=n_components)
        self.model_file = model_file

    def train(self, data):
        self.pca.fit(data)
        print("Training finished, saving model to {}...".format(self.model_file))
        print("Explained variance ratio: {}".format(self.pca.explained_variance_ratio_))
        print("Explained variance: {}".format(self.pca.explained_variance_))
        print("Singular values: {}".format(self.pca.singular_values_))
        print("Mean: {}".format(self.pca.mean_))
        print("Components: {}".format(self.pca.components_))
        print("Noise variance: {}".format(self.pca.noise_variance_))
        print("n_components: {}".format(self.pca.n_components_))
        print("n_features: {}".format(self.pca.n_features_))
        print("n_samples: {}".format(self.pca.n_samples_))
        joblib.dump(self.pca, self.model_file)
    
    def transform_data(self, data):
        return self.pca.transform(data)

    def reverse_transform_data(self, data):
        return self.pca.inverse_transform(data)


class RepoDiscreteFeatureLoader():
    @staticmethod
    def load_topic_feature():
        repo_topics = {}
        topic2cnt = {}
        with open(REPO_STATISTIC_FILE, "r", encoding="utf-8") as inf:
            for line in inf:
                obj = json.loads(line)
                repo_name = obj["full_name"].lower()
                topics = obj["topics"]
                topics = [t.lower() for t in topics]
                for t in topics:
                    topic2cnt[t] = topic2cnt.get(t, 0) + 1
                repo_topics[repo_name] = topics

        # Output: 50285
        print("We've got {} kinds of topics".format(len(topic2cnt)))
        # topic2cnt = {key: val for key, val in topic2cnt.items() if val >= 3}
        # Output: 10130
        # print("If we limit the frequency bount to 3, we got {} kinds of topics".format(len(topic2cnt)))
        # topic2cnt = {key: val for key, val in topic2cnt.items() if val >= 5}
        # print("If we limit the frequency bount to 5, we got {} kinds of topics".format(len(topic2cnt)))
        topic2idx = {}
        for t in topic2cnt:
            topic2idx[t] = len(topic2idx)

        for repo in repo_topics:
            repo_vec = np.zeros(len(topic2idx), dtype=np.int32)
            for t in repo_topics[repo]:
                if t in topic2idx:
                    repo_vec[topic2idx[t]] = 1
            repo_topics[repo] = repo_vec
        
        return topic2idx, repo_topics
    
    @staticmethod
    def load_language_feature():
        lang2max = {}
        lang2idx = {}
        repo_languages = {}
        with open(REPO_LANGUAGE_FILE, "r", encoding="utf-8") as inf:
            for line in inf:
                try:
                    repo_name, langs = line.strip().split("\t")
                except:
                    print(line)
                    sys.exit(0)
                langs = json.loads(langs)
                for l in langs:
                    if l not in lang2idx:
                        lang2idx[l] = len(lang2idx)
                    lang2max[l] = max(lang2max.get(l, 0), langs[l])
                repo_languages[repo_name] = langs
        
        # Output: 393
        print("We've got {} kinds of languages".format(len(lang2idx)))
        for repo in repo_languages:
            lang_vec = np.zeros(len(lang2idx), dtype=np.float32)
            for l in repo_languages[repo]:
                lang_vec[lang2idx[l]] = repo_languages[repo][l] / lang2max[l]
            repo_languages[repo] = lang_vec
        
        return lang2idx, lang2max, repo_languages

if __name__ == "__main__":
    topic2idx, repo_topics = RepoDiscreteFeatureLoader.load_topic_feature()
    with open("./export/topic2idx.json", "w", encoding="utf-8") as outf:
        json.dump(topic2idx, outf, ensure_ascii=False)
    with open("./export/repo_topics_raw.pkl", "wb") as outf:
        pickle.dump(repo_topics, outf)
    lang2idx, lang2max, repo_languages = RepoDiscreteFeatureLoader.load_language_feature()
    with open("./export/lang2idx.json", "w", encoding="utf-8") as outf:
        json.dump(lang2idx, outf, ensure_ascii=False)
    with open("./export/lang2max.json", "w", encoding="utf-8") as outf:
        json.dump(lang2max, outf, ensure_ascii=False)
    with open("./export/repo_languages_raw.pkl", "wb") as outf:
        pickle.dump(repo_languages, outf)

    topic_pac_model = PCAModel(
        n_components=256,
        model_file="./export/topic_pca_model.bin"
    )

    repo_topics_arr = np.array(list(repo_topics.values()))
    topic_pac_model.train(repo_topics_arr)
    repo_topics_arr_pca = topic_pac_model.transform_data(repo_topics_arr)
    repo_topics_pca = {}
    idx = 0
    for key in repo_topics:
        repo_topics_pca[key] = repo_topics_arr_pca[idx]
        idx += 1
    with open("./export/repo_topics_pca.pkl", "wb") as outf:
        pickle.dump(repo_topics_pca, outf)

    language_pac_model = PCAModel(
        n_components=256,
        model_file="./export/language_pca_model.bin"
    )
    repo_languages_arr = np.array(list(repo_languages.values()))
    language_pac_model.train(repo_languages_arr)
    repo_languages_arr_pca = language_pac_model.transform_data(repo_languages_arr)
    repo_languages_pca = {}
    idx = 0
    for key in repo_languages:
        repo_languages_pca[key] = repo_languages_arr_pca[idx]
        idx += 1
    with open("./export/repo_languages_pca.pkl", "wb") as outf:
        pickle.dump(repo_languages_pca, outf)