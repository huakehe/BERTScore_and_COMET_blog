# MT BLOG -- BERTScore and COMET
by Huake He, 11/01/2021

Reading Time: About 15 minutes.

COMET Paper: https://aclanthology.org/2020.emnlp-main.213.pdf

COMET Github:  https://github.com/Unbabel/COMET

BERTScore Paper: https://openreview.net/pdf?id=SkeHuCVFDr

BERTScore Github: https://github.com/Tiiiger/bert_score

## A brief history of MT evaluation metrics
### Human evaluation
In 1966, the Automatic Language Processing Advisory Committee (ALPAC) conducted a large scale evaluation study on evaluation Machine Translation (MT) systems from Russian to English. Even though the ALPAC study was infamous for concluding MT is hopless and suspending the research into related fields for two decodes, it indeed developed a practical method for evaluation of translations. Six trained translators each evaluated 144 sentences from 4 passages. The evaluation was based on "intelligibility" and "fidelity". "Intelligibility" measures to what extent the sentence can be understood, and "fidelity" measures how much information the translated sentence retained compared to the original. Human evaluation were based on the two variables, by giving a score on the scale of 1-9. This is one of the earlest practical MT evaluation metrics based on human judgement.
<p align="center">
  <img width="400" src="ALPAC.png">
</p>

### Automatic evaluation
Even though human judgement measuring metrics has evolved throught the years, purely depending on human evaluation is expensive as well as slow in face of large scale data, which promoted the need for automation. In 2002, the most commonly used evaluation metric, Bilingual Evaluation Understudy (BLEU), was developed by Kishore et al. BLEU measures the difference between human and machine translation output through n-grams and brevity penalty. Based on the “highest correlation with monolingual human judgements” found to be four, n-grams measure the exact word segment correspondence of length 1,2,3,4 in the source and target sentence pair. The brevity penalty is included to avoid short candidates receiving unreasonable high BLEU scores. BLEU remains popular till today due to its light-weightedness and fastness.

## BERTScore
Recent works on machine translation quality evaluation techniques have provided stronger metrics and support to the prospering machine translation realm of research. BERTScore, which appeared in the 2020 International Conference on Learning Representations, aims to develop “an automatic evaluation metric for text generation.” As a high level summary, BERTScore develops one important step forward from the commonly used BLEU, which is to incorporate the additional contextual information into consideration to calculate the degree of difference between source and target sentence. 

### Motivation
“an increased research interest in neural methods for training MT models and systems has resulted in a recent, dramatic improvement in MT quality, MT evaluation has fallen behind”

Generally speaking, there are two drawbacks in the n-gram-based metrics. Firstly, semantically-correct translations or paraphrases are excessively penalized for the n-gram metrics. In other words, different usage of words on the surface level will result in a low BLEU score. In the paper, the authors give the example of the source reference sentence “people like foreign cars,” and two of the candidates are “people like visiting places abroad” and “consumers prefer imported cars.” The latter uses synonyms to replace certain words in the reference, while preserving the semantic meanings. However, n-gram-based metrics like BLEU will give higher score to the former candidate, even though the meaning is far from that of the reference sentence,  since the exact string matching of unigram and bigram values are higher. In face of this pitfall, BERTScore takes advantage of contextualized token embedding as matching metrics, which considers the similarities of all of the words in the reference and candidate, and has been proven to be more effective in paraphrase detection. Secondly, n-gram metric cannot capture semantic dependencies of distant words or penalize semantically-critical order changes. For example, for short sentences, BLEU is able to capture swap of cause and effect clauses, like “A results in B”, but when A and B are long phrases, even the longest four-gram will fail to capture the semantic dependencies and critical word orders, and thus measures the similarity in a shallow way. The trained contextual embedding in BERTScore is more effective in tackling the distant dependencies and ordering problems.

### Technique
The workflow of BERTScore calculation is illustrated in Figure x. Having a reference sentence x = (x1, …, xk) and a candidate sentence x prime = (...), the technique transforms the tokens into contextual imbeddings, and compute the match among all takens by cosine similarity, and as an option, adding an additional weight based on the inverse document frequency of matching words. 

Figure x


BERTScore uses the BERT model to generate contextual embeddings for each token. BERT tokenizes the input text into a sequence of word pieces, and splits the unknown words into commonly observed sequences of characters. The Transformer encoder computes the representation for each word piece by repeatedly applying self-attention and nonlinear transformation alternatively. The resulting contextual embedding from word piece will generate different vector representation for the same word piece in different contexts with regard to surrounding words, which is significantly different from the exact string match metric in BLEU. 

Due to the vector representation of word embedding, BERTScore is able to perform a soft measure of similarity compared to exact-string matching in BLEU. The cosine similarity of a reference token xi and a candidate token xj prime is :
[equation]

With similarity measurement of each pair of reference token and candidate token in preparation, we can move on to calculating precision and recall. In the greedy match perspective, we match each token in x with the highest similarity score in x prime, and recall is computed by matching each token in x to a token in x prime, while precision is by matching each token in x prime to the corresponding token in x prime. F1 score is calculated by: . Extensive experiments indicated that F1 score performs reliably well across different settings, and therefore is the most recommended score to be used for evaluation.
[equation]
[possibly an example]

Optionally, we can add an importance weighting to different words to optimize the metric, because previous works indicated that “are words can be more indicative for sentence similarity than common words” [cite]. From experiments, apply idf-based weight can render small benefits in some scenarios, but have limited contribution in other cases, and  The authors use the inverse document frequency (idf) scores to assign higher weights to rare words. (need to expand??)

### Effectiveness
For evaluation of BERTScore, this blog will focus on the machine translation tasks in the original paper. The experiment’s main evaluation corpus is the WMT18 metric evaluation dataset, containing predictions of 149 translation systems across 14 language pairs, gold references, and two types of human judgment scores. “Segment-level human judgments assign a score to each reference-candidate pair. System-level human judgments associate each system with a single score based on all pairs in the test set.”

Table x demonstrates the system-level correlation to human judgements. The higher the score is, the closer the system evaluation is to human evaluation. Focusing on FBERT score (F1 score), we can see a large number of bold correlations of metrics for FBERT, indicating it is the top performance system compared to the others. 



Apart from system-level correlation, In Table x illustrating the segment-level correlations, BERTScore shows a considerably higher performance compared to the others. The outperformance in segment-level correlations further exhibits the quality of BERTScore for sentence level evaluation.


## COMET
In the same year, Rei et al. presented “a neural framework for training multilingual machine translation evaluation models which obtains new state-of-the-art levels of correlation with human judgements” at the 2020 Conference of Empirical Methods in Natural Language Processing. The system, COMET, employs a different approach in improving evaluation metric by building an additional regression model to exploit information from source, hypothesis, and reference embeddings, and training the model to give a prediction on the quality of translation. 

### Motivation
“The MT research community still relies largely on outdated metrics and no new, widely-adopted standard has emerged”. This creates motivation for a metric scheme that uses a network model to actually learn and predict how well a machine translation will be in a human rating perspective. We knew that BLEU transformed MT quality evaluation from human rating to automated script, BERTScore improved the evaluation scheme by considering context, while COMET is motivated to estimate how human will evaluate the quality of the translation, specifically scores from direct assessment (DA), human-mediated translation edit rate (HTER), and metrics compliant with multidimensional quality metric framework (MQM). After all, humans are the best to evaluate the translation quality of our own language. To sum up, COMET aims at closing the gap between automated rule-based metric with human evaluation.

### Technique
The first step of COMET score calculation is to encode the source, MT hypothesis, and reference sentence into token embeddings. The authors take advantage of a pretrained, cross-lingual encoder model, XLM_RoBERTa, to generate the three sequences (src, hyp, ref) into token embeddings. For each input sequence x = [x0, …, xn], the encoder will produce an embedding ejl for each token xj and each layer l {9, …, k}. 

The word embeddings from the last layer of the encoders are fed into a pooling layer. Using a layer-wise attention mechanism, the information from the most important encoder layers are pooled into a single embedding for each token ej. 
After applying an average pooling to the resulting word embeddings, a sentence embedding can be concatenated into a single vector from segments. The process is repeated three times for source, hypothesis, and reference sequences. Two models with different usage, the Estimator model and the Translation Ranking model,  take the sentence embedding sas input.

For the Estimator model, a single vector x is computed from the three sentence embeddings s, h, and r, by : x = [h; r; h ⊙ s;h ⊙ r;|h − s|;|h − r|]. The combined feature x serves as input to a feed-forward regression network. The network is trained to minimize the mean squared error loss between its predicted scores and human quality assessment scores (DA, HTER or MQM).

The Translation Ranking model on the other hand, has different inputs {s,h+,h-,r}, i.e. a source,, higher-ranked hypothesis h+, a lower-ranked hypothesis h-, and reference. After transforming them into sentence embeddings bold {s,h+,h-,r}, the triplet margin loss in relation to the source and reference is calculated:

d(u, v) denotes the euclidean distance between u and v and ε is a margin.
In the inference stage, the model will receive a triplet input (s,h prime,r) with only one hypothesis, and the quality score will be the harmonic mean between the distance to the source d(s,h prime) and that to the reference d(r,h prime), and normalized it to a 0 to 1 range:

In short, the Translation Ranking model is trained to minimize the distance between a “better” hypothesis and both its corresponding reference and its original source. 

 

### Effectiveness
To test the effectiveness of COMET, the authors trained 3 MT translations models that target different types of human judgment (DA, HTER, and MQM) from the corresponding datasets: the QT21 corpus, the WMT DARR corpus, and the MQM corpus. Two Estimator models and one Translation Ranking model are trained. One regressed on HTER (COMET-HTER) is trained with the QT21 corpus, and another model regressed on MQM (COMET-MQM) is trained with the MQM corpus. COMET-RANK is trained with the WMT DARR corpus. The evaluation method employed is the official Kendall’s Tau-like formulation: 


As shown in table x1, for seven in eight language pair evaluation with English as source, COMET-RANK outperforms all other evaluation systems to a significant extent, including BLERU, two encoder models of BERTScore, and its two Estimator models. Similarly, for the language pair evaluation with English as target, COMET also exceeds the other metrics in performance. 

## Case Study
In order to evaluate how well BLEU, BERTScore, and COMET can evaluate on existing MT systems, I tried to find the translated data with human judgment scores (e.g DA). Unfortunately, the MT system is not available to the public, e.g. I cannot access the Baidu-system.6940 with the highest DA score in WMT19. With this preliminary, the experiment to compare how our evaluation metrics scores with a human judgement score is unattainable. Another simpler case study for the metrics is initialized instead.
For the setup, a group of 10 source-reference sentence pairs were prepared from a Chinese-English parallel corpus XX, as illustrated in Figure x1. The source Chinese sentences are fed to two common MT systems: Google translation and Systran translation, and the output of translation is stored in each hypothesis.txt, as shown in Figure x2. 

Figure x1

Figure x2

For BERTScore, we use the encoder from roberta without the importance weighting, and F1 score to evaluate as supported by the paper. For COMET, we use the Estimation model “wmt20-comet-qe-da”, trained based on DA and used Quality Estimation (QE) as a metric, and it is worth noting that this model is reference-free. The evaluation quality from BLEU, BERTScore, and COMET are illustrated in Figure x3, x4, x5. With limited 10 data samples, BERTScore and COMET consider Google Translator performing better, while BLEU score for Systran Translator is higher. 

Figure 3, x4, x5

The limitation of BLEU as compared to BERTScore and COMET is mostly exposed in the second sentence -- “我们在网络搜索和广告的创新，已使我们的网站成为全世界的顶级网站，使我们的品牌成全世界最获认可的品牌”. The BLEU score for Google is 19.29, while that of Systran is 44.96. The pure measurement of n-grams based on the exact string match causes the large difference in the evaluation of the translation qualities between the two systems. In comparison, the context based BERTScore and human judgement score based COMET do not have a significant difference in their scores, and this example proves the outdatedness of BLEU to some extent.

Let’s take a closer look at the 8th sentence-- “我们于1998年9月在加利福尼亚州注册成立 2003年8月在美国特拉华州重新注册.” Because the Systran’s translation exactly matched the reference sentence, so BLEU for this sentence is 100. However, Google’s translation “We were registered in California in September 1998 and re-registered in Delaware, USA in August 2003”, matches more with the source sentence in Chinese, especially the choice of word of “registered” instead of “incorporated”, and “Delaware, USA” instead of “Delaware”. The same lacking aspect is also shown in BERTScore, with a gap of 0.2 between the two systems. The COMET score for this sentence is 0.5144 for Google Translation versus 0.3090 for Systran Translation. We can see that the score for Systran is even lower, because COMET does not take reference sentences but the source sentences in Chinese as input. COMET aims to mimic how human judgement (DA in this case) will evaluate the translation, and clearly the Google translation provides a more exact translation as explained above. 

Not a trained translator myself, I cannot give my personal judgements on Google Translator and Systran Translator, but through the two examples, we clearly see the limitation of BLEU, and the limitation of BERTScore to some extent. However, it is still debatable if reference sentences should be evaluated in the metric. For COMET, inferring human judgement directly from source is appealing, but free-of-reference may result in loss of information in certain perspectives. Considering the paper’s experiment has proven the stronger effectiveness compared to BLEU and BERTScore, COMET may have pointed another direction for future MT evaluation metrics.

## Reference










