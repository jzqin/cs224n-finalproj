# cs224n-finalproj
Final Project for Stanford CS224N, winter 2021

Motivating Research:
* Backgrond on task (basis challenge MRQA): https://arxiv.org/pdf/1910.09753.pdf (Fisch et al. 2019)
  * Extrapolation vs. Interpolation to training distribution
  * How in what ways might "in-domain" and "out-of-domain" datasets differ? Passage sources, Question styles and sources, Joint Distribution of questions and passages (i.e. some questions written based on passage, others written independently). This paper has more details regarding datasets used in class project.
  * Baseline improvement techniques include: Dataset sampling (look for similarities between training and test sets), Multitask Learning, Adversarial Training, Ensembling. Best performance was Li et al. (Baidu) - https://www.aclweb.org/anthology/D19-5828.pdf
* Multi-Task Learning:
  * Multi-Task Deep Neural Networks for Natural Language Understanding: https://arxiv.org/pdf/1901.11504.pdf (Liu et al 2019)
  * An embarrassingly simple approach for transfer learning from pretrained language models: https://arxiv.org/pdf/1902.10547.pdf (Chronopoulo et al. 2019)  
* Fine-Tuning BERT: https://arxiv.org/abs/2006.05987

* In-Context Learning: https://arxiv.org/pdf/2012.15723.pdf (Gao et al. 2020)
* Meta-Learning: https://arxiv.org/pdf/1911.03863.pdf (Bansal et al. 2020)
