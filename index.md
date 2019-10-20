 * [paper](https://arxiv.org/abs/1809.02869) (old version, to be updated)
 * [poster](poster.pdf)
 * [code](https://github.com/AaltoPML/machine-teaching-of-active-sequential-learners/)

**Reference**:

Tomi Peltola, Mustafa Mert Çelikok, Pedram Daee, Samuel Kaski<br />
**Machine Teaching of Active Sequential Learners**<br />
NeurIPS 2019

**Abstract**:

Machine teaching addresses the problem of finding the best training data that can guide a learning algorithm to a target model with minimal effort. However, for sequential learners which actively choose their queries, such as multi-armed bandits and active learners, the teacher can only provide responses to the learner's queries. We formulate this sequential teaching problem, which the current techniques in machine teaching do not address, as a Markov decision process. Furthermore, we address the complementary problem of learning from a teacher: to recognise the teaching intent of received responses, the learner is endowed with a model of the teacher. We test the formulation with multi-armed bandit learners in simulated experiments and a user study. The results show that learning is improved by (1) computing teaching policies from the model and by (2) the learner having a model of the teacher. The approach gives tools to taking into account strategic (planning) behaviour of users of interactive intelligent systems, such as recommendation engines, by considering them as boundedly optimal teachers.
