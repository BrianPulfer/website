import AppLayout from '@/components/Layout/AppLayout'

import Head from 'next/head'
import { Link } from '@chakra-ui/react'

import Publication, { PublicationTitle, PublicationAbstract, PublicationVenue, PublicationCitation } from '../../components/Publication'

export default function Publications (): JSX.Element {
  return (
      <>
          <Head><title>Research</title></Head>

          <Publication>
              <PublicationTitle className={'mb-4'}>
                <Link textColor={'blue.500'} href='https://arxiv.org/abs/2503.03842'>Task-Agnostic Attacks Against Vision Foundation Models</Link> (Mar 25)
              </PublicationTitle>
              <PublicationAbstract>
                The study of security in machine learning mainly focuses on downstream task-specific attacks, where the adversarial example is obtained by optimizing a loss function specific to the downstream task. At the same time, it has become standard practice for machine learning practitioners to adopt publicly available pre-trained vision foundation models, effectively sharing a common backbone architecture across a multitude of applications such as classification, segmentation, depth estimation, retrieval, question-answering and more. The study of attacks on such foundation models and their impact to multiple downstream tasks remains vastly unexplored. This work proposes a general framework that forges task-agnostic adversarial examples by maximally disrupting the feature representation obtained with foundation models. We extensively evaluate the security of the feature representations obtained by popular vision foundation models by measuring the impact of this attack on multiple downstream tasks and its transferability between models.
              </PublicationAbstract>
              <PublicationVenue className={'mt-2'}>
                Arxiv
              </PublicationVenue>
              <PublicationCitation>
                Pulfer, B., Belousov, Y., Kinakh, V., Furon, T., & Voloshynovskiy, S. (2025). Task-Agnostic Attacks Against Vision Foundation Models. https://arxiv.org/abs/2503.03842
              </PublicationCitation>
          </Publication>

          <Publication>
              <PublicationTitle className={'mb-4'}>
                <Link textColor={'blue.500'} href='https://arxiv.org/abs/2502.09696'>ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models</Link> (Feb 25)
              </PublicationTitle>
              <PublicationAbstract>
              Large Multimodal Models (LMMs) exhibit major shortfalls when interpreting images and, by some measures, have poorer spatial cognition than small children or animals. Despite this, they attain high scores on many popular visual benchmarks, with headroom rapidly eroded by an ongoing surge of model progress. To address this, there is a pressing need for difficult benchmarks that remain relevant for longer. We take this idea to its limit by introducing ZeroBench-a lightweight visual reasoning benchmark that is entirely impossible for contemporary frontier LMMs. Our benchmark consists of 100 manually curated questions and 334 less difficult subquestions. We evaluate 20 LMMs on ZeroBench, all of which score 0.0%, and rigorously analyse the errors. To encourage progress in visual understanding, we publicly release ZeroBench.
             </PublicationAbstract>
              <PublicationVenue className={'mt-2'}>
                Arxiv
              </PublicationVenue>
              <PublicationCitation>
                Roberts, J., Taesiri, M. R., Sharma, A., Gupta, A., Roberts, S., Croitoru, I., Bogolin, S.-V., Tang, J., Langer, F., Raina, V., Raina, V., Xiong, H., Udandarao, V., Lu, J., Chen, S., Purkis, S., Yan, T., Lin, W., Shin, G., … Albanie, S. (2025). ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models. https://arxiv.org/abs/2502.09696
              </PublicationCitation>
          </Publication>

          <Publication>
              <PublicationTitle className={'mb-4'}>
                <Link textColor={'blue.500'} href='https://link.springer.com/chapter/10.1007/978-3-031-73202-7_7'>Robustness Tokens: Towards Adversarial Robustness of Transformers</Link> (Jul 24)
              </PublicationTitle>
              <PublicationAbstract>
              Recently, large pre-trained foundation models have become widely adopted by machine learning practitioners for a multitude of tasks. Given that such models are publicly available, relying on their use as backbone models for downstream tasks might result in high vul- nerability to adversarial attacks crafted with the same public model. In this work, we propose Robustness Tokens, a novel approach specific to the transformer architecture that fine-tunes a few additional private tokens with low computational requirements instead of tuning model parameters as done in traditional adversarial training. We show that Robustness Tokens make Vision Transformer models significantly more robust to white-box adversarial attacks while also retaining the original downstream performances.
              </PublicationAbstract>
              <PublicationVenue className={'mt-2'}>
                European Conference on Computer Vision (ECCV), 2024
              </PublicationVenue>
              <PublicationCitation>
                Pulfer, B., Belousov, Y., & Voloshynovskiy, S. (2025). Robustness Tokens: Towards Adversarial Robustness of Transformers. In A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, & G. Varol (Eds.), Computer Vision – ECCV 2024 (pp. 110–127). Springer Nature Switzerland.
              </PublicationCitation>
          </Publication>
          <Publication>
              <PublicationTitle className={'mb-4'}>
                <Link textColor={'blue.500'} href='https://neurips.cc/virtual/2022/competition/50099'>Weather4cast at NeurIPS 2022: Super-Resolution Rain Movie Prediction under Spatio-temporal Shifts</Link> (Sep 23)
              </PublicationTitle>
              <PublicationAbstract>
                Weather4cast again advanced modern algorithms in AI and machine learning through a highly topical interdisciplinary competition challenge: The prediction of hi-res rain radar movies from multi-band satellite sensors, requiring data fusion, multi-channel video frame prediction, and super-resolution. Accurate predictions of rain events are becoming ever more critical, with climate change increasing the frequency of unexpected rainfall. The resulting models will have a particular impact where costly weather radar is not available. We here present highlights and insights emerging from the thirty teams participating from over a dozen countries. To extract relevant patterns, models were challenged by spatio-temporal shifts. Geometric data augmentation and test-time ensemble models with a suitable smoother loss helped this transfer learning. Even though, in ablation, static information like geographical location and elevation was not linked to performance, the general success of models incorporating physics in this competition suggests that approaches combining machine learning with application domain knowledge seem a promising avenue for future research. Weather4cast will continue to explore the powerful benchmark reference data set introduced here, advancing competition tasks to quantitative predictions, and exploring the effects of metric choice on model performance and qualitative prediction properties.
              </PublicationAbstract>
              <PublicationVenue className={'mt-2'}>
                Proceedings of Machine Learning Research (PMLR), 2023
              </PublicationVenue>
              <PublicationCitation>
                Gruca, A., Serva, F., Lliso, L., Rípodas, P., Calbet, X., Herruzo, P., Pihrt, J., Raevskyi, R., Šimánek, P., Choma, M., Li, Y., Dong, H., Belousov, Y., Polezhaev, S., Pulfer, B., Seo, M., Kim, D., Shin, S., Kim, E., … Kreil, D. P. (2022). Weather4cast at NeurIPS 2022: Super-Resolution Rain Movie Prediction under Spatio-temporal Shifts. In M. Ciccone, G. Stolovitzky, & J. Albrecht (Eds.), Proceedings of the NeurIPS 2022 Competitions Track (Vol. 220, pp. 292–313). PMLR. https://proceedings.mlr.press/v220/gruca22a.html
              </PublicationCitation>
          </Publication>
          <Publication>
              <PublicationTitle className={'mb-4'}>
                <Link textColor={'blue.500'} href="https://arxiv.org/abs/2212.02456">Solving the Weather4cast Challenge via Visual Transformers for 3D Images</Link> (Dec 22)
              </PublicationTitle>
              <PublicationAbstract>
                Accurately forecasting the weather is an important task, as many real-world processes and decisions depend on future meteorological conditions. The NeurIPS 2022 challenge entitled Weather4cast poses the problem of predicting rainfall events for the next eight hours given the preceding hour of satellite observations as a context. Motivated by the recent success of transformer-based architectures in computer vision, we implement and propose two methodologies based on this architecture to tackle this challenge. We find that ensembling different transformers with some baseline models achieves the best performance we could measure on the unseen test data. Our approach has been ranked 3rd in the competition.
              </PublicationAbstract>
              <PublicationVenue className={'mt-2'}>
                ArXiv (NeurIPS 2022 challenge), 2022
              </PublicationVenue>
              <PublicationCitation>
                Belousov, Y., Polezhaev, S., & Pulfer, B. (2022). Solving the Weather4cast Challenge via Visual Transformers for 3D Images.
              </PublicationCitation>
          </Publication>
          <Publication>
              <PublicationTitle className={'mb-4'}>
                <Link textColor={'blue.500'} href="https://arxiv.org/abs/2209.15625">Anomaly localization for copy detection patterns through print estimation</Link> (Aug 22)
              </PublicationTitle>
              <PublicationAbstract>
                Copy detection patterns (CDP) are recent technologies for protecting products from counterfeiting. However, in contrast to traditional copy fakes, deep learning-based fakes have shown to be hardly distinguishable from originals by traditional authentication systems. Systems based on classical supervised learning and digital templates assume knowledge of fake CDP at training time and cannot generalize to unseen types of fakes. Authentication based on printed copies of originals is an alternative that yields better results even for unseen fakes and simple authentication metrics but comes at the impractical cost of acquisition and storage of printed copies. In this work, to overcome these shortcomings, we design a machine learning (ML) based authentication system that only requires digital templates and printed original CDP for training, whereas authentication is based solely on digital templates, which are used to estimate original printed codes. The obtained results show that the proposed system can efficiently authenticate original and detect fake CDP by accurately locating the anomalies in the fake CDP. The empirical evaluation of the authentication system under investigation is performed on the original and ML-based fakes CDP printed on two industrial printers.
              </PublicationAbstract>
              <PublicationVenue className={'mt-2'}>
                IEEE International Workshop on Information Forensics and Security (WIFS), 2022
              </PublicationVenue>
              <PublicationCitation>
                {'Pulfer, B., Belousov, Y., Tutt, J., Chaban, R., Taran, O., Holotyak, T., & Voloshynovskiy, S.. (2022). "Anomaly localization for copy detection patterns through print estimations," in IEEE International Workshop on Information Forensics and Security (WIFS), 2022.'}
              </PublicationCitation>
          </Publication>
          <Publication>
              <PublicationTitle className={'mb-4'}>
                <Link textColor={'blue.500'} href="https://arxiv.org/abs/2206.11793">Authentication of Copy Detection Patterns under Machine Learning Attacks: A Supervised Approach</Link> (Jun 22)
              </PublicationTitle>
              <PublicationAbstract>
              Copy detection patterns (CDP) are an attractive technology that allows manufacturers to defend their products against counterfeiting. The main assumption behind the protection mechanism of CDP is that these codes printed with the smallest symbol size (1x1) on an industrial printer cannot be copied or cloned with sufficient accuracy due to data processing inequality. However, previous works have shown that Machine Learning (ML) based attacks can produce high-quality fakes, resulting in decreased accuracy of authentication based on traditional feature-based authentication systems. While Deep Learning (DL) can be used as a part of the authentication system, to the best of our knowledge, none of the previous works has studied the performance of a DL-based authentication system against ML-based attacks on CDP with 1x1 symbol size. In this work, we study such a performance assuming a supervised learning (SL) setting.
              </PublicationAbstract>
              <PublicationVenue className={'mt-2'}>
                IEEE International Conference on Image Processing (ICIP), Bordeaux, France, 2022.
              </PublicationVenue>
              <PublicationCitation>
              B. Pulfer, R. Chaban, Y. Belousov, J. Tutt, O. Taran, T. Holotyak, and S. Voloshynovskiy, “Authentication of copy detection patterns under machine learning attacks: A supervised approach,” in IEEE International Conference on Image Processing (ICIP), Bordeaux, France, October 2022.
              </PublicationCitation>
          </Publication>
          <Publication>
              <PublicationTitle className={'mb-4'}>
                <Link textColor={'blue.500'} href="https://ieeexplore.ieee.org/document/9869302">
                  Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems
                </Link> (Dec 21)
              </PublicationTitle>
              <PublicationAbstract>Safe deployment of self-driving cars (SDC) necessitates thorough simulated and in-field testing. Most testing techniques consider virtualized SDCs within a simulation environment, whereas less effort has been directed towards assessing whether such techniques transfer to and are effective with a physical real-world vehicle. In this paper, we leverage the Donkey Car open-source framework to empirically compare testing of SDCs when deployed on a physical small-scale vehicle vs its virtual simulated counterpart. In our empirical study, we investigate transferability of behavior and failure exposure between virtual and real-world environments on a vast set of corrupted and adversarial settings. While a large number of testing results do transfer between virtual and physical environments, we also identified critical shortcomings that contribute to the reality gap between the virtual and physical world, threatening the potential of existing testing solutions when applied to physical SDCs</PublicationAbstract>
              <PublicationVenue className={'mt-2'}>IEEE Transactions on Software Engineering</PublicationVenue>
              <PublicationCitation>{'A. Stocco, B. Pulfer and P. Tonella, "Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems," in IEEE Transactions on Software Engineering, 2022, doi: 10.1109/TSE.2022.3202311.'}</PublicationCitation>
          </Publication>
      </>
  )
}

Publications.getLayout = function getLayout (page: React.ReactElement) {
  return (
      <AppLayout>
        {page}
      </AppLayout>
  )
}
