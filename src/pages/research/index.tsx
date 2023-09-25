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
                <Link textColor={'blue.500'} href="https://proceedings.mlr.press/v220/gruca22a.html">Weather4cast at NeurIPS 2022: Super-Resolution Rain Movie Prediction under Spatio-temporal Shifts</Link> (Sep 23)
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
