import Head from 'next/head'

import { Box, HStack, Icon, Image, Link, Text } from '@chakra-ui/react'

import News from '@/components/News'
import AppLayout from '@/components/Layout/AppLayout'

// Icon paths adapted from the former react-social-icons dependency.
const SOCIAL_ICONS = {
  github: {
    color: '#24292e',
    path: 'M0,0v64h64V0H0z M37.1,47.2c-0.8,0.2-1.1-0.3-1.1-0.8c0-0.5,0-2.3,0-4.4c0-1.5-0.5-2.5-1.1-3 c3.6-0.4,7.3-1.7,7.3-7.9c0-1.7-0.6-3.2-1.6-4.3c0.2-0.4,0.7-2-0.2-4.2c0,0-1.3-0.4-4.4,1.6c-1.3-0.4-2.6-0.5-4-0.5 c-1.4,0-2.7,0.2-4,0.5c-3.1-2.1-4.4-1.6-4.4-1.6c-0.9,2.2-0.3,3.8-0.2,4.2c-1,1.1-1.6,2.5-1.6,4.3c0,6.1,3.7,7.5,7.3,7.9 c-0.5,0.4-0.9,1.1-1,2.1c-0.9,0.4-3.2,1.1-4.7-1.3c0,0-0.8-1.5-2.5-1.6c0,0-1.6,0-0.1,1c0,0,1,0.5,1.8,2.3c0,0,0.9,3.1,5.4,2.1 c0,1.3,0,2.3,0,2.7c0,0.4-0.3,0.9-1.1,0.8C20.6,45.1,16,39.1,16,32c0-8.8,7.2-16,16-16c8.8,0,16,7.2,16,16 C48,39.1,43.4,45.1,37.1,47.2z'
  },
  linkedin: {
    color: '#007fb1',
    path: 'M0,0v64h64V0H0z M25.8,44h-5.4V26.6h5.4V44z M23.1,24.3c-1.7,0-3.1-1.4-3.1-3.1c0-1.7,1.4-3.1,3.1-3.1 c1.7,0,3.1,1.4,3.1,3.1C26.2,22.9,24.8,24.3,23.1,24.3z M46,44h-5.4v-8.4c0-2,0-4.6-2.8-4.6c-2.8,0-3.2,2.2-3.2,4.5V44h-5.4V26.6 h5.2V29h0.1c0.7-1.4,2.5-2.8,5.1-2.8c5.5,0,6.5,3.6,6.5,8.3V44z'
  },
  x: {
    color: '#000000',
    path: 'M 0 0 L 0 64 L 64 64 L 64 0 L 0 0 z M 16 17.537109 L 26.125 17.537109 L 33.117188 26.779297 L 41.201172 17.537109 L 46.109375 17.537109 L 35.388672 29.789062 L 48 46.462891 L 38.125 46.462891 L 30.390625 36.351562 L 21.541016 46.462891 L 16.632812 46.462891 L 28.097656 33.357422 L 16 17.537109 z M 21.730469 20.320312 L 39.480469 43.525391 L 42.199219 43.525391 L 24.648438 20.320312 L 21.730469 20.320312 z'
  }
} satisfies Record<string, { color: string, path: string }>

type SocialKey = keyof typeof SOCIAL_ICONS

function SocialLink ({ href, label, iconKey }: { href: string, label: string, iconKey: SocialKey }): JSX.Element {
  const { color, path } = SOCIAL_ICONS[iconKey]

  return (
    <Link
      href={href}
      isExternal
      aria-label={label}
      display={'inline-flex'}
      alignItems={'center'}
      justifyContent={'center'}
      w={12}
      h={12}
      borderRadius={'full'}
      transition={'transform 0.2s ease, box-shadow 0.2s ease'}
      boxShadow={'lg'}
      _hover={{ transform: 'translateY(-2px)', boxShadow: 'xl' }}
    >
      <Icon viewBox={'0 0 64 64'} boxSize={8}>
        <path fill={'white'} d={`M0,0H64V64H0Z${path}`} />
        <path fill={color} d={path} />
      </Icon>
    </Link>
  )
}

export default function Home (): JSX.Element {
  return (
    <>
      <Head><title>Brian Pulfer</title></Head>

      <Image src="/imgs/home/avatar.png" alt="Brian"/>

      <Text fontSize={'6xl'} bgGradient={'linear(to-b, gray.100, gray.900)'} className={'bg-clip-text text-transparent'}>Brian Pulfer</Text>

      <HStack spacing={4}>
        <SocialLink href={'https://www.github.com/BrianPulfer'} label={'GitHub'} iconKey={'github'} />
        <SocialLink href={'https://www.linkedin.com/in/brianpulfer/'} label={'LinkedIn'} iconKey={'linkedin'} />
        <SocialLink href={'https://www.x.com/peutlefaire'} label={'X'} iconKey={'x'} />
      </HStack>

      <Text textAlign={'center'} fontWeight={'bold'} className={'mt-10'} fontSize={'xl'}>
        Hey there, this is Brian! 👋
      </Text>
      <Text fontSize={'l'} maxW={'container.md'}>
        I am a Machine Learning practitioner and enthusiast. I am currently a Ph.D. student in Machine Learning, with a focus on anomaly detection and adversarial attacks, at the University of Geneva, Switzerland 🇨🇭. <Link textColor={'blue.500'} href='cv/BrianPulfer_CV.pdf'>{"Here's my CV"}</Link>.
        This is my personal portfolio, where I publish updates on my career, projects, publications and more. Hope you enjoy it!
      </Text>

      <Text textAlign={'center'} fontWeight={'bold'} fontSize={'5xl'} className={'mt-10'}>
        News 🗞️
      </Text>
      <Box className='flex flex-col space-y-4'>
        <News title={'June 2026'}>
          🔎 I contributed as a reviewer for NeurIPS 2026 (4 papers) and BMVC 2026 (3 papers).
        </News>
        <News title={'May 2026'}>
          🥈 I received the <i>Silver Reviewer Award</i> for my reviews spanning 5 papers for ICML 2026.
        </News>
        <News title={'May 2026'}>
          📃 Our paper ZeroBench <Link textColor={'blue.500'} href={'https://arxiv.org/abs/2502.09696'}>ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models</Link> has been accepted at <i>ICML 2026</i>.
        </News>
        <News title={'May 2026'}>
          📃 Our paper <Link textColor={'blue.500'} href={'https://arxiv.org/abs/2605.23065'}>Dithering Defense: Adversarial Robustness of Vision Foundation Models via Multi-Level Floyd-Steinberg Dithering</Link> has been accepted at the <i>IEEE International Conference on Image Processing (ICIP 2026)</i>.
        </News>
        <News title={'September 2025'}>
          📃 Our paper <Link textColor={'blue.500'} href={'https://arxiv.org/abs/2505.15594'}>Beyond Classification: Evaluating Diffusion Denoised Smoothing for Security-Utility Trade off</Link> has been accepted at the <i>IEEE 33rd European Signal Processing Conference (EUSIPCO 2025)</i>.
        </News>
        <News title={'June 2025'}>
           👨🏽‍💻I am interning a as a Ph.D. student at <b>Meta</b> in Menlo Park, California, for the summer of 2025 in the GPU Techniques Team!
        </News>
        <News title={'June 2025'}>
          📃 I presented our paper <Link textColor={'blue.500'} href={'https://openaccess.thecvf.com/content/CVPR2025W/AdvML/papers/Pulfer_Task-Agnostic_Attacks_Against_Vision_Foundation_Models_CVPRW_2025_paper.pdf'}>Task-Agnostic Attacks Against Vision Foundation Models</Link> at <b>CVPR 2025</b> for the <i>5th Workshop of Adversarial Machine Learning on Computer Vision: Foundation Models + X</i>.
        </News>
        <News title={'March 2025'}>
          📃 Our paper <Link textColor={'blue.500'} href={'https://arxiv.org/abs/2503.03842'}>Task-Agnostic Attacks Against Vision Foundation Models</Link> is now available on ArXiv.
        </News>
        <News title={'February 2025'}>
          📃 I contributed as part of the red team for <Link textColor={'blue.500'} href={'https://arxiv.org/abs/2502.09696'}>ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models</Link>.
        </News>
        <News title={'November 2024'}>
          🕵️ I contributed as a reviewer of 3 papers for <Link textColor={'blue.500'} href={'https://cvpr.thecvf.com/Conferences/2025'}>CVPR 2025</Link>.
        </News>
        <News title={'July 2024'}>
          📃 Our paper <Link textColor={'blue.500'} href={'https://link.springer.com/chapter/10.1007/978-3-031-73202-7_7'}>Robustness Tokens: Towards Adversarial Robustness for Transformers</Link> (deep double blind) has been accepted for publication at the <Link textColor={'blue.500'} href={'https://eccv2024.ecva.net/'}>European Conference on Computer Vision (ECCV) 2024.</Link>.
        </News>
        <News title={'September 2023'}>
          🥇 My team and I won the Migros workshop at the <Link textColor={'blue.500'} href={'https://hackzurich.com/'}>HackZurich 2023</Link> hackathon. <Link textColor={'blue.500'} href={'/blog/hackzurich23'}>Read the blog post</Link>.
        </News>
        <News title={'September 2023'}>
          📃 Our work <Link textColor={'blue.500'} href={'https://neurips.cc/virtual/2022/competition/50099'}>Weather4cast at NeurIPS 2022: Super-Resolution Rain Movie Prediction under Spatio-temporal Shifts</Link> has been accepted for publication in <Link textColor={'blue.500'} href={'https://proceedings.mlr.press/'}>Proceedings of Machine Learning Research</Link>.
        </News>
        <News title={'January 2023'}>
          📃 Our work <Link textColor={'blue.500'} href={'https://link.springer.com/article/10.1007/s10664-023-10306-x'}>Model vs System Level Testing of Autonomous Driving Systems: A Replication and Extension Study</Link> has been accepted for publication in <Link textColor={'blue.500'} href={'https://www.springer.com/journal/10664'}>Empirical Software Engineering</Link>.
        </News>
        <News title={'December 2022'}>
          🥉 Our work <Link textColor={'blue.500'} href={'https://arxiv.org/abs/2212.02456'}>Solving the Weather4cast Challenge via Visual Transformers for 3D Images</Link> got us the third place in the <Link textColor={'blue.500'} href={'https://www.iarai.ac.at/weather4cast/'}>2022 NeurIPS Weather4cast competition workshop</Link>.
        </News>
        <News title={'August 2022'}>
          📃 Our work <Link textColor={'blue.500'} href={'https://arxiv.org/abs/2209.15625'}>Anomaly localization for copy detection patterns through print estimations</Link> was accepted for publication in the <Link textColor={'blue.500'} href={'https://wifs2022.utt.fr/'}>IEEE International Workshop on Information Forensics & Security</Link>.
        </News>
        <News title={'June 2022'}>
          📃 Our work <Link textColor={'blue.500'} href={'https://arxiv.org/abs/2206.11793'}>Authentication of Copy Detection Patterns under Machine Learning Attacks: A Supervised Approach</Link> was accepted for publication in the <Link textColor={'blue.500'} href={'https://2022.ieeeicip.org/'}>29th IEEE International Conference on Image Processing (ICIP)</Link>.
        </News>
        <News title={'May 2022'}>
        🥇 I won the best presentation award for the 2022 edition of the <Link textColor={'blue.500'} href={'https://www.fondazionepremio.ch/premiati/'}>SwissEngineering Award Foundation</Link>.
        </News>
        <News title={'December 2021'}>
        📃 Our work <Link textColor={'blue.500'} href={'https://arxiv.org/abs/2112.11255'}>Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems</Link> was accepted for publication in the <Link textColor={'blue.500'} href={'https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=32'}>IEEE Transactions of Software Engineering (TSE)</Link>.
        </News>
        <News title={'November 2021'}>
        👥 I Joined the <Link textColor={'blue.500'} href={'http://sip.unige.ch/'}>Stochastic Information Processing (SIP)</Link> group of the University of Geneva in quality of Ph.D. Student in Machine Learning.
        </News>
      </Box>

  </>
  )
}

Home.getLayout = function getLayout (page: React.ReactElement) {
  return (
    <AppLayout>
      {page}
    </AppLayout>
  )
}
