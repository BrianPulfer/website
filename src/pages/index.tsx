import Head from 'next/head'
import { SocialIcon } from 'react-social-icons'

import { Image, Text, Box, Link } from '@chakra-ui/react'

import News from '@/components/News'
import AppLayout from '@/components/Layout/AppLayout'

export default function Home() {
  return (
    <>
      <Head><title>Brian - Home</title></Head>
      
      <Image src="/imgs/home/avatar.png" alt="Brian"/>

      
      <Text fontSize={"6xl"} bgGradient={"linear(to-b, gray.100, gray.900)"} className={"bg-clip-text text-transparent"}>Brian Pulfer</Text>
      
      <Box>
          <SocialIcon url="www.github.com" href="https://www.github.com/BrianPulfer"/>
          <SocialIcon url="www.linkedin.com" href="https://www.linkedin.com/in/brianpulfer/"/>
          <SocialIcon url="www.x.com" href="https://www.x.com/PulferBrian21"/>
      </Box>

      <Text textAlign={"center"} fontWeight={"bold"} className={'mt-10'} fontSize={"xl"}>
        Hey there, this is Brian! 👋
      </Text>
      <Text fontSize={"l"} maxW={"container.md"}>
        I am a Machine Learning practitioner and enthusiast. I am currently a Ph.D. student in Machine Learning, with a focus on anomaly detection and adversarial attacks, at the University of Geneva, Switzerland 🇨🇭. <Link textColor={"blue.500"} href='cv/BrianPulfer_CV.pdf'>{"Here's my CV"}</Link>.
        This is my personal portfolio, where I publish updates on my career, projects, publications and more. Hope you enjoy it!
      </Text>

      <Text textAlign={"center"} fontWeight={"bold"} fontSize={"5xl"} className={'mt-10'}>
        News 🗞️
      </Text>
      <Box className='flex flex-col space-y-4'>
        <News title={"September 2023"}>
          🥇 My team and I won the Migros workshop at the <Link textColor={"blue.500"} href={"https://hackzurich.com/"}>HackZurich 2023</Link> hackathon.<Link textColor={"blue.500"} href={"/blog/hackzurich23"}>Read the blog post</Link>.
        </News>
        <News title={"September 2023"}>
          📃 Our work <Link textColor={"blue.500"} href={"https://proceedings.mlr.press/v220/gruca22a.html"}>Weather4cast at NeurIPS 2022: Super-Resolution Rain Movie Prediction under Spatio-temporal Shifts</Link> has been accepted for publication in <Link textColor={"blue.500"} href={"https://proceedings.mlr.press/"}>Proceedings of Machine Learning Research</Link>.
        </News>
        <News title={"January 2023"}>
          📃 Our work <Link textColor={"blue.500"} href={"https://link.springer.com/article/10.1007/s10664-023-10306-x"}>Model vs System Level Testing of Autonomous Driving Systems: A Replication and Extension Study</Link> has been accepted for publication in <Link textColor={"blue.500"} href={"https://www.springer.com/journal/10664"}>Empirical Software Engineering</Link>.
        </News>
        <News title={"December 2022"}>
          🥉 Our work <Link textColor={"blue.500"} href={"https://arxiv.org/abs/2212.02456"}>Solving the Weather4cast Challenge via Visual Transformers for 3D Images</Link> got us the third place in the <Link textColor={"blue.500"} href={'https://www.iarai.ac.at/weather4cast/'}>2022 NeurIPS Weather4cast competition workshop</Link>.
        </News>
        <News title={"August 2022"}>
          📃 Our work <Link textColor={"blue.500"} href={'https://arxiv.org/abs/2209.15625'}>Anomaly localization for copy detection patterns through print estimations</Link> was accepted for publication in the <Link textColor={"blue.500"} href={"https://wifs2022.utt.fr/"}>IEEE International Workshop on Information Forensics & Security</Link>.
        </News>
        <News title={"June 2022"}>
          📃 Our work <Link textColor={"blue.500"} href={'https://arxiv.org/abs/2206.11793'}>Authentication of Copy Detection Patterns under Machine Learning Attacks: A Supervised Approach</Link> was accepted for publication in the <Link textColor={"blue.500"} href={'https://2022.ieeeicip.org/'}>29th IEEE International Conference on Image Processing (ICIP)</Link>.
        </News>
        <News title={"May 2022"}>
        🥇 I won the best presentation award for the 2022 edition of the <Link textColor={"blue.500"} href={'https://www.fondazionepremio.ch/premiati/'}>SwissEngineering Award Foundation</Link>.
        </News>
        <News title={"December 2021"}>
        📃 Our work <Link textColor={"blue.500"} href={'https://arxiv.org/abs/2112.11255'}>Mind the Gap! A Study on the Transferability of Virtual vs Physical-world Testing of Autonomous Driving Systems</Link> was accepted for publication in the <Link textColor={"blue.500"} href={'https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=32'}>IEEE Transactions of Software Engineering (TSE)</Link>.
        </News>
        <News title={"November 2021"}>
        👥 I Joined the <Link textColor={"blue.500"} href={'http://sip.unige.ch/'}>Stochastic Information Processing (SIP)</Link> group of the University of Geneva in quality of Ph.D. Student in Machine Learning.
        </News>
      </Box>

  </>
  )
}

Home.getLayout = function getLayout(page: React.ReactElement) {
  return (
    <AppLayout>
      {page}
    </AppLayout>
  )
}