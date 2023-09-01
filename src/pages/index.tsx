import Head from 'next/head'
import { Container, Image, Text, Box, Link } from '@chakra-ui/react'

import News from '@/components/News'

export default function Home() {
  return (
    <>
      <Head><title>Brian - Home</title></Head>
      
      <Image src="/imgs/home/avatar.png" alt="Brian" />
      
      <Text fontSize={"6xl"} bgGradient={"linear(to-b, gray.100, gray.900)"} className={"bg-clip-text text-transparent"}>Brian Pulfer</Text>
      
      <Box>
        <Link
          href={"https://www.github.com/BrianPulfer"}
          type="button"
          data-te-ripple-init
          data-te-ripple-color="light"
          className={"mb-2 inline-block rounded px-6 py-2.5 text-xs font-medium uppercase leading-normal text-white shadow-md transition duration-150 ease-in-out hover:shadow-lg focus:shadow-lg focus:outline-none focus:ring-0 active:shadow-lg"}
          style={{"backgroundColor": "#333"}}>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className={"h-4 w-4"}
            fill="white"
            viewBox="0 0 24 24">
            <path
              d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
          </svg>
        </Link>

        <Link
          href={"https://www.linkedin.com/in/brianpulfer/"}
          type="button"
          data-te-ripple-init
          data-te-ripple-color="light"
          className={"mb-2 inline-block rounded px-6 py-2.5 text-xs font-medium uppercase leading-normal text-white shadow-md transition duration-150 ease-in-out hover:shadow-lg focus:shadow-lg focus:outline-none focus:ring-0 active:shadow-lg"}
          style={{"backgroundColor": "#0077b5"}}>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className={"h-4 w-4"}
            fill="white"
            viewBox="0 0 24 24">
            <path
              d="M4.98 3.5c0 1.381-1.11 2.5-2.48 2.5s-2.48-1.119-2.48-2.5c0-1.38 1.11-2.5 2.48-2.5s2.48 1.12 2.48 2.5zm.02 4.5h-5v16h5v-16zm7.982 0h-4.968v16h4.969v-8.399c0-4.67 6.029-5.052 6.029 0v8.399h4.988v-10.131c0-7.88-8.922-7.593-11.018-3.714v-2.155z" />
          </svg>
        </Link>

        <Link
          href={"https://www.x.com/PulferBrian21"}
          type="button"
          data-te-ripple-init
          data-te-ripple-color="light"
          className={"mb-2 inline-block rounded px-6 py-2.5 text-xs font-medium uppercase leading-normal text-white shadow-md transition duration-150 ease-in-out hover:shadow-lg focus:shadow-lg focus:outline-none focus:ring-0 active:shadow-lg"}
          style={{"backgroundColor": "#1da1f2"}}>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className={"h-4 w-4"}
            fill="white"
            viewBox="0 0 24 24">
            <path
              d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z" />
          </svg>
        </Link>
      </Box>

      <Text textAlign={"center"} fontWeight={"bold"} className={'mt-10'} fontSize={"xl"}>
        Hey there, this is Brian! 👋
      </Text>
      <Text fontSize={"l"} maxW={"container.md"}>
        I am a Machine Learning practitioner and enthusiast. I am currently a Ph.D. student in Machine Learning, with a focus on anomaly detection and adversarial attacks, at the University of Geneva, Switzerland 🇨🇭. <Link textColor={"blue.500"} href='cv/BrianPulfer_CV.pdf'>Here's my CV</Link>.
        This is my personal portfolio, where I publish updates on my career, projects, publications and more. Hope you enjoy it!
      </Text>

      <Text textAlign={"center"} fontWeight={"bold"} fontSize={"5xl"} className={'mt-10'}>
        News 🗞️
      </Text>
      <Box className='flex flex-col space-y-4'>
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