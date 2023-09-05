import AppLayout from '@/components/Layout/AppLayout';

import { AspectRatio, Center, Image, Link, Stack, StackItem, Text } from "@chakra-ui/react";

export default function Projects() {
  return (
    <>
      <Text fontSize={"3xl"}>Projects</Text>
      <Text fontSize={"xl"}>
        Here's a few projects that I have worked or am currently working on.
      </Text>
      <Stack mt={100} alignContent={"center"}>
        <StackItem mb={10}>
          <Link href="projects/self-driving" textColor={"blue.500"}>
            <Text fontSize={"2xl"} fontWeight={"bold"} textAlign={"center"}>
              From Simulated to Real Test Environments for Self Driving Cars
            </Text>
            <Center>
              <AspectRatio width={"100%"} maxWidth={"700px"} ratio={1920/1080} mb={5}>
                  <iframe title="From Simulated to Real Test Environments for Self Driving Cars" src="https://www.youtube.com/embed/7q2hwzWo7Cw" allowFullScreen/>
              </AspectRatio>
            </Center>
          </Link>
        </StackItem>
        <StackItem mb={10}>
          <Link href="projects/stylegan2-distillation" textColor={"blue.500"}>
            <Text fontSize={"2xl"} fontWeight={"bold"} textAlign={"center"}>
              StyleGAN2 Distillation
            </Text>
            <Center>
              <AspectRatio width={"100%"} maxWidth={"700px"} ratio={1} mb={5}>
                <Image src={"/imgs/projects/stylegan2-distillation.png"} alt={"StyleGAN Distillation"} mb={5}/>
              </AspectRatio>
            </Center>
          </Link>
        </StackItem>
        <StackItem mb={10}>
          <Link href="projects/disambiguation" textColor={"blue.500"}>
            <Text fontSize={"2xl"} fontWeight={"bold"} textAlign={"center"}>
              Machine Learning for disambiguation of scientific article authors
            </Text>
            <Center>
              <AspectRatio width={"100%"} maxWidth={"700px"} ratio={9/5} mb={5}>
                <Image src={"/imgs/projects/disambiguation.png"} alt={"Disambiguation"}  mb={5}/>
              </AspectRatio>
            </Center>
          </Link>
        </StackItem>
      </Stack>
    </>
  );
}

Projects.getLayout = function getLayout(page: React.ReactElement) {
  return (
    <AppLayout>
      {page}
    </AppLayout>
  )
}