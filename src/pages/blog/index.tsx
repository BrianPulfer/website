import AppLayout from '@/components/Layout/AppLayout';

import { Center, Image, Link, Stack, StackItem, Text } from "@chakra-ui/react";

export default function Projects() {
    return (
      <>
          <Text fontSize={"3xl"}>Blog</Text>
          <Text fontSize={"xl"} mb={10}>
            Welcome to my blog! If you like its content and would like to stay up-to-date,
            consider subscribing to the mailing list (coming soon!)
          </Text>
          <Stack spacing={10}>
            <StackItem>
              <Link href="/blog/ppo" textColor={"blue.500"}>
                <Text fontWeight={"bold"} fontSize={"2xl"} textAlign={"center"}>PPO — Intuitive guide to state-of-the-art Reinforcement Learning</Text>
                <Center>
                  <Image src="/imgs/blog/ppo.gif"/>
                </Center>
              </Link>
            </StackItem>
            <StackItem>
              <Link href="/blog/ddpm" textColor={"blue.500"}>
                <Text fontWeight={"bold"} fontSize={"2xl"} textAlign={"center"}>Generating images with DDPMs: A PyTorch Implementation</Text>
                <Center>
                  <Image src="/imgs/blog/ddpm.gif"/>
                </Center>
              </Link>
            </StackItem>
            <StackItem>
              <Link href="/blog/vit" textColor={"blue.500"}>
                <Text fontWeight={"bold"} fontSize={"2xl"} textAlign={"center"}>Vision Transformers from Scratch (PyTorch): A step-by-step guide</Text>
                <Center>
                  <Image src="/imgs/blog/vit.png"/>
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