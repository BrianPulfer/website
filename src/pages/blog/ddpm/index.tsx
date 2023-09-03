import AppLayout from "@/components/Layout/AppLayout";
import BlogLayout from "../layout";
import { Center, Image, Text } from "@chakra-ui/react";

export default function DDPM() {
  return (
    <>
      <Text fontSize={"5xl"}>
        Generating images with DDPMs: A PyTorch Implementation
      </Text>
      <Center>
        <Image src="/imgs/blog/ddpm.gif" />
      </Center>
    </>
  );
}

DDPM.getLayout = function getLayout(page: React.ReactElement) {
  return (
    <AppLayout>
      <BlogLayout>{page}</BlogLayout>
    </AppLayout>
  );
};
