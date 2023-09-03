import AppLayout from "@/components/Layout/AppLayout"
import BlogLayout from "../layout"
import { Center, Image, Text } from "@chakra-ui/react"

export default function PPO(){
    return (
        <>
            <Text fontSize={"5xl"}>PPO — Intuitive guide to state-of-the-art Reinforcement Learning</Text>
            <Text fontSize={"3xl"} fontWeight={"bold"} mb={10}>Introduction</Text>
            <Center>
                <Image src="/imgs/blog/ppo.gif" />
            </Center>
        </>
    )
}


PPO.getLayout = function getLayout(page: React.ReactElement) {
    return (
        <AppLayout>
            <BlogLayout>
                {page}
            </BlogLayout>
        </AppLayout>
    )
}