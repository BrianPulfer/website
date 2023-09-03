import AppLayout from "@/components/Layout/AppLayout"
import BlogLayout from "../layout"
import { Center, Image, Link, Text } from "@chakra-ui/react"

export default function PPO(){
    return (
        <>
            <Text fontSize={"5xl"}>PPO — Intuitive guide to state-of-the-art Reinforcement Learning</Text>
            <Center>
                <Link target="_blank" href="https://colab.research.google.com/drive/1u7YTohPaQFJPud8289pV6H65f9ZqSKWp?usp=sharing">
                    <Image src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
                </Link>
            </Center>
            <Text fontSize={"3xl"} fontWeight={"bold"} mb={10}>Introduction</Text>
            <Text mb={5}>
                Proximal Policy Optimization (PPO) has been a state-of-the-art Reinforcement Learning (RL) algorithm since its proposal in the paper <Link href="https://arxiv.org/abs/1707.06347" textColor={"blue.500"}>Proximal Policy Optimization Algorithms (Schulman et. al., 2017)</Link>. This elegant algorithm can be and has been used for various tasks. Recently, it has also been used in the training of ChatGPT, the hottest machine-learning model at the moment.
            </Text>
            <Text mb={5}>
                PPO is not just widely used within the RL community, but it is also an excellent introduction to tackling RL through Deep Learning (DL) models.
            </Text>
            <Text mb={5}>
                In this article, I give a quick overview of the field of Reinforcement Learning, the taxonomy of algorithms to solve RL problems, and a review of the PPO algorithm proposed in the <Link href="https://arxiv.org/abs/1707.06347" textColor={"blue.500"}>paper</Link>. Finally, I share <Link href="https://colab.research.google.com/drive/1u7YTohPaQFJPud8289pV6H65f9ZqSKWp?usp=sharing" textColor={"blue.500"}>my own implementation</Link> of the PPO algorithm in PyTorch, comment on the obtained results and finish with a conclusion.
            </Text>
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