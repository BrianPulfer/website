import AppLayout from "@/components/Layout/AppLayout"

import BlogLayout from "../layout"
import { AspectRatio, Center, Image, Link, Text } from "@chakra-ui/react"

export default function HackZurich23(){

    return(
        <>
            <Text fontSize={"5xl"} textAlign={"center"}>
                HackZurich 2023
            </Text>
            <Center mb={5} className="flex flex-col">
                <Image className="flex flex-row" src="/imgs/blog/hackzurich23/win.png" alt="HackZurich 2023 team" />
                <Text textAlign={"center"} textColor={"gray.500"} fontSize={"sm"}>
                    Image of me (guy with glasses) and my team after winning the Migros workshop at the HackZurich 2023 hackathon. One team member is missing from the picture.
                </Text>
            </Center>
            
            <Text fontSize={"xl"} mb={5}>
                In the weekend from 15th to 17th September 2023, I participated in the HackZurich 2023 hackathon. After finding 4 amazing teammates from the HackZurich Discord server, we decided to participate in the Migros workshop.
            </Text>
            <Text fontSize={"xl"} mb={5}>
                The task was to create an application to incentivize Migros customers to buy more sustainable products. In pure Migros spirit, we proposed the following solution: When buying sustainable products, the customers are rewarded on the shop with scannable stickers that can be used, together with the packaging of sustainable products, to create toy monsters that kids can play with.
                We called them <b>Scrapsters</b>!
            </Text>

            <Center mb={5} className="flex flex-col">
                <Image src="/imgs/blog/hackzurich23/scrapsters.jpeg" alt="HackZurich 2023 team" />
                <Text textAlign={"center"} textColor={"gray.500"} fontSize={"sm"}>
                    Image of some of the scrapsters we created.
                </Text>
            </Center>

            <Text fontSize={"xl"} mb={5}>
                Here are the few Scrapsters we created. Tetra Breeze is a sort of scary bird, Turtle Tank is the slowest of scrapsters whil Octo Potz, my personal favourite, is a funny looking fish. All scrapsters are made from sustainable packaging and scannable stickers.
            </Text>

            <Text fontSize={"xl"} mb={5}>
                Together with the toys, we also created a web application that allows the customers to scan the stickers and show the story or "lore" along with an AI-generated image of the assembled scrapster.
            </Text>
            <Text fontSize={"xl"} mb={5}>
                If you want to learn more and access the application, check out our project on <Link textColor={"blue.500"} href="https://devpost.com/software/scrapsters">Devpost</Link>!
            </Text>

            <Center mb={5} className="flex flex-col">
                <Image src="/imgs/blog/hackzurich23/team.jpeg" alt="HackZurich 2023 team" />
                <Text textAlign={"center"} textColor={"gray.500"} fontSize={"sm"}>
                    The team while working on the project. From left to right: Jan, Brian (me), Oliver, Julia, Nicola.
                </Text>
            </Center>
            <Text fontSize={"xl"} mb={5}>
                I'd like once more to thank the team for the amazing work we did together and for the very unique experience. I had a lot of fun and I hope we can meet again in the future!
            </Text>
            <Text fontSize={"xl"} mb={5}>
                I'm also thanking the HackZurich team for the amazing organization, the sponsors and the workshop providers, especially Migros for providing us with the opportunity to work on such an interesting project.
            </Text>

        </>
    )
}

HackZurich23.getLayout = function getLayout(page: React.ReactElement) {
    return (
        <AppLayout>
            <BlogLayout>{page}</BlogLayout>
        </AppLayout>
    )
}