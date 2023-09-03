import { Image, Text } from "@chakra-ui/react"

export default function DDPM(){
    return (
        <>
            <Text fontSize={"3xl"}>Generating images with DDPMs: A PyTorch Implementation</Text>
            <Image src="/imgs/blog/ddpm.gif" />
        </>
    )
}