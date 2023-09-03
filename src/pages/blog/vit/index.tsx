import AppLayout from "@/components/Layout/AppLayout"
import BlogLayout from "../layout"
import { Image, Text} from "@chakra-ui/react"

export default function ViT(){
    return (
        <>
            <Text fontSize={"5xl"}>Vision Transformers from Scratch (PyTorch): A step-by-step guide</Text>
            <Image src="/imgs/blog/vit.png" />
        </>
    )
}

ViT.getLayout = function getLayout(page: React.ReactElement) {
    return (
        <AppLayout>
            <BlogLayout>
                {page}
            </BlogLayout>
        </AppLayout>
    )
}