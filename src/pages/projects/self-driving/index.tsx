import Head from 'next/head';
import AppLayout from '@/components/Layout/AppLayout';
import {AspectRatio, Link, Text} from "@chakra-ui/react";

export default function SelfDriving(){
    return (
        <>
            <Head><title>Projects - Self-Driving</title></Head>
            <Text fontSize={"3xl"} textAlign={"center"} mb={10}>
                From Simulated to Real Test Environments for Self Driving Cars
            </Text>
            <Text fontSize={"xl"} mb={5}>
                In my master thesis in Artificial Intelligence, I studied testing in the field of self-driving cars through a small-scale car and simulator.<br/><br/>
                Through the use of CycleGAN, I propose a method to estimate the Cross-Track Error in the real world (important testing metric already in use for simulators) and use it to assess whether offline and online testing for self-driving cars yields similar results, both in a real and simulated environment.<br/><br/>
            </Text>
            <AspectRatio minWidth={"485px"} width={"100%"} ratio={1920/1080} mb={5}>
                <iframe title="From Simulated to Real Test Environments for Self Driving Cars" src="https://www.youtube.com/embed/7q2hwzWo7Cw" allowFullScreen/>
            </AspectRatio>
            <Text fontSize={"xl"} mb={5}>
                Given the enthusiasm that me and my co-supervisor had towards this small-scale car, we even organized the first <Link textColor={"blue.500"} href="https://formulausi.si.usi.ch/2021/">FormulaUSI</Link> event! The goal of the event was to educate participants on Artificial Intelligence while racing self-driving small-scale cars. We had much fun organizing the event, and I have personally grown by such an experience.
            </Text>
            <Text fontSize={"xl"} mb={5} textAlign={"center"} fontWeight={"bold"}>
                My master thesis can be downloaded at this <Link textColor={"blue.500"} href="/docs/Brian Pulfer - From Simulated to Real Test Environments for Self Driving Cars.pdf">link</Link>.<br/>
                {"Here's"} the links to the <Link textColor={"blue.500"} href="https://formulausi.si.usi.ch/2021/">FormulaUSI competition website</Link> and <Link textColor={"blue.500"} href="https://www.youtube.com/watch?v=PDeCb4vBEC4&amp;ab_channel=SoftwareInstitute">highlights</Link>.
            </Text>
        </>
    );
}

SelfDriving.getLayout = function getLayout(page: React.ReactElement) {
    return (
        <AppLayout>
            {page}
        </AppLayout>
    )
}