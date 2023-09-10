import AppLayout from '@/components/Layout/AppLayout';
import {Image, Link, Text} from "@chakra-ui/react";

export default function Disambiguation(){
    return (
        <>
            <Text fontSize={"3xl"} textAlign={"center"} mb={10}>
                Machine Learning for disambiguation of scientific article authors
            </Text>
            <Text fontSize={"xl"} mb={5}>
            This project is an open-source implementation of a classifier which goal is to predict whether a pair of scientific articles (biomedical articles from the PubMed dataset) belongs to the same author or not.<br/><br/>
            The final classifier (Random Forest) used 15 features and had an accuracy of 87% with a 10-fold cross-validation. Further studies on the datasets revealed that for some combinations of last names and initial of first names (namespaces), over {"100'000"} articles could be found. This study explains the need for a classifier able to distinguish between these authors.<br/><br/>
            The project was my bachelor thesis job commissioned by Hoffmann-La Roche A.G.
            </Text>
            <Image src={"/imgs/projects/disambiguation.png"} alt={"Scientific disambiguation"} maxW={"xl"} mb={5}/>
            <Text fontSize={"xl"} mb={5} textAlign={"center"} fontWeight={"bold"}>
                You can visit the {"project's"} repository at the following <Link textColor={"blue.500"} href="https://github.com/BrianPulfer/AuthorNameDisambiguation">link</Link>.<br/>
                You can also visit the study on the Pubmed dataset at the following <Link textColor={"blue.500"} href="https://github.com/BrianPulfer/PubMed-Namespacer">link</Link>.<br/>
                Documentation (Italian Only) of the {"bachelor's"} thesis can be downloaded at this <Link textColor={"blue.500"} href="/docs/Brian Pulfer - Machine Learning for disambiguation of scientific article authors.pdf">link</Link>.
            </Text>
        </>
    );
}

Disambiguation.getLayout = function getLayout(page: React.ReactElement) {
    return (
        <AppLayout>
            {page}
        </AppLayout>
    )
}