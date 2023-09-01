import {Box, Container, Text} from "@chakra-ui/react";
import { useColorModeValue } from "@chakra-ui/react";

export function PublicationTitle({children, className}: {children: React.ReactNode, className?: string}){
    return (
        <Text fontSize={"xl"} textAlign={"left"} className={className}>
            {children}
        </Text>
    )
}

export function PublicationAbstract({children}: {children: React.ReactNode}){
    return (
        <>
            <Text fontWeight={"bold"} >
                Abstract:
            </Text>
            <Text fontSize={"14"}>
                {children}
            </Text>
        </>
    )
}

export function PublicationVenue({children, className}: {children: React.ReactNode, className?: string}){
    return (
        <Text textAlign={"center"} fontWeight={"bold"} className={className}>
            {children}
        </Text>
    )
}

export function PublicationCitation({children, className}: {children: React.ReactNode, className?: string}){
    return (
        <Text fontStyle={"italic"} className={className}>
            {children}
        </Text>
    )
}


export default function Publication({children}: {children: React.ReactNode}){
    return (
        <Container m={0} maxW={"100%"} p={"4"} mb={"10"} borderRadius={"xl"} boxShadow={"2xl"} backgroundColor={useColorModeValue("gray.100", "gray.900")}>
            {children}
        </Container>
    )
}
