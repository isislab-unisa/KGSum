"use client"
import {ReactNode} from "react";

export default function QueryBuilder(): ReactNode {

    return (
        <iframe
            src="http://172.20.0.1:7400/sparql"
            className="grow"
        />
    );
}
