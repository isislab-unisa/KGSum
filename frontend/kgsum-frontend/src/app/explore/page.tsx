"use client"
import {ReactNode} from "react";

export default function QueryBuilder(): ReactNode {

    return (
        <iframe
            src="http://www.isislab.it:12280/kgsum-graphdb"
            className="grow"
        />
    );
}
