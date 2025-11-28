'use client';

import Image from "next/image";
import {ReactNode} from "react";
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || '';

export function Banner(): ReactNode {
  return (
     <Image
         alt="Banner"
         src={`${BASE_PATH}/banner.png`}
         fill
         className="object-cover"
         priority
     />
  );
}