"use client";
import { Footer } from "@/app/components/footer";
import { Logo } from "@/app/components/logo";
import { PresetQuery } from "@/app/components/preset-query";
import { Search } from "@/app/components/search";
import React from "react";

export default function Home() {
  return (
    <div className="absolute inset-0 bg-[url('/bg.svg')]">
      <div className="absolute inset-0 min-h-[500px] min-w-[1000px] flex items-center justify-center">
        <div className="relative flex flex-col gap-8 px-4 -mt-24">
          <Logo></Logo>
          <Search></Search>
          <div className="flex gap-2 flex-wrap justify-center">
            <PresetQuery query="如何快速开始 Apache Doris？"></PresetQuery>
            <PresetQuery query="Doris 支持哪些数据源？"></PresetQuery>
            <PresetQuery query="什么场景下适合使用物化视图？"></PresetQuery>
          </div>
          <Footer></Footer>
        </div>
      </div>
    </div>
  );
}
