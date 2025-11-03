"use client";

import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";

import React from "react";
import Slider from "react-slick";

export default function MySlider() {
  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
    Infinity: true,
    autoplay: true,
  };

  const slides = [
    { image: "./images/pexels-danxavier-1239291.jpg", title: "This journal helps me untangle the mess in my head — one entry at a time." },
    { image: "./images/pexels-justin-shaifer-501272-1222271.jpg", title: "It's comforting to see my feelings being understood and analyzed without judgment." },
    { image: "./images/pexels-olly-837358.jpg", title: "I track my moods, I reflect, and now I see patterns I never noticed before" },
  ];

  return (

    <>

      <div style={{display: "flex", flexDirection: "column",  justifyContent: "space-between",alignItems: "center", marginTop: "2rem"}}>

        <div style={{margin:"0rem 8rem",fontSize:"2rem"}} className="op">
          Our User's Opinions 
        </div>
        <div style={{ width: "40%" }} className="w-90 max-w-5xl mx-auto mt-10 px-4">
          <Slider {...settings}>
            {slides.map((slide, index) => (
              <div key={index}>
                <div className="relative h-[400px] overflow-hidden rounded-xl">
                  <img style={{ objectFit: "cover", backgroundSize: "cover" }}
                    src={slide.image}

                    className="w-full opacity-0.6 h-full object-cover"
                  />
                  {/* Overlay */}
                  <div className="absolute inset-0 bg-opacity-40 flex items-end">
                    <p className="text-white text-xl md:text-2xl font-semibold p-6">
                      ''{slide.title}''
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </Slider>
        </div>

      </div>
    </>
  );
}
