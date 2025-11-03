

import Slider from "react-slick";

import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import Link from "next/link";
import MySlider from "./components/MySlider";

export default async function Home() {






  const slides = [
    { image: "./images/pexels-kindelmedia-8566526.jpg", title: "Your best AI Mental health assistant" },
    { image: "./images/pexels-pixabay-208147.jpg", title: "Identify your mood using emoji's" },
    { image: "./images/pexels-tara-winstead-8849295.jpg", title: "Check your level of Depression" },
  ];

  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 2,
    slidesToScroll: 1,
    autoplay: true,
    autoplaySpeed: 3000,
    arrows: true,
  };


  return (
    <>

      <div className="container">

        <div className="hero">
          <div style={{textAlign: "center", marginTop: "2rem"}} className="hero-text">
            <h1>Trusted by Millions of Users</h1>
            <p>Join our community of over 1 million users who trust us for their mental health journey.</p>
            <div>

              <button className="btn"><Link href="/dashboard">Get Started</Link></button>
              <button className="btn1"><Link href="/about">Learn More</Link></button>
            </div>

          </div>
        </div>

        <div className="main">

        </div>

      </div>
      <div className="slides" >
        {slides.map((slide, index) => (
          <div
            key={index}
            className="group relative overflow-hidden rounded-lg shadow-md h-64 w-full cursor-pointer"

          >
            <img
              src={slide.image}
              alt={slide.title} style={{ width: "40rem", height: "100%" }}

            />

            <div className="absolute inset-0 bg-black bg-opacity-10 opacity-0.5 group-hover:opacity-100 transition duration-500 flex items-center justify-center">
              <p className="text-white text-lg font-semibold">{slide.title}</p>
            </div>
          </div>
        ))}

      </div>

      <div style={{display:"flex",justifyContent:"center",alignItems:"center"}}>

      <div className="men">
        <div  className="cont">
          <h2>How it Works?</h2>
          <p>Our AI mental health assistant uses advanced algorithms to analyze your journal entries
            and provide personalized insights and advice.</p>
        </div>
      </div>
      </div>

      <div>
        <MySlider />
      </div>



    </>
  );
}