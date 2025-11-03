export default function AboutPage() {
  return (
    <main className="min-h-screen bg-gray-50 text-gray-800 p-6">
      <section className="max-w-4xl mx-auto text-center py-10">
        <h1 className="text-4xl font-bold mb-4 text-blue-600">About Us</h1>
        <p className="text-lg text-gray-600 mb-6">
          Your safe space to reflect, heal, and grow — with the help of AI.
        </p>
      </section>

      <section className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-10 items-center">
        <div>
          {/* <img
            src="/images/mental-health-support.jpg"
            alt="Mental Health Support"
            className="rounded-lg shadow-md w-full object-cover"
          /> */}
        </div>
        <div>
          <h2 className="text-2xl font-semibold mb-3">Why We Built This</h2>
          <p className="mb-4">
            Mental health matters — and we believe everyone deserves a tool that helps them
            process thoughts, track emotions, and feel supported, even when they’re alone.
          </p>
          <p>
            This AI-powered journal was designed to make mental health support more
            accessible, reflective, and personal using modern technology.
          </p>
        </div>
      </section>

      <section className="max-w-5xl mx-auto mt-16 grid grid-cols-1 md:grid-cols-2 gap-10 items-center">
        <div>
          <h2 className="text-2xl font-semibold mb-3">What Makes Us Different</h2>
          <ul className="list-disc pl-5 space-y-2">
            <li> Get detailed analysis of your problem and put forward solutions</li>
            <li> AI-generated reflections using Mistral LLM via OpenRouter</li>
            <li> Mood & depression tracking with charts</li>
            <li> Privacy-first design with user-specific entries</li>
          </ul>
        </div>
        <div>
          
        </div>
      </section>

      <section className="max-w-4xl mx-auto text-center mt-20">
        <h2 className="text-2xl font-semibold mb-2">Our Vision</h2>
        <p className="text-gray-600">
          We want to empower individuals to take control of their emotional well-being
          through reflective journaling
        </p>
      </section>

      <footer className="text-center mt-20 text-sm text-gray-500">
        &copy; {new Date().getFullYear()} Multimodal Mental Disorder Prediction System
      </footer>
    </main>
  );
}
