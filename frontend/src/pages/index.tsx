import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <section className="w-full min-h-screen py-12 md:py-24 lg:py-32 xl:py-48 bg-[#d1f7e7]">
      <div className="container flex flex-col items-center px-4 space-y-4 md:px-6 md:space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold tracking-tighter text-green-800 sm:text-4xl md:text-5xl">
            We're here to support your lifestyle.
          </h1>
          <p className="mx-auto max-w-[700px] text-gray-500 md:text-xl dark:text-gray-400">
            Tell us about your habits, diet, age, genetic predispositions,
            satisfaction with social life, etc.
          </p>
        </div>
        <div className="w-full max-w-sm space-y-2">
          <form className="flex flex-col min-h-[200px] space-y-2">
            <Textarea placeholder="Enter your response" required />
            <Button className="bg-green-800 text-white" type="submit">
              Get my suggestions
            </Button>
          </form>
        </div>
      </div>
    </section>
  );
}
