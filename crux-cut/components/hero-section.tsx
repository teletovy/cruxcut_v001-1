import { Button } from "@/components/ui/button"
import { ArrowRight, Link } from "lucide-react"

export function HeroSection() {
  return (
    <section className="relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-purple-900/20 to-black z-0"></div>
      <div
        className="absolute inset-0 z-0 opacity-30"
        style={{
          backgroundImage: "url('/placeholder.svg?height=1080&width=1920')",
          backgroundSize: "cover",
          backgroundPosition: "center",
        }}
      ></div>
      <div className="container mx-auto px-4 py-24 md:py-32 lg:py-40 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
            <span className="block">Focus your climb, and movement.</span>
            <span className="block mt-2">당신의 등반과 동작에 집중하세요.</span>
          </h1>
          <p className="text-lg md:text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            고정된 카메라로 광각 영상을 촬영하기만 하세요. CruxCut의 AI가 당신만을 따라다니는 역동적인 시점의 영상을
            자동으로 생성해 드립니다. 더 이상 촬영자가 필요 없습니다.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button 
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-lg px-8 py-6">
              지금 시작하기
            </Button>
            <Button
              variant="outline"
              className="border-gray-700 text-gray-300 hover:text-white hover:border-gray-500 text-lg px-8 py-6"
            >
              {/* <a href="https://demo.cruxcut.com" target="_blank" rel="noopener noreferrer" className="flex items-center"> */}
              데모 보기 <ArrowRight className="ml-2 h-5 w-5" />
              {/* </a> */}
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}
